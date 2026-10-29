from typing import Callable, List, Dict, Iterable, Optional, Sequence, Any
from collections import deque
from dataclasses import dataclass
from transformers import TrainerCallback
import math
import random
import statistics as stats
from varrs import L_MAX, DELTA, E, PG_TARGET, ALLOWED_G, HIST_WINDOW
import logging
from select_g_methods import estimate_tail_prob, choose_G_hybrid
from pd_g_select import LagrangianRiskController, straggler_ratio

# setup file for logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# ---------------------------


class UpdateGroupSizeCallback(TrainerCallback):
    def __init__(
        self,
        trainer,
        chooser: Callable[[], int],
        *,
        every_n_steps: int = 16,
    ):
        """Periodically updates num_generations using a chooser.

        - every_n_steps: throttle decisions to once every N update steps.
        """
        super().__init__()
        self.trainer = trainer
        self.chooseG = chooser
        self.every_n_steps = max(1, int(every_n_steps))
        self._last_applied_step = -1

    def on_optimizer_step(self, args, state, control, **kwargs):
        # Global step counts completed update steps (after accumulation). This hook
        # runs once per update step, so throttle by modulo and warmup.
        step = int(getattr(state, "global_step", 0) or 0)
        if self._last_applied_step == step:
            return control
        if (step % self.every_n_steps) != 0 and step <= 0:
            return control

        G = int(self.chooseG())
        self.trainer.num_generations = G
        args.num_generations = G
        self._last_applied_step = step
        print(f"[UpdateGroupSizeCallback] Step {step}: set num_generations = {G}")
        return control


# ---------------------------
# 1) Metrics per group (three definitions)
# ---------------------------
def group_metrics(
    group: List[int], L_max: int = L_MAX, delta: int = DELTA
) -> Dict[str, float]:
    # 1) cap-hit rate (fraction near cap)
    h = sum(1 for l in group if l >= L_max - delta) / max(1, len(group))
    # 2) idle inefficiency (how much others wait for the max)
    m = max(group) if group else 1
    w = 1.0 - (sum(group) / (max(1, len(group)) * m))
    # 3) straggler index (tail vs central tendency)
    med = stats.median(group) if group else 1.0
    s = (m / med) if med > 0 else float("inf")
    return {
        "cap_hit_rate": h,
        "idle_inefficiency": w,
        "straggler_index": s,
        "max_len": m,
        "median_len": med,
    }


def batch_metrics(
    groups: List[List[int]], L_max: int = L_MAX, delta: int = DELTA
) -> Dict[str, float]:
    gm = [group_metrics(g, L_max, delta) for g in groups]
    # simple averages across groups
    avg = lambda k: sum(d[k] for d in gm) / max(1, len(gm))
    return {
        "avg_cap_hit_rate": avg("cap_hit_rate"),
        "avg_idle_inefficiency": avg("idle_inefficiency"),
        "avg_straggler_index": avg("straggler_index"),
        "max_of_max": max((d["max_len"] for d in gm), default=0),
        "median_of_medians": stats.median([d["median_len"] for d in gm]) if gm else 0.0,
    }


# ---------------------------
# 2) Threshold T from history (min(L_MAX, P99))
# ---------------------------
def percentile(data: Iterable[int], q: float) -> float:
    xs = sorted(data)
    if not xs:
        return 0.0
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return float(xs[int(i)])
    return xs[lo] * (hi - i) + xs[hi] * (i - lo)


def pG(p: float, G: int) -> float:
    return 1.0 - (1.0 - p) ** G


def pick_threshold(
    length_history: List[int],
    L_max: int = L_MAX,
    q: float = 0.95,
    delta: int = DELTA,
) -> int:
    """Use a high percentile but require lengths to also be near the cap."""
    if not length_history:
        return max(0, L_max - delta)
    pct = percentile(length_history, q)
    # Only treat near-cap groups as stragglers by enforcing a hard margin.
    return int(min(L_max, max(L_max - delta, pct)))


@dataclass
class GroupSizeController:
    g_min: int = 2
    g_max: int = 16
    n: int = 60
    tau: float = 2.0
    p_threshold: float = 0.05
    step_up: int = 2
    step_down: int = 2
    min_samples: int = 60
    hysteresis: float = 0.0

    def _median(self, xs: Sequence[float]) -> float:
        s = sorted(xs)
        m = len(s)
        mid = m // 2
        if m % 2 == 1:
            return float(s[mid])
        return 0.5 * (s[mid - 1] + s[mid])

    def is_straggler_group(self, lengths: Sequence[int]) -> bool:
        if not lengths:
            raise ValueError("lengths must be non-empty")
        med = self._median(lengths)
        if med <= 0:
            return any(l > 0 for l in lengths)
        mx = max(lengths)
        return (mx / med) > self.tau

    def update_group_size(self, current_g: int, history: Sequence[bool]) -> int:
        if current_g < self.g_min or current_g > self.g_max:
            raise ValueError(f"current_g must be within [{self.g_min}, {self.g_max}]")

        if len(history) < self.min_samples:
            return current_g

        window = history[-self.n :] if len(history) >= self.n else history
        k = sum(1 for x in window if x)
        p_hat = k / len(window)

        thr_up = self.p_threshold - self.hysteresis
        thr_down = self.p_threshold + self.hysteresis
        print(f"p_hat: {p_hat:.4f}, thr_up: {thr_up:.4f}, thr_down: {thr_down:.4f}")

        if p_hat <= thr_up:
            return min(self.g_max, current_g + self.step_up)
        if p_hat >= thr_down:
            return max(self.g_min, current_g - self.step_down)

        return current_g


def choose_G_straggler(
    groups: List[List[int]],
    current_G: int,
    straggler_history: Dict[int, deque],
    controller: Optional["GroupSizeController"] = None,
    allowed: Iterable[int] = ALLOWED_G,
) -> int:
    """Straggler-driven step rule based on a sliding window of group flags."""
    ctl = controller or GroupSizeController(
        g_min=min(ALLOWED_G), g_max=max(ALLOWED_G), p_threshold=PG_TARGET
    )
    ladder = sorted(g for g in allowed if ctl.g_min <= g <= ctl.g_max)
    if not ladder:
        return int(current_G)
    try:
        idx = ladder.index(int(current_G))
    except ValueError:
        idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - int(current_G)))

    hist = straggler_history.setdefault(int(ladder[idx]), deque())
    for g in groups:
        try:
            flag = ctl.is_straggler_group(g)
        except Exception:
            flag = False
        hist.append(bool(flag))
    while len(hist) > ctl.n:
        hist.popleft()

    window = list(hist)[-ctl.n :]
    if len(window) < ctl.min_samples:
        return int(ladder[idx])

    k = sum(1 for x in window if x)
    p_hat = k / len(window)
    thr_up = ctl.p_threshold - ctl.hysteresis
    thr_down = ctl.p_threshold + ctl.hysteresis

    if p_hat <= thr_up and idx < len(ladder) - 1:
        idx = min(idx + ctl.step_up, len(ladder) - 1)
    elif p_hat >= thr_down and idx > 0:
        idx = max(idx - ctl.step_down, 0)

    return int(ladder[idx])


# ---------------------------
# 4) Control logic for one step
#     - compute metrics for current batch
#     - update history
#     - compute T, p, choose G_next
# ---------------------------
def control_step(
    groups: List[List[int]],
    length_history: List[int],
    straggler_history: Optional[Dict[int, deque]] = None,
    controller: Optional["GroupSizeController"] = None,
    current_G: int = ALLOWED_G[0],
    E_eff: int = E,
    L_max: int = L_MAX,
    delta: int = DELTA,
    target_pG: float = PG_TARGET,
    q: float = 0.99,
) -> Dict[str, object]:
    # metrics on this batch
    bm = batch_metrics(groups, L_max, delta)

    # update history with per-group maxima so the history length reflects groups seen
    group_maxes = [max(g) for g in groups if g]
    for gm in group_maxes:
        length_history.append(int(gm))
    if len(length_history) > HIST_WINDOW:
        del length_history[: len(length_history) - HIST_WINDOW]

    # threshold and tail prob
    T = pick_threshold(length_history, L_max, q=q, delta=delta)
    p = estimate_tail_prob(length_history, T)

    # choose next G using straggler feedback plus hybrid safety
    straggler_history = straggler_history if straggler_history is not None else {}
    G_next = choose_G_straggler(
        groups,
        current_G=current_G,
        straggler_history=straggler_history,
        controller=controller,
        allowed=ALLOWED_G,
    )

    return {
        "batch_metrics": bm,
        "T": T,
        "tail_prob_p": p,
        "G_next": G_next,
        "pG_next": pG(p, G_next),
    }

# ---------------------------
# 5) PD Control logic for one step (Alternative)
# ---------------------------
def control_step_pd(
    groups: List[List[int]],
    current_G: int,
    controller: Optional["LagrangianRiskController"] = None,
    L_max: int = L_MAX,
    delta: int = DELTA,
    # arguments below are for signature compatibility if needed, though unused
    length_history: Optional[List[int]] = None, 
    straggler_history: Optional[Any] = None,
    **kwargs
) -> Dict[str, object]:
    """
    Control step using the Primal-Dual Lagrangian Risk Controller.
    groups: list of generated token lengths per group
    controller: instance of LagrangianRiskController (should be persistent)
    current_G: the group size used for these generations
    """
    # 1. Standard batch metrics
    bm = batch_metrics(groups, L_max, delta)

    # 2. Update controller with observations
    if controller is None:
        # Initialize with project defaults if not provided
        controller = LagrangianRiskController(allowed_sizes=ALLOWED_G)
        controller.__post_init__()

    # Collect all lengths from the current step
    all_lengths = []
    for g in groups:
        all_lengths.extend(g)

    # If no data, keep current G
    if not all_lengths:
        return {
            "batch_metrics": bm,
            "G_next": current_G,
            "p_hat": {},
            "utilities": {},
            "lam": controller.lam if controller else 0.0,
        }

    # Calculate average failure rate (S_t) across groups instead of aggregate max/median
    # This prevents sample size bias when aggregating many steps.
    failures = 0.0
    valid_groups = 0
    for g in groups:
        if not g: 
            continue
        valid_groups += 1
        # Check individual group for straggler behavior
        # straggler_ratio is defined in pd_g_select.py as max(L)/median(L)
        try:
            r = straggler_ratio(g)
            if r > controller.tau:
                failures += 1.0
        except ValueError:
            pass # skip empty/invalid groups
            
    avg_S_t = (failures / valid_groups) if valid_groups > 0 else 0.0

    # Update state
    # Note: control_step_pd assumes the controller instance is mutable and persistent across steps
    update_info = controller.observe_and_update(current_G, [], passed_S_t=avg_S_t)

    # 3. Select next G
    G_next, utilities, p_hat = controller.select_group_size()

    return {
        "batch_metrics": bm,
        "G_next": G_next,
        "update_info": update_info,
        "p_hat": p_hat,
        "utilities": utilities,
        "lam": controller.lam,
    }
