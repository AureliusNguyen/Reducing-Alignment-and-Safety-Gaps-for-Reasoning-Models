from typing import Callable, List, Dict, Iterable, Optional
from varrs import L_MAX, DELTA, E, PG_TARGET, ALLOWED_G, HIST_WINDOW
import math


# ---------------------------
# 3) p = Pr(L >= T) and G selection targeting p_G <= PG_TARGET
#     p_G = 1 - (1 - p)^G
# ---------------------------
def estimate_tail_prob(length_history: List[int], T: int) -> float:
    if not length_history:
        return 0.0
    hits = sum(1 for l in length_history if l >= T)
    return hits / len(length_history)


def pG(p: float, G: int) -> float:
    return 1.0 - (1.0 - p) ** G


def choose_G_from_idle(
    idle_inefficiency: float,
    *,
    allowed: Iterable[int] = ALLOWED_G,
    E_eff: int = E,
    very_low: float = 0.14,
    low: float = 0.22,
    high: float = 0.30,
) -> int:
    """Suggest G from idle inefficiency (padding waste fraction).

    Heuristic tiers (lower idle -> larger G):
      - idle <= very_low: highest rung
      - very_low < idle <= low: next rung down
      - low < idle <= high: next rung down
      - idle > high: smallest rung
    """
    ladder = ALLOWED_G
    n = max(1, len(ladder))
    if idle_inefficiency <= very_low:
        return ladder[-1]
    if idle_inefficiency <= low:
        return ladder[-2] if n >= 2 else ladder[-1]
    if idle_inefficiency <= high:
        return ladder[-3] if n >= 3 else (ladder[0] if n == 1 else ladder[-2])
    return ladder[0]


def choose_G_from_straggler(
    straggler_index: float,
    *,
    allowed: Iterable[int] = ALLOWED_G,
    E_eff: int = E,
    s_low: float = 1.10,
    s_mid: float = 1.20,
    s_high: float = 1.40,
) -> int:
    """Suggest G from straggler index (max/median).

    Heuristic tiers (lower s -> larger G):
      - s <= s_low: highest rung
      - s_low < s <= s_mid: next rung down
      - s_mid < s <= s_high: next rung down
      - s > s_high: smallest rung
    """
    ladder = ALLOWED_G
    n = max(1, len(ladder))
    s = float(straggler_index)
    if s <= s_low:
        return ladder[-1]
    if s <= s_mid:
        return ladder[-2] if n >= 2 else ladder[-1]
    if s <= s_high:
        return ladder[-3] if n >= 3 else (ladder[0] if n == 1 else ladder[-2])
    return ladder[0]


def choose_G_from_cap_hit(
    cap_hit_rate: float,
    *,
    allowed: Iterable[int] = ALLOWED_G,
    E_eff: int = E,
    very_low: float = 0.001,
    low: float = 0.005,
    high: float = 0.020,
) -> int:
    """Suggest G from near-cap hit rate (fraction >= L_MAX - delta).

    Heuristic tiers (lower cap_hit -> larger G):
      - cap_hit <= very_low: highest rung
      - very_low < cap_hit <= low: next rung down
      - low < cap_hit <= high: next rung down
      - cap_hit > high: smallest rung
    """
    ladder = ALLOWED_G
    n = max(1, len(ladder))
    r = float(cap_hit_rate)
    if r <= very_low:
        return ladder[-1]
    if r <= low:
        return ladder[-2] if n >= 2 else ladder[-1]
    if r <= high:
        return ladder[-3] if n >= 3 else (ladder[0] if n == 1 else ladder[-2])
    return ladder[0]


def choose_G(
    p: float, allowed: Iterable[int] = ALLOWED_G, target_pG: float = PG_TARGET
) -> int:
    """Pick the largest G whose pG <= target (maximize throughput under constraint).

    If no value is feasible, fall back to the smallest allowed to remain safe.
    """
    candidates = sorted(g for g in allowed if g >= 2 and g % 2 == 0)
    feasible = [g for g in candidates if pG(p, g) <= target_pG]
    return max(feasible) if feasible else min(candidates)


def choose_G_hybrid(
    batch_metrics: Dict[str, float],
    tail_prob: float,
    *,
    allowed: Iterable[int] = ALLOWED_G,
    target_pG: float = PG_TARGET,
) -> int:
    """Combine dispersion heuristics with a tail-probability safety constraint."""
    ladder = sorted(g for g in allowed if g >= 2 and g % 2 == 0)
    if not ladder:
        raise ValueError("Allowed group sizes must include at least one even >=2.")

    heuristic_choices: List[int] = []
    # idle = batch_metrics.get("avg_idle_inefficiency")
    # if idle is not None:
    #     heuristic_choices.append(choose_G_from_idle(idle))
    straggler = batch_metrics.get("avg_straggler_index")
    if straggler is not None:
        heuristic_choices.append(choose_G_from_straggler(straggler))
    # cap_hit = batch_metrics.get("avg_cap_hit_rate")
    # if cap_hit is not None:
    #     heuristic_choices.append(choose_G_from_cap_hit(cap_hit))

    preferred = min(heuristic_choices)

    candidates = [g for g in ladder if g <= preferred]
    if not candidates:
        candidates = ladder

    for g in sorted(candidates, reverse=True):
        if pG(tail_prob, g) <= target_pG:
            return g

    for g in reversed(ladder):
        if pG(tail_prob, g) <= target_pG:
            return g
    return ladder[0]
