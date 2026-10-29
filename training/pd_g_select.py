from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Any, Optional
import statistics


def straggler_ratio(lengths: Sequence[float]) -> float:
    """
    Rt = max(L) / median(L)
    lengths: token lengths from G generations at step t
    """
    if not lengths:
        raise ValueError("lengths must be non-empty")
    med = statistics.median(lengths)
    if med == 0:
        raise ValueError("median(lengths) is 0, cannot form ratio")
    return max(lengths) / med


@dataclass
class LagrangianRiskController:
    """
    Algorithm 1: Lagrangian Risk Controller for GRPO (Dynamic-GRPO.pdf)

    - Allowed group sizes: G = {4, 8, 16, 32} (configurable)
    - Beta-Bernoulli belief per group size k: phi_k ~ Beta(alpha_k, beta_k)
      p_hat_k = alpha_k / (alpha_k + beta_k)
    - Primal step: choose G_t = argmax_k [ k - lambda * p_hat_k ]
    - Observe S_t = 1 if Rt > tau else 0, with Rt = max(L)/median(L)
    - Dual step: lambda <- max(0, lambda + eta * (S_t - delta))
    - Belief update for chosen size G_t (with forgetting factor gamma):
        alpha_Gt <- gamma * alpha_Gt + S_t
        beta_Gt  <- gamma * beta_Gt  + (1 - S_t)
    """

    allowed_sizes: Tuple[int, ...] = (4, 8, 16, 32)
    delta: float = 0.15  # target max straggler rate
    eta: float = 0.25  # dual step size
    gamma: float = 0.99  # forgetting factor for Beta stats
    tau: float = 1.25  # straggler threshold (Rt > tau => S_t = 1)

    # Priors for Beta distribution (default to uniform/neutral: alpha=1, beta=1 => p=0.5)
    # init options for alpha, beta per group size k
    # 1. Set alpha > beta to be pessimistic (start assuming high risk)
    # 2. Set alpha < beta to be optimistic (start assuming low risk)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    lam: float = 0.0
    alpha: Dict[int, float] = field(default_factory=dict)
    beta: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.alpha or not self.beta:
            # Initialize with user-defined priors
            self.alpha = {k: self.prior_alpha for k in self.allowed_sizes}
            self.beta = {k: self.prior_beta for k in self.allowed_sizes}

    def risk_estimates(self) -> Dict[int, float]:
        """p_hat_k = alpha_k / (alpha_k + beta_k)"""
        return {
            k: self.alpha[k] / (self.alpha[k] + self.beta[k])
            for k in self.allowed_sizes
        }

    def select_group_size(self) -> Tuple[int, Dict[int, float], Dict[int, float]]:
        """
        Primal step:
          G_t = argmax_k [ k - lam * p_hat_k ]
        Returns (chosen_k, utilities, p_hat)
        """
        p_hat = self.risk_estimates()
        utilities = {k: (k - self.lam * p_hat[k]) for k in self.allowed_sizes}
        # deterministic tie-break: largest utility, then largest k
        chosen = max(self.allowed_sizes, key=lambda k: (utilities[k], k))
        return chosen, utilities, p_hat

    def update_dual(self, S_t: int) -> float:
        """lambda <- max(0, lambda + eta * (S_t - delta))"""
        self.lam = max(0.0, self.lam + self.eta * (float(S_t) - self.delta))
        return self.lam

    def update_belief(self, k: int, S_t: int) -> None:
        """Belief update (only for chosen k): alpha_k, beta_k with forgetting factor gamma."""
        self.alpha[k] = self.gamma * self.alpha[k] + float(S_t)
        self.beta[k] = self.gamma * self.beta[k] + (1.0 - float(S_t))

    def observe_and_update(self, k: int, lengths: Sequence[float]) -> Dict[str, Any]:
        """
        Observe lengths from group size k, calculate S_t, and update internal state.
        Returns intermediate metrics.
        """
        Rt = straggler_ratio(lengths)
        S_t = 1 if Rt > self.tau else 0

        old_lam = self.lam
        new_lam = self.update_dual(S_t)
        self.update_belief(k, S_t)

        return {
            "Rt": Rt,
            "S_t": S_t,
            "lambda_before": old_lam,
            "lambda_after": new_lam,
        }
