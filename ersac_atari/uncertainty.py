
from typing import List
import torch


def worst_case_box(q_list: List[torch.Tensor], pi_logits: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise min across ensemble members (box set)."""
    q_stack = torch.stack(q_list, dim=0)  # (N, B, A)
    q_min = q_stack.min(dim=0).values     # (B, A)
    return q_min


def worst_case_ellipsoid(mu: torch.Tensor,
                         Sigma: torch.Tensor,
                         pi_logits: torch.Tensor,
                         radius: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    q*(s) = mu(s) - r * (Sigma(s) @ pi) / || Sigma(s)^{1/2} pi ||,
    where r is the ellipsoid radius (per-sample or scalar).
    Implements the closed-form from the paper's Appendix (ellipsoidal set).
    """
    pi = torch.softmax(pi_logits, dim=-1)               # (B, A)
    # v = pi^T Sigma pi
    v = torch.einsum('bi,bij,bj->b', pi, Sigma, pi)     # (B,)
    denom = torch.sqrt(torch.clamp(v, min=eps))         # (B,)
    Sig_pi = torch.einsum('bij,bj->bi', Sigma, pi)      # (B, A)
    r = radius if radius.dim() > 0 else radius.expand(denom.shape[0])
    coeff = (r / (denom + eps)).unsqueeze(-1)           # (B, 1)
    q_star = mu - coeff * Sig_pi                        # (B, A)
    return q_star
