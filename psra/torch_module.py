"""
PyTorch implementation of Parametric Smooth Rational Activation (P-SRA)
with box-constrained PGD projection and elastic initialization anchoring.
"""

import torch
import torch.nn as nn


class PSRA(nn.Module):
    """
    Parametric Smooth Rational Activation (P-SRA) for PyTorch.

    Forward definition:
        f(z; alpha, beta, gamma) = (alpha * z) / (1 + beta * z)   if z >= 0
                                   (gamma * z) / (1 - gamma * z)  if z < 0
    """

    def __init__(
        self,
        num_features: int,
        alpha0: float = 1.0,
        beta0: float = 0.08,
        gamma0: float = 0.35,
        wd_act: float = 1e-3,
        layerwise: bool = False,
    ):
        super().__init__()
        shape = (1, 1) if layerwise else (1, num_features)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        self.wd_act = wd_act

        self.alpha = nn.Parameter(torch.full(shape, alpha0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.full(shape, beta0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full(shape, gamma0, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_pos = torch.clamp(z, min=0.0)
        z_neg = torch.clamp(z, max=0.0)
        pos_mask = z >= 0.0

        out_pos = (self.alpha * z_pos) / (1.0 + self.beta * z_pos)
        out_neg = (self.gamma * z_neg) / (1.0 - self.gamma * z_neg)
        return torch.where(pos_mask, out_pos, out_neg)

    def elastic_anchor_loss(self) -> torch.Tensor:
        """Elastic regularization term toward initial shape parameters."""
        return 0.5 * self.wd_act * (
            torch.sum((self.alpha - self.alpha0) ** 2)
            + torch.sum((self.beta - self.beta0) ** 2)
            + torch.sum((self.gamma - self.gamma0) ** 2)
        )

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Box-constrained PGD step to be called immediately after optimizer.step()."""
        self.alpha.clamp_(0.2, 2.5)
        self.beta.clamp_(0.05, 2.0)
        self.gamma.clamp_(0.05, 1.5)
