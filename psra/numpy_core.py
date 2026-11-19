"""
Core NumPy implementation of Parametric Smooth Rational Activation (P-SRA)
and box-constrained Adam with Projected Gradient Descent (AdamPGD).
"""

import numpy as np


PSRA_CANONICAL = {
    "alpha0": 1.0,
    "beta0": 0.08,
    "gamma0": 0.35,
    "bounds": {
        "alpha": (0.2, 2.5),
        "beta": (0.05, 2.0),
        "gamma": (0.05, 1.5),
    },
    "wd_act": 1e-3,
}


class AdamPGD:
    """
    Adam optimizer supporting:
    - Differentiated learning rates for synaptic weights (lr_w) and activation
      shape parameters (lr_act).
    - Elastic initialization anchoring (wd_act) to prevent parameter drift on
      infrequently activated branches.
    - Direct box-constrained Projected Gradient Descent (PGD) after each step.
    """

    def __init__(
        self,
        lr_w: float = 0.004,
        lr_act: float = 0.001,
        wd_act: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr_w = lr_w
        self.lr_act = lr_act
        self.wd_act = wd_act
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, param_groups) -> None:
        """
        Updates parameters in-place.

        Each entry in param_groups must be a 5-tuple:
            (param, grad, is_act, bounds, init_val)
        """
        self.t += 1
        for idx, (param, grad, is_act, bounds, init_val) in enumerate(param_groups):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(param)
                self.v[idx] = np.zeros_like(param)

            if is_act and self.wd_act > 0.0 and init_val is not None:
                eff_grad = grad + self.wd_act * (param - init_val)
            else:
                eff_grad = grad

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * eff_grad
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (eff_grad ** 2)

            m_hat = self.m[idx] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1.0 - self.beta2 ** self.t)

            lr = self.lr_act if is_act else self.lr_w
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if bounds is not None:
                low, high = bounds
                if low is not None and high is not None:
                    np.clip(param, low, high, out=param)
                elif low is not None:
                    np.maximum(param, low, out=param)
                elif high is not None:
                    np.minimum(param, high, out=param)


class PSRA:
    """
    Parametric Smooth Rational Activation (P-SRA) in pure NumPy.

    Forward definition:
        f(z; alpha, beta, gamma) = (alpha * z) / (1 + beta * z)   if z >= 0
                                   (gamma * z) / (1 - gamma * z)  if z < 0
    """

    def __init__(
        self,
        units: int,
        alpha0: float = 1.0,
        beta0: float = 0.08,
        gamma0: float = 0.35,
        bounds: dict = None,
        layerwise: bool = False,
    ):
        shape = (1, 1) if layerwise else (1, units)
        self.units = units
        self.layerwise = layerwise
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        self.bounds = bounds or PSRA_CANONICAL["bounds"]

        self.alpha = np.full(shape, alpha0, dtype=np.float64)
        self.beta = np.full(shape, beta0, dtype=np.float64)
        self.gamma = np.full(shape, gamma0, dtype=np.float64)

        self.d_alpha = np.zeros_like(self.alpha)
        self.d_beta = np.zeros_like(self.beta)
        self.d_gamma = np.zeros_like(self.gamma)
        self.cache = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        z_pos = np.maximum(z, 0.0)
        z_neg = np.minimum(z, 0.0)
        pos_mask = z >= 0.0

        denom_pos = 1.0 + self.beta * z_pos
        denom_neg = 1.0 - self.gamma * z_neg

        out = np.where(
            pos_mask,
            (self.alpha * z_pos) / denom_pos,
            (self.gamma * z_neg) / denom_neg,
        )
        self.cache = (z_pos, z_neg, pos_mask, denom_pos, denom_neg)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        z_pos, z_neg, pos_mask, denom_pos, denom_neg = self.cache

        grad_z = np.where(
            pos_mask,
            self.alpha / (denom_pos ** 2),
            self.gamma / (denom_neg ** 2),
        )
        d_z = d_out * grad_z

        ax = None if self.layerwise else 0
        self.d_alpha[:] = np.sum(d_out * (z_pos / denom_pos), axis=ax, keepdims=True)
        self.d_beta[:] = np.sum(
            d_out * (-self.alpha * (z_pos ** 2) / (denom_pos ** 2)),
            axis=ax,
            keepdims=True,
        )
        self.d_gamma[:] = np.sum(
            d_out * (z_neg / (denom_neg ** 2)),
            axis=ax,
            keepdims=True,
        )
        return d_z

    def get_params(self) -> list:
        return [
            (self.alpha, self.d_alpha, True, self.bounds["alpha"], self.alpha0),
            (self.beta, self.d_beta, True, self.bounds["beta"], self.beta0),
            (self.gamma, self.d_gamma, True, self.bounds["gamma"], self.gamma0),
        ]
