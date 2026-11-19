"""
Unified 10-Seed Experimental Benchmark for P-SRA (Parametric Smooth Rational Activation).
Reproduces all four tables and three 300-DPI figures reported in the paper on a
single-threaded CPU environment.

Usage:
    python experiments/psra_unified_benchmark.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gzip
import time
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURACIÓN CANÓNICA UNIFICADA
# ==============================================================================

SEEDS_10 = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
SEEDS_FMNIST = [42, 101, 202, 303, 404]

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


# ==============================================================================
# 1. OPTIMIZADOR ADAM + PGD ACOTADO CON ANCLAJE ELÁSTICO
# ==============================================================================

class AdamPGD:
    def __init__(
        self,
        lr_w=0.004,
        lr_act=0.001,
        wd_act=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
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

    def step(self, param_groups):
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


# ==============================================================================
# 2. FUNCIONES DE ACTIVACIÓN (BASELINES, CONTROL ACOTADO Y P-SRA)
# ==============================================================================

class Sigmoid:
    def __init__(self, units):
        self.out = None

    def forward(self, z):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        return self.out

    def backward(self, d_out):
        return d_out * self.out * (1.0 - self.out)

    def get_params(self):
        return []


class ReLU:
    def __init__(self, units):
        self.mask = None

    def forward(self, z):
        self.mask = z > 0.0
        return np.where(self.mask, z, 0.0)

    def backward(self, d_out):
        return np.where(self.mask, d_out, 0.0)

    def get_params(self):
        return []


class PReLU:
    def __init__(self, units, init_a=0.35):
        self.init_a = init_a
        self.a = np.full((1, units), init_a, dtype=np.float64)
        self.d_a = np.zeros_like(self.a)
        self.z = None
        self.mask = None

    def forward(self, z):
        self.z = z
        self.mask = z > 0.0
        return np.where(self.mask, z, self.a * z)

    def backward(self, d_out):
        self.d_a[:] = np.sum(
            d_out * np.where(self.mask, 0.0, self.z), axis=0, keepdims=True
        )
        return d_out * np.where(self.mask, 1.0, self.a)

    def get_params(self):
        return [(self.a, self.d_a, True, None, self.init_a)]


class ClippedPReLU:
    """
    Control experimental: mismas pendientes iniciales (alpha0=1.0, gamma0=0.35)
    y mismo rango acotado [-1.0, alpha0/beta0 = 12.5] que P-SRA Equilibrada,
    pero mediante recorte lineal duro en lugar de curvatura racional suave.
    """

    def __init__(
        self, units, alpha0=1.0, gamma0=0.35, low_clip=-1.0, high_clip=12.5
    ):
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.low_clip = low_clip
        self.high_clip = high_clip

        self.alpha = np.full((1, units), alpha0, dtype=np.float64)
        self.gamma = np.full((1, units), gamma0, dtype=np.float64)
        self.d_alpha = np.zeros_like(self.alpha)
        self.d_gamma = np.zeros_like(self.gamma)
        self.cache = None

    def forward(self, z):
        pos_mask = z >= 0.0
        raw = np.where(pos_mask, self.alpha * z, self.gamma * z)
        out = np.clip(raw, self.low_clip, self.high_clip)
        self.cache = (z, pos_mask, raw)
        return out

    def backward(self, d_out):
        z, pos_mask, raw = self.cache
        active = (raw > self.low_clip) & (raw < self.high_clip)
        d_raw = d_out * active

        d_z = d_raw * np.where(pos_mask, self.alpha, self.gamma)
        self.d_alpha[:] = np.sum(
            d_raw * np.where(pos_mask, z, 0.0), axis=0, keepdims=True
        )
        self.d_gamma[:] = np.sum(
            d_raw * np.where(~pos_mask, z, 0.0), axis=0, keepdims=True
        )
        return d_z

    def get_params(self):
        return [
            (self.alpha, self.d_alpha, True, (0.2, 2.5), self.alpha0),
            (self.gamma, self.d_gamma, True, (0.05, 1.5), self.gamma0),
        ]


class Elliott:
    def __init__(self, units):
        self.denom = None

    def forward(self, z):
        self.denom = 1.0 + np.abs(z)
        return z / self.denom

    def backward(self, d_out):
        return d_out / (self.denom ** 2)

    def get_params(self):
        return []


class GELU:
    def __init__(self, units):
        self.cache = None
        self.c = np.sqrt(2.0 / np.pi)

    def forward(self, z):
        tanh_in = np.tanh(self.c * (z + 0.044715 * (z ** 3)))
        self.cache = (z, tanh_in)
        return 0.5 * z * (1.0 + tanh_in)

    def backward(self, d_out):
        z, tanh_in = self.cache
        sech2 = 1.0 - tanh_in ** 2
        d_in = self.c * (1.0 + 3.0 * 0.044715 * (z ** 2))
        return d_out * (0.5 * (1.0 + tanh_in) + 0.5 * z * sech2 * d_in)

    def get_params(self):
        return []


class PSRA:
    """
    P-SRA Canónica Equilibrada:
    f(z) = (alpha * z)/(1 + beta * z) si z >= 0; (gamma * z)/(1 - gamma * z) si z < 0.
    """

    def __init__(
        self,
        units,
        alpha0=1.0,
        beta0=0.08,
        gamma0=0.35,
        bounds=None,
        layerwise=False,
    ):
        shape = (1, 1) if layerwise else (1, units)
        self.layerwise = layerwise
        self.alpha0, self.beta0, self.gamma0 = alpha0, beta0, gamma0
        self.bounds = bounds or PSRA_CANONICAL["bounds"]

        self.alpha = np.full(shape, alpha0, dtype=np.float64)
        self.beta = np.full(shape, beta0, dtype=np.float64)
        self.gamma = np.full(shape, gamma0, dtype=np.float64)

        self.d_alpha = np.zeros_like(self.alpha)
        self.d_beta = np.zeros_like(self.beta)
        self.d_gamma = np.zeros_like(self.gamma)
        self.cache = None

    def forward(self, z):
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

    def backward(self, d_out):
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
            d_out * (z_neg / (denom_neg ** 2)), axis=ax, keepdims=True
        )
        return d_z

    def get_params(self):
        return [
            (self.alpha, self.d_alpha, True, self.bounds["alpha"], self.alpha0),
            (self.beta, self.d_beta, True, self.bounds["beta"], self.beta0),
            (self.gamma, self.d_gamma, True, self.bounds["gamma"], self.gamma0),
        ]


class PSRA_QuadAblation:
    """Variante de ablación: reparametrización cuadrática (beta = q_b^2, gamma = q_g^2) sin PGD."""

    def __init__(self, units, alpha0=1.0, beta0=0.08, gamma0=0.35):
        self.alpha = np.full((1, units), alpha0, dtype=np.float64)
        self.q_beta = np.full(
            (1, units), np.sqrt(max(beta0 - 1e-4, 1e-4)), dtype=np.float64
        )
        self.q_gamma = np.full(
            (1, units), np.sqrt(max(gamma0 - 1e-3, 1e-4)), dtype=np.float64
        )

        self.d_alpha = np.zeros_like(self.alpha)
        self.d_q_beta = np.zeros_like(self.q_beta)
        self.d_q_gamma = np.zeros_like(self.q_gamma)
        self.cache = None

    def forward(self, z):
        beta
