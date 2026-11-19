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
        beta = (self.q_beta ** 2) + 1e-4
        gamma = (self.q_gamma ** 2) + 1e-3

        z_pos = np.maximum(z, 0.0)
        z_neg = np.minimum(z, 0.0)
        pos_mask = z >= 0.0

        denom_pos = 1.0 + beta * z_pos
        denom_neg = 1.0 - gamma * z_neg

        out = np.where(
            pos_mask,
            (self.alpha * z_pos) / denom_pos,
            (gamma * z_neg) / denom_neg,
        )
        self.cache = (z_pos, z_neg, pos_mask, denom_pos, denom_neg, beta, gamma)
        return out

    def backward(self, d_out):
        z_pos, z_neg, pos_mask, denom_pos, denom_neg, beta, gamma = self.cache
        grad_z = np.where(
            pos_mask,
            self.alpha / (denom_pos ** 2),
            gamma / (denom_neg ** 2),
        )
        d_z = d_out * grad_z

        self.d_alpha[:] = np.sum(d_out * (z_pos / denom_pos), axis=0, keepdims=True)
        d_beta = np.sum(
            d_out * (-self.alpha * (z_pos ** 2) / (denom_pos ** 2)),
            axis=0,
            keepdims=True,
        )
        d_gamma = np.sum(
            d_out * (z_neg / (denom_neg ** 2)), axis=0, keepdims=True
        )

        self.d_q_beta[:] = d_beta * (2.0 * self.q_beta)
        self.d_q_gamma[:] = d_gamma * (2.0 * self.q_gamma)
        return d_z

    def get_params(self):
        return [
            (self.alpha, self.d_alpha, True, None, None),
            (self.q_beta, self.d_q_beta, True, None, None),
            (self.q_gamma, self.d_q_gamma, True, None, None),
        ]


# ==============================================================================
# 3. ARQUITECTURA MLP Y UTILIDADES DE EVALUACIÓN
# ==============================================================================

class MLP:
    def __init__(
        self, in_dim, hidden_dims, out_dim, act_factory, seed, init_mode="he"
    ):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.W, self.b, self.dW, self.db, self.acts = [], [], [], [], []
        self.inputs = []

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            if i < len(dims) - 2 and init_mode == "he":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            w = rng.normal(0.0, scale, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.W.append(w)
            self.b.append(b)
            self.dW.append(np.zeros_like(w))
            self.db.append(np.zeros_like(b))
            if i < len(dims) - 2:
                self.acts.append(act_factory(fan_out))

    def forward(self, X):
        self.inputs = [X]
        h = X
        for i, act in enumerate(self.acts):
            z = h @ self.W[i] + self.b[i]
            h = act.forward(z)
            self.inputs.append(h)
        return h @ self.W[-1] + self.b[-1]

    def backward(self, d_loss):
        self.dW[-1][:] = self.inputs[-1].T @ d_loss
        self.db[-1][:] = np.sum(d_loss, axis=0, keepdims=True)
        d_h = d_loss @ self.W[-1].T

        for i in reversed(range(len(self.acts))):
            d_z = self.acts[i].backward(d_h)
            self.dW[i][:] = self.inputs[i].T @ d_z
            self.db[i][:] = np.sum(d_z, axis=0, keepdims=True)
            if i > 0:
                d_h = d_z @ self.W[i].T

    def all_param_groups(self):
        groups = []
        for i in range(len(self.W)):
            groups.append((self.W[i], self.dW[i], False, None, None))
            groups.append((self.b[i], self.db[i], False, None, None))
        for act in self.acts:
            groups.extend(act.get_params())
        return groups


def softmax_ce_loss(logits, y_int, num_classes):
    m = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    y_oh = np.eye(num_classes)[y_int]
    loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
    d_logits = (probs - y_oh) / m
    return loss, d_logits


def inject_outliers(X_test, seed, mode="pos", sigma=10.0, frac=0.25, n_cols=3):
    rng = np.random.default_rng(seed + 777)
    X_c = X_test.copy()
    n_samples, n_features = X_c.shape
    rows = rng.choice(n_samples, size=max(1, int(n_samples * frac)), replace=False)
    for r in rows:
        cols = rng.choice(n_features, size=min(n_cols, n_features), replace=False)
        if mode == "pos":
            signs = np.ones(len(cols))
        elif mode == "neg":
            signs = -np.ones(len(cols))
        else:
            signs = rng.choice([-1.0, 1.0], size=len(cols))
        X_c[r, cols] += signs * sigma
    return X_c


# ==============================================================================
# TABLA 1: BENCHMARK TABULAR UNIFICADO (10 SEMILLAS, 3 DATASETS + CONTROL)
# ==============================================================================

def run_unified_tabular_benchmark():
    print("=" * 116)
    print("TABLA 1: BENCHMARK TABULAR UNIFICADO (10 SEMILLAS | INCLUYE CONTROL CLIPPED PReLU)")
    print("=" * 116)

    wine = load_wine()
    bc = load_breast_cancer()
    diab = load_diabetes()

    datasets = [
        {
            "name": "Wine (Clasif. | 13 in | Acc % ↑)",
            "key": "wine",
            "X": wine.data,
            "y": wine.target,
            "task": "cls",
            "out_dim": 3,
            "epochs": 300,
        },
        {
            "name": "Breast Cancer (Clasif. | 25 in | Acc % ↑)",
            "key": "bc",
            "X": bc.data[:, :25],
            "y": bc.target,
            "task": "cls",
            "out_dim": 2,
            "epochs": 300,
        },
        {
            "name": "Diabetes (Regresión | 10 in | MSE ↓)",
            "key": "diabetes",
            "X": diab.data,
            "y": diab.target.reshape(-1, 1),
            "task": "reg",
            "out_dim": 1,
            "epochs": 300,
        },
    ]

    models = {
        "ReLU": lambda u: ReLU(u),
        "PReLU": lambda u: PReLU(u, init_a=0.35),
        "Clipped PReLU (Control)": lambda u: ClippedPReLU(
            u, alpha0=1.0, gamma0=0.35, low_clip=-1.0, high_clip=12.5
        ),
        "Elliott": lambda u: Elliott(u),
        "GELU": lambda u: GELU(u),
        "P-SRA Equilibrada (Ours)": lambda u: PSRA(
            u, alpha0=1.0, beta0=0.08, gamma0=0.35
        ),
    }

    # Warm-up BLAS/NumPy para evitar latencia de inicialización en el primer modelo
    _w = MLP(10, [48, 32], 1, lambda u: ReLU(u), seed=0)
    _w.forward(np.ones((32, 10)))
    _w.backward(np.ones((32, 1)))

    pct_acc = np.zeros((len(SEEDS_10), 300, 2, 3, 3))
    diabetes_curves = {}

    for ds in datasets:
        print(f"--- {ds['name']} ---")
        print(
            f"{'Modelo':<26} | {'Limpio':<13} | {'Pos +10σ':<13} | {'Pos +20σ':<13} | "
            f"{'Neg -10σ':<13} | {'Mix ±10σ':<13} | {'CPU (ms)':<12}"
        )
        print("-" * 116)

        for m_name, factory in models.items():
            res = {
                k: []
                for k in [
                    "clean",
                    "pos_5",
                    "pos_10",
                    "pos_20",
                    "neg_10",
                    "neg_20",
                    "mix_10",
                    "ms",
                ]
            }

            for s_idx, seed in enumerate(SEEDS_10):
                strat = ds["y"] if ds["task"] == "cls" else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    ds["X"],
                    ds["y"],
                    test_size=0.25,
                    random_state=seed,
                    stratify=strat,
                )
                sx = StandardScaler()
                X_tr = sx.fit_transform(X_tr)
                X_te = sx.transform(X_te)

                if ds["task"] == "reg":
                    sy = StandardScaler()
                    y_tr = sy.fit_transform(y_tr)
                    y_te = sy.transform(y_te)

                net = MLP(X_tr.shape[1], [48, 32], ds["out_dim"], factory, seed=seed)
                opt = AdamPGD(
                    lr_w=0.004, lr_act=0.001, wd_act=PSRA_CANONICAL["wd_act"]
                )

                t0 = time.perf_counter()
                for ep in range(ds["epochs"]):
                    out = net.forward(X_tr)
                    if ds["task"] == "cls":
                        _, d_out = softmax_ce_loss(out, y_tr, ds["out_dim"])
                    else:
                        d_out = 2.0 * (out - y_tr) / X_tr.shape[0]
                    net.backward(d_out)
                    opt.step(net.all_param_groups())

                    if (
                        ds["key"] == "diabetes"
                        and m_name == "P-SRA Equilibrada (Ours)"
                    ):
                        for l_idx in range(2):
                            act = net.acts[l_idx]
                            pct_acc[s_idx, ep, l_idx, 0] = np.percentile(
                                act.alpha, [10, 50, 90]
                            )
                            pct_acc[s_idx, ep, l_idx, 1] = np.percentile(
                                act.beta, [10, 50, 90]
                            )
                            pct_acc[s_idx, ep, l_idx, 2] = np.percentile(
                                act.gamma, [10, 50, 90]
                            )

                res["ms"].append((time.perf_counter() - t0) * 1000.0)

                def eval_fn(X_ev):
                    p = net.forward(X_ev)
                    if ds["task"] == "cls":
                        return np.mean(np.argmax(p, axis=1) == y_te) * 100.0
                    return np.mean((p - y_te) ** 2)

                res["clean"].append(eval_fn(X_te))
                res["pos_5"].append(eval_fn(inject_outliers(X_te, seed, "pos", 5.0)))
                res["pos_10"].append(eval_fn(inject_outliers(X_te, seed, "pos", 10.0)))
                res["pos_20"].append(eval_fn(inject_outliers(X_te, seed, "pos", 20.0)))
                res["neg_10"].append(eval_fn(inject_outliers(X_te, seed, "neg", 10.0)))
                res["neg_20"].append(eval_fn(inject_outliers(X_te, seed, "neg", 20.0)))
                res["mix_10"].append(eval_fn(inject_outliers(X_te, seed, "mix", 10.0)))

            fmt = lambda a: f"{np.mean(a):.2f}±{np.std(a):.2f}"
            print(
                f"{m_name:<26} | {fmt(res['clean']):<13} | {fmt(res['pos_10']):<13} | "
                f"{fmt(res['pos_20']):<13} | {fmt(res['neg_10']):<13} | "
                f"{fmt(res['mix_10']):<13} | {np.mean(res['ms']):.1f}±{np.std(res['ms']):.1f}"
            )

            if ds["key"] == "diabetes":
                diabetes_curves[m_name] = {
                    "pos": [
                        np.mean(res["clean"]),
                        np.mean(res["pos_5"]),
                        np.mean(res["pos_10"]),
                        np.mean(res["pos_20"]),
                    ],
                    "neg": [
                        np.mean(res["clean"]),
                        np.mean(res["neg_10"]),
                        np.mean(res["neg_20"]),
                    ],
                }
        print()

    return np.mean(pct_acc, axis=0), diabetes_curves


# ==============================================================================
# TABLA 2: ESTUDIO DE ABLACIÓN UNIFICADO (10 SEMILLAS)
# ==============================================================================

def run_unified_ablation():
    print("=" * 116)
    print("TABLA 2: ESTUDIO DE ABLACIÓN UNIFICADO (10 SEMILLAS | ECUACIÓN VS. COTAS VS. PGD)")
    print("=" * 116)

    diab = load_diabetes()
    bc = load_breast_cancer()

    variants = {
        "P-SRA Completa (Ecuación + PGD + Neurona)": lambda u: PSRA(
            u, alpha0=1.0, beta0=0.08, gamma0=0.35, layerwise=False
        ),
        "Ablación 1: Sin Ecuación (Clipped PReLU)": lambda u: ClippedPReLU(
            u, alpha0=1.0, gamma0=0.35, low_clip=-1.0, high_clip=12.5
        ),
        "Ablación 2: Sin Neurona (PGD Por Capa)": lambda u: PSRA(
            u, alpha0=1.0, beta0=0.08, gamma0=0.35, layerwise=True
        ),
        "Ablación 3: Sin PGD Acotado (Reparam. q²)": lambda u: PSRA_QuadAblation(
            u, alpha0=1.0, beta0=0.08, gamma0=0.35
        ),
    }

    tasks = [
        ("Diabetes (MSE ↓)", diab.data, diab.target.reshape(-1, 1), "reg", 1, 300),
        ("Breast Cancer (Acc % ↑)", bc.data[:, :25], bc.target, "cls", 2, 300),
    ]

    for t_name, X_raw, y_raw, t_type, out_dim, epochs in tasks:
        print(f"--- {t_name} ---")
        print(
            f"{'Variante':<42} | {'Métrica Limpia':<18} | {'Outliers (+15σ)':<18} | {'Neg (-15σ)':<18}"
        )
        print("-" * 116)

        for v_name, factory in variants.items():
            c_list, p15_list, n15_list = [], [], []
            for seed in SEEDS_10:
                strat = y_raw if t_type == "cls" else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_raw, y_raw, test_size=0.25, random_state=seed, stratify=strat
                )
                sx = StandardScaler()
                X_tr = sx.fit_transform(X_tr)
                X_te = sx.transform(X_te)

                if t_type == "reg":
                    sy = StandardScaler()
                    y_tr = sy.fit_transform(y_tr)
                    y_te = sy.transform(y_te)

                net = MLP(X_tr.shape[1], [48, 32], out_dim, factory, seed=seed)
                opt = AdamPGD(
                    lr_w=0.004, lr_act=0.001, wd_act=PSRA_CANONICAL["wd_act"]
                )

                for _ in range(epochs):
                    out = net.forward(X_tr)
                    if t_type == "cls":
                        _, d_out = softmax_ce_loss(out, y_tr, out_dim)
                    else:
                        d_out = 2.0 * (out - y_tr) / X_tr.shape[0]
                    net.backward(d_out)
                    opt.step(net.all_param_groups())

                def eval_fn(X_ev):
                    p = net.forward(X_ev)
                    if t_type == "cls":
                        return np.mean(np.argmax(p, axis=1) == y_te) * 100.0
                    return np.mean((p - y_te) ** 2)

                c_list.append(eval_fn(X_te))
                p15_list.append(eval_fn(inject_outliers(X_te, seed, "pos", 15.0)))
                n15_list.append(eval_fn(inject_outliers(X_te, seed, "neg", 15.0)))

            fmt = lambda a: f"{np.mean(a):.2f} ± {np.std(a):.2f}"
            print(
                f"{v_name:<42} | {fmt(c_list):<18} | {fmt(p15_list):<18} | {fmt(n15_list):<18}"
            )
        print()


# ==============================================================================
# TABLA 3: FLUJO DE GRADIENTE EN 12 CAPAS (10 SEMILLAS UNIFICADAS)
# ==============================================================================

def run_unified_deep_gradient():
    print("=" * 116)
    print("TABLA 3: RETENCIÓN DE GRADIENTE EN RED DE 12 CAPAS SIN RESIDUALES (10 SEMILLAS)")
    print("=" * 116)
    print(
        f"{'Activación':<30} | {'Norma Capa 12':<16} | {'Norma Capa 1':<16} | {'Retención (L1/L12)':<18}"
    )
    print("-" * 116)

    configs = [
        ("Sigmoid (Xavier)", lambda u: Sigmoid(u), "xavier"),
        ("ReLU (He)", lambda u: ReLU(u), "he"),
        ("Clipped PReLU (He)", lambda u: ClippedPReLU(u), "he"),
        ("GELU (He)", lambda u: GELU(u), "he"),
        ("P-SRA Equilibrada (β0=0.08)", lambda u: PSRA(u, beta0=0.08), "he"),
    ]

    for name, factory, init_mode in configs:
        n12, n1, ret = [], [], []
        for s in SEEDS_10:
            rng = np.random.default_rng(s)
            X = rng.normal(0.0, 1.0, (256, 64))
            y = rng.normal(0.0, 1.0, (256, 1))
            net = MLP(64, [64] * 12, 1, factory, seed=s, init_mode=init_mode)
            out = net.forward(X)
            net.backward(2.0 * (out - y) / 256.0)

            g1 = np.linalg.norm(net.dW[0])
            g12 = np.linalg.norm(net.dW[11])
            n1.append(g1)
            n12.append(g12)
            ret.append(g1 / (g12 + 1e-18))

        print(
            f"{name:<30} | {np.mean(n12):<16.4e} | {np.mean(n1):<16.4e} | {np.mean(ret):<18.4f}"
        )
    print()


# ==============================================================================
# TABLA 4: ESCALABILIDAD EN FASHION-MNIST (60,000 TRAIN / 10,000 TEST | 5 SEMILLAS)
# ==============================================================================

def load_fashion_mnist():
    urls = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ]
    files = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs("fmnist_data", exist_ok=True)
    data = {}
    for key, fname in files.items():
        fpath = os.path.join("fmnist_data", fname)
        if not os.path.exists(fpath):
            for u in urls:
                try:
                    urllib.request.urlretrieve(u + fname, fpath)
                    break
                except Exception:
                    continue
        with gzip.open(fpath, "rb") as f:
            offset = 16 if "images" in fname else 8
            arr = np.frombuffer(f.read(), dtype=np.uint8, offset=offset)
            data[key] = arr.reshape(-1, 784) if "images" in fname else arr
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def run_unified_fmnist():
    print("=" * 116)
    print("TABLA 4: ESCALABILIDAD EN FASHION-MNIST (60,000 TRAIN / 10,000 TEST | 5 SEMILLAS)")
    print("=" * 116)
    X_tr, y_tr, X_te, y_te = load_fashion_mnist()

    mean_px, std_px = np.mean(X_tr), np.std(X_tr) + 1e-8
    X_tr = (X_tr - mean_px) / std_px
    X_te = (X_te - mean_px) / std_px

    rng_noise = np.random.default_rng(2026)
    X_te_noisy = X_te.copy()
    for i in range(X_te_noisy.shape[0]):
        cols = rng_noise.choice(784, size=15, replace=False)
        X_te_noisy[i, cols] += 10.0

    models = {
        "ReLU": lambda u: ReLU(u),
        "Clipped PReLU (Control)": lambda u: ClippedPReLU(u),
        "GELU": lambda u: GELU(u),
        "P-SRA Equilibrada (Ours)": lambda u: PSRA(u, beta0=0.08),
    }

    batch_size = 256
    epochs = 8

    print(
        f"{'Activación':<26} | {'Acc Limpia (%)':<18} | {'Acc Ruido (+10σ)':<18} | {'Tiempo CPU (s)':<16}"
    )
    print("-" * 116)

    for m_name, factory in models.items():
        acc_c, acc_n, secs = [], [], []
        for s in SEEDS_FMNIST:
            rng = np.random.default_rng(s)
            net = MLP(784, [128, 64], 10, factory, seed=s)
            opt = AdamPGD(
                lr_w=0.002, lr_act=0.0005, wd_act=PSRA_CANONICAL["wd_act"]
            )

            t0 = time.perf_counter()
            n_samples = X_tr.shape[0]
            for _ in range(epochs):
                perm = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = perm[start : start + batch_size]
                    logits = net.forward(X_tr[idx])
                    _, d_loss = softmax_ce_loss(logits, y_tr[idx], 10)
                    net.backward(d_loss)
                    opt.step(net.all_param_groups())
            secs.append(time.perf_counter() - t0)

            acc_c.append(
                np.mean(np.argmax(net.forward(X_te), axis=1) == y_te) * 100.0
            )
            acc_n.append(
                np.mean(np.argmax(net.forward(X_te_noisy), axis=1) == y_te) * 100.0
            )

        print(
            f"{m_name:<26} | {np.mean(acc_c):.2f} ± {np.std(acc_c):.2f}      | "
            f"{np.mean(acc_n):.2f} ± {np.std(acc_n):.2f}      | {np.mean(secs):.2f} ± {np.std(secs):.2f}"
        )
    print()


# ==============================================================================
# GENERACIÓN DE LAS 3 FIGURAS DEFINITIVAS (300 DPI)
# ==============================================================================

def generate_paper_figures(pct_trajectories, diabetes_curves, out_dir="assets"):
    os.makedirs(out_dir, exist_ok=True)

    # --- FIGURA 1: P-SRA Equilibrada vs Clipped PReLU vs GELU ---
    z = np.linspace(-6.0, 16.0, 1000)
    a0, b0, g0 = 1.0, 0.08, 0.35
    psra_y = np.where(z >= 0, (a0 * z) / (1 + b0 * z), (g0 * z) / (1 - g0 * z))
    psra_dy = np.where(
        z >= 0, a0 / ((1 + b0 * z) ** 2), g0 / ((1 - g0 * z) ** 2)
    )

    clip_raw = np.where(z >= 0, a0 * z, g0 * z)
    clip_y = np.clip(clip_raw, -1.0, a0 / b0)
    clip_dy = np.where(
        (clip_raw > -1.0) & (clip_raw < a0 / b0), np.where(z >= 0, a0, g0), 0.0
    )

    c = np.sqrt(2.0 / np.pi)
    th = np.tanh(c * (z + 0.044715 * (z ** 3)))
    gelu_y = 0.5 * z * (1.0 + th)
    gelu_dy = 0.5 * (1.0 + th) + 0.5 * z * (1.0 - th ** 2) * c * (
        1.0 + 3 * 0.044715 * z ** 2
    )

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=300)
    ax1.plot(
        z,
        psra_y,
        color="#1f77b4",
        lw=2.5,
        label=r"P-SRA Equilibrada ($\alpha_0=1.0, \beta_0=0.08, \gamma_0=0.35$)",
    )
    ax1.plot(
        z,
        clip_y,
        color="#d62728",
        lw=2.0,
        ls="--",
        label="Clipped PReLU (Control $[-1, 12.5]$)",
    )
    ax1.plot(z, gelu_y, color="#ff7f0e", lw=1.8, ls="-.", label="GELU")
    ax1.axhline(
        a0 / b0,
        color="gray",
        ls=":",
        lw=1.2,
        label=r"Techo Asintótico $\alpha_0/\beta_0 = 12.5$",
    )
    ax1.axhline(-1.0, color="gray", ls=":", lw=1.2)
    ax1.set_title("Respuesta de Activación $f(z)$")
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$f(z)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5, loc="upper left")

    ax2.plot(
        z,
        psra_dy,
        color="#1f77b4",
        lw=2.5,
        label="P-SRA $f'(z)$ (Cola suave $O(z^{-2})$)",
    )
    ax2.plot(
        z,
        clip_dy,
        color="#d62728",
        lw=2.0,
        ls="--",
        label="Clipped PReLU $f'(z)$ (Corte abrupto a 0)",
    )
    ax2.plot(z, gelu_dy, color="#ff7f0e", lw=1.8, ls="-.", label="GELU $f'(z)$")
    ax2.set_title("Derivada Espacial $f'(z)$")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$f'(z)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_psra_geometry.png"), dpi=300)
    plt.close(fig1)

    # --- FIGURA 2: Evolución de Percentiles P10, P50, P90 (10 Semillas) ---
    epochs_axis = np.arange(1, pct_trajectories.shape[0] + 1)
    p_names = [
        r"$\alpha$ (Ganancia Positiva)",
        r"$\beta$ (Compresión Positiva)",
        r"$\gamma$ (Amortiguación Negativa)",
    ]
    cols = ["#1f77b4", "#d62728", "#2ca02c"]

    fig2, axes = plt.subplots(2, 3, figsize=(13.5, 6.5), dpi=300, sharex=True)
    for l_idx in range(2):
        for p_idx in range(3):
            ax = axes[l_idx, p_idx]
            p10 = pct_trajectories[:, l_idx, p_idx, 0]
            p50 = pct_trajectories[:, l_idx, p_idx, 1]
            p90 = pct_trajectories[:, l_idx, p_idx, 2]
            ax.plot(epochs_axis, p50, color=cols[p_idx], lw=2.2, label="Mediana (P50)")
            ax.fill_between(
                epochs_axis,
                p10,
                p90,
                color=cols[p_idx],
                alpha=0.22,
                label="Banda Interdecil (P10–P90)",
            )
            ax.set_title(f"Capa Oculta {l_idx + 1} — {p_names[p_idx]}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if l_idx == 1:
                ax.set_xlabel("Época")
            if p_idx == 0:
                ax.set_ylabel("Valor del Parámetro")
            if l_idx == 0 and p_idx == 0:
                ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_neuron_percentiles.png"), dpi=300)
    plt.close(fig2)

    # --- FIGURA 3: Ecuación Racional vs. Recorte Duro ante Outliers ---
    fig3, (ax_p, ax_n) = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=300)
    x_p = ["Limpio", "+5σ", "+10σ", "+20σ"]
    x_n = ["Limpio", "-10σ", "-20σ"]

    styles = {
        "ReLU": ("--", "o", "#7f7f7f"),
        "PReLU": ("--", "s", "#8c564b"),
        "Clipped PReLU (Control)": ("-.", "^", "#d62728"),
        "Elliott": (":", "v", "#17becf"),
        "GELU": ("-.", "d", "#ff7f0e"),
        "P-SRA Equilibrada (Ours)": ("-", "D", "#1f77b4"),
    }

    for name, (ls, mk, col) in styles.items():
        lw = 2.6 if "P-SRA" in name or "Clipped" in name else 1.6
        ax_p.plot(
            x_p,
            diabetes_curves[name]["pos"],
            ls=ls,
            marker=mk,
            color=col,
            lw=lw,
            label=name,
        )
        ax_n.plot(
            x_n,
            diabetes_curves[name]["neg"],
            ls=ls,
            marker=mk,
            color=col,
            lw=lw,
            label=name,
        )

    ax_p.set_yscale("log")
    ax_n.set_yscale("log")
    ax_p.set_title("Outliers Positivos Puros (+5σ a +20σ)")
    ax_n.set_title("Outliers Negativos Puros (-10σ a -20σ)")
    ax_p.set_ylabel("MSE de Prueba (Escala Log ↓)")
    ax_p.grid(True, which="both", alpha=0.3)
    ax_n.grid(True, which="both", alpha=0.3)
    ax_p.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_asymmetry_ablation.png"), dpi=300)
    plt.close(fig3)
    print(
        f"Figuras guardadas a 300 DPI en '{out_dir}/': "
        "fig1_psra_geometry.png, fig2_neuron_percentiles.png, fig3_asymmetry_ablation.png"
    )


if __name__ == "__main__":
    pct_traj, diab_curves = run_unified_tabular_benchmark()
    run_unified_ablation()
    run_unified_deep_gradient()
    run_unified_fmnist()
    generate_paper_figures(pct_traj, diab_curves, out_dir="assets")
