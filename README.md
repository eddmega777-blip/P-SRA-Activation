# P-SRA: Parametric Smooth Rational Activation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24378742.svg)](https://doi.org/10.5281/zenodo.24378742)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![Reproducible Benchmark](https://img.shields.io/badge/Experiments-10--Seed%20Unified-orange.svg)](experiments/psra_unified_benchmark.py)
[![Preprint](https://img.shields.io/badge/Paper-IEEEtran%20Preprint-red.svg)](paper/)

Official reference implementation and reproducible experimental suite for **P-SRA** (*Parametric Smooth Rational Activation*), a lightweight, three-parameter per-neuron piecewise rational activation function trained via box-constrained **Projected Gradient Descent (PGD)** with elastic initialization anchoring under Adam.

---

## 1. Mathematical Formulation

For a pre-activation scalar $z_{i,j}$ (sample $i$, neuron $j$), P-SRA is defined using only elementary algebraic operations (one comparison, two multiplications, one addition, and one division):

$$
f(z; \alpha_j, \beta_j, \gamma_j) =
\begin{cases}
\dfrac{\alpha_j z}{1 + \beta_j z} & \text{if } z \ge 0 \\\\[8pt]
\dfrac{\gamma_j z}{1 - \gamma_j z} & \text{if } z < 0
\end{cases}
$$

- **Asymmetric Bounded Range:** $f(z) \in \left(-1,\; \frac{\alpha_j}{\beta_j}\right)$, providing an adaptive upper asymptote for positive activations while naturally bounding negative excursions at $-1$.
- **Canonical Balanced Configuration:** Initialized at $\alpha_0 = 1.0$, $\beta_0 = 0.08$, $\gamma_0 = 0.35$, with direct PGD box constraints $\alpha_j \in [0.2, 2.5]$, $\beta_j \in [0.05, 2.0]$, $\gamma_j \in [0.05, 1.5]$ and elastic anchoring $\lambda_{\text{act}} = 10^{-3}$ toward initialization.

**Exact Spatial Derivative ($O(z^{-2})$ tail decay):**

$$
\frac{\partial f}{\partial z} =
\begin{cases}
\dfrac{\alpha_j}{(1 + \beta_j z)^2} & \text{if } z \ge 0 \\\\[8pt]
\dfrac{\gamma_j}{(1 - \gamma_j z)^2} & \text{if } z < 0
\end{cases}
$$

![P-SRA Geometry and Spatial Derivative](assets/fig1_psra_geometry.png)

---

## 2. Key Unified Findings

To decouple the intrinsic effect of the smooth rational curve from simple range truncation, P-SRA is evaluated against standard baselines alongside a range-matched **Clipped PReLU** control ($[-1.0, 12.5]$) across identical 10-seed splits:

| Benchmark / Metric | ReLU | Clipped PReLU (Control) | Elliott | GELU | **P-SRA Balanced (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer (25 in)** — Clean Acc (%) $\uparrow$ | $97.41 \pm 0.89$ | $97.41 \pm 0.77$ | $97.06 \pm 0.69$ | $97.20 \pm 0.88$ | **$97.62 \pm 0.78$** |
| **Diabetes (10 in)** — Pos $+20\sigma$ MSE $\downarrow$ | $36.57 \pm 21.33$ | $14.00 \pm 10.75$ | **$1.62 \pm 0.31$** | $32.00 \pm 11.13$ | $7.81 \pm 4.32$ |
| **Diabetes (10 in)** — Neg $-10\sigma$ MSE $\downarrow$ | $11.03 \pm 3.98$ | $7.69 \pm 3.73$ | **$1.31 \pm 0.27$** | $12.28 \pm 2.71$ | $3.65 \pm 1.41$ |
| **12-Layer Deep MLP** — Gradient Retention ($L_1/L_{12}$) $\uparrow$ | $0.4505$ | $0.2778$ | — | $0.6680$ | **$0.7096$** |
| **Fashion-MNIST (60k)** — Clean / $+10\sigma$ Acc (%) $\uparrow$ | $87.83 / 71.49$ | $87.95 / 71.51$ | — | $87.94 / 71.00$ | **$88.04 / 71.90$** |
| **Fashion-MNIST (60k)** — CPU Wall-Clock Time (s) $\downarrow$ | **$23.30 \pm 1.67$** | $31.05 \pm 3.39$ | — | $32.39 \pm 1.88$ | $29.65 \pm 1.56$ |

### Takeaways & Honest Trade-offs
1. **Smooth Curvature vs. Hard Clipping:** Comparing P-SRA against *Clipped PReLU* proves that progressive rational compression cuts $+20\sigma$ regression MSE nearly in half ($7.81$ vs. $14.00$) while preserving deep 12-layer gradient flow ($0.7096$ vs. $0.2778$).
2. **Saddle-Free PGD:** Ablation studies confirm that direct box-constrained PGD avoids the absorbing saddle points introduced by quadratic reparameterizations ($\gamma = q^2$).
3. **Known Limitations:** Fixed saturating units (*Elliott*) still achieve stronger extreme-outlier suppression in pure regression at the cost of classification flexibility, and per-neuron parameter updates incur Python overhead on very small networks unless fused in C++/SIMD.

---

## 3. Quick Reproduction

```bash
git clone [https://github.com/eddmega777-blip/P-SRA-Activation.git](https://github.com/eddmega777-blip/P-SRA-Activation.git)
cd P-SRA-Activation
pip install -r requirements.txt

# Run the complete 10-seed benchmark, ablations, 12-layer test, Fashion-MNIST, and 300-DPI figures:
python experiments/psra_unified_benchmark.py
```

---

## 4. Citation

```bibtex
@misc{calzada2026psra,
  author       = {Calzada Hern{\'a}ndez, Eduardo},
  title        = {{P-SRA: Exploring Parametric Smooth Rational Activations Under Box-Constrained Projected Gradient Descent}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24378742},
  url          = {[https://doi.org/10.5281/zenodo.24378742](https://doi.org/10.5281/zenodo.24378742)}
}
```
  note={Preprint and reproducible codebase available at [https://github.com/eddmega777-blip/P-SRA-Activation](https://github.com/eddmega777-blip/P-SRA-Activation)}
}
```
