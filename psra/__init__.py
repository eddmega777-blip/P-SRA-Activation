"""
P-SRA: Parametric Smooth Rational Activation
Official reference package for NumPy and PyTorch with box-constrained PGD.
"""

from .numpy_core import PSRA, AdamPGD, PSRA_CANONICAL

__version__ = "1.0.0"
__author__ = "Eduardo Calzada Hernández"
__all__ = ["PSRA", "AdamPGD", "PSRA_CANONICAL"]

# Importación opcional de PyTorch: evita errores si el usuario solo instaló NumPy
try:
    from .torch_module import PSRA as PSRATorch
    __all__.append("PSRATorch")
except ImportError:
    pass
