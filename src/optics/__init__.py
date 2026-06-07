"""Optics module — TMM-based multilayer optical solver."""

from .field_profile import FieldProfileResult, LayerAbsorption, compute_field_profile
from .nk_data import NKDataProvider
from .rta import RTAResult, RTASolver
from .tmm import compute_rta, compute_rta_single

__all__ = [
    "NKDataProvider",
    "RTASolver",
    "RTAResult",
    "FieldProfileResult",
    "LayerAbsorption",
    "compute_field_profile",
    "compute_rta",
    "compute_rta_single",
]
