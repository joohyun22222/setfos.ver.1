"""J-V-L calculator module (Steps 11-12).

Orchestrates the full electro-optical sweep pipeline:
  electrical J-V → recombination profiles → G-weighted EL → luminance → EQE/CE/PE

Public API
----------
JVLCalculator  : main calculator — runs a complete J-V-L sweep
JVLResult      : sweep-level result (V, J, L, EQE, CE, PE, spectra)
JVLPoint       : per-bias result
compute_eqe    : external quantum efficiency formula
compute_ce     : current efficiency formula
compute_pe     : power efficiency formula
"""

from .calculator import JVLCalculator
from .efficiency import compute_ce, compute_eqe, compute_pe
from .models import JVLPoint, JVLResult

__all__ = [
    "JVLCalculator",
    "JVLResult",
    "JVLPoint",
    "compute_eqe",
    "compute_ce",
    "compute_pe",
]
