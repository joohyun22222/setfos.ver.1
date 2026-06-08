"""Electro-optical coupling module (Step 10).

Bridges the electrical drift-diffusion solver (Steps 8-9) with the
optical emission solver (Steps 1-6) via G-weighted emission integrals.

Public API
----------
CoupledEmissionSolver  : main solver — RecombinationProfile + FieldProfileResult → EL spectrum + luminance
ElectroOpticalResult   : per-bias result (emission spectrum, luminance, cd/A)
CoupledSweepResult     : sweep-level aggregation (L-V, cd/A-V curves)
"""

from .models import CoupledSweepResult, ElectroOpticalResult
from .solver import CoupledEmissionSolver

__all__ = [
    "CoupledEmissionSolver",
    "ElectroOpticalResult",
    "CoupledSweepResult",
]
