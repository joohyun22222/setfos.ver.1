"""EQE, current efficiency (CE), and power efficiency (PE) formulas (Step 12).

Derivation
----------
Luminance formula (from Step 10 coupling solver):

    L = (K_m / π) × (J / q) × η_EQE × photopic_overlap

where:
    K_m              = 683 lm/W  (max luminous efficacy)
    J                = current density [A/m²]
    q                = elementary charge [C]
    photopic_overlap = ∫ emission_norm(λ) × V(λ) dλ  [nm]
    V(λ)             = CIE 1931 Gaussian: exp(−0.5 × ((λ−555)/40)²)

Solving for EQE:

    η_EQE = (π × L × q) / (K_m × J × photopic_overlap)

Metric definitions
------------------
EQE [dimensionless, 0–1]:  η_EQE above
CE  [cd/A]:                L / (J × 1e-4)   — J in A/m², ×1e-4 → A/cm²
PE  [lm/W]:                L × π / (J × V)  — Lambertian emitter
"""

from __future__ import annotations

import numpy as np

_Q   = 1.602176634e-19   # elementary charge [C]
_K_M = 683.0             # max luminous efficacy [lm/W]
_PI  = np.pi


def compute_eqe(
    wavelengths_nm: np.ndarray,
    emission_norm: np.ndarray,
    L_cd_m2: float,
    J_Am2: float,
) -> float:
    """External quantum efficiency η_EQE [0, 1].

    Derived from the luminance formula:
        η_EQE = (π × L × q) / (K_m × J × ∫ emission_norm(λ)×V(λ) dλ)

    Parameters
    ----------
    wavelengths_nm : (N_wl,) [nm]
    emission_norm  : (N_wl,) area-normalised EL spectrum (sums to 1)
    L_cd_m2        : luminance [cd/m²]
    J_Am2          : current density [A/m²]

    Returns
    -------
    eta_EQE : float in [0, 1];  0.0 when J ≤ 0 or L ≤ 0
    """
    if J_Am2 <= 0.0 or L_cd_m2 <= 0.0:
        return 0.0

    V_cie = np.exp(-0.5 * ((wavelengths_nm - 555.0) / 40.0) ** 2)
    photopic_overlap = float(np.trapezoid(emission_norm * V_cie, wavelengths_nm))
    if photopic_overlap <= 0.0:
        return 0.0

    return (_PI * L_cd_m2 * _Q) / (_K_M * J_Am2 * photopic_overlap)


def compute_ce(L_cd_m2: float, J_Am2: float) -> float:
    """Current efficiency [cd/A].

    CE = L [cd/m²] / (J [A/m²] × 1e-4)

    Returns 0.0 when J_Am2 ≤ 0.
    """
    if J_Am2 <= 0.0:
        return 0.0
    return L_cd_m2 / (J_Am2 * 1e-4)


def compute_pe(L_cd_m2: float, J_Am2: float, voltage_V: float) -> float:
    """Power efficiency [lm/W] for a Lambertian emitter.

    PE = L × π / (J × V)

    Returns 0.0 when J_Am2 ≤ 0 or voltage_V ≤ 0.
    """
    if J_Am2 <= 0.0 or voltage_V <= 0.0:
        return 0.0
    return L_cd_m2 * _PI / (J_Am2 * voltage_V)
