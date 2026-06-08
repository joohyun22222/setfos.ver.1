"""Luminance calculation from a normalised emission spectrum.

Physical formulas
-----------------
CIE photopic response (Gaussian approximation):
    V(λ) = exp(−0.5 × ((λ − 555) / 40)²)

Luminance [cd/m²]:
    L = (683 / π) × (J / q) × η_rad × η_out × ∫ emission_norm(λ) × V(λ) dλ

Luminous efficacy:
    cd_per_A = L / (J × 1e-4)   [cd/A; J in A/m², ×1e-4 → A/cm²]

η_spin is absorbed into η_rad by the caller:
    fluorescent:     η_rad = PLQY × 0.25
    phosphorescent:  η_rad = PLQY × 1.0
"""

from __future__ import annotations

import numpy as np

_Q   = 1.602176634e-19   # elementary charge [C]
_K_M = 683.0             # max luminous efficacy [lm/W]
_PI  = np.pi


def cie_photopic_response(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Gaussian approximation to CIE 1931 photopic luminosity V(λ).

    V(λ) = exp(−0.5 × ((λ − 555) / 40)²)

    Parameters
    ----------
    wavelengths_nm : (N_wl,) float [nm]

    Returns
    -------
    V : (N_wl,) float in [0, 1]
    """
    return np.exp(-0.5 * ((wavelengths_nm - 555.0) / 40.0) ** 2)


def compute_luminance(
    wavelengths_nm: np.ndarray,
    emission_norm: np.ndarray,
    J_Am2: float,
    eta_rad: float,
    eta_out: float,
) -> float:
    """Compute luminance L [cd/m²].

        L = (683 / π) × (J / q) × η_rad × η_out × ∫ emission_norm(λ) × V(λ) dλ

    Parameters
    ----------
    wavelengths_nm : (N_wl,) float [nm]
    emission_norm  : (N_wl,) float — area-normalised emission spectrum
    J_Am2          : float > 0 — current density [A/m²]
    eta_rad        : float in [0,1] — radiative efficiency (includes η_spin)
    eta_out        : float in [0,1] — outcoupling efficiency

    Returns
    -------
    L : float [cd/m²];  0.0 when J_Am2 ≤ 0
    """
    if J_Am2 <= 0.0:
        return 0.0

    V = cie_photopic_response(wavelengths_nm)
    photopic_overlap = float(np.trapezoid(emission_norm * V, wavelengths_nm))

    return (_K_M / _PI) * (J_Am2 / _Q) * eta_rad * eta_out * photopic_overlap


def compute_cd_per_A(luminance_cd_m2: float, J_Am2: float) -> float:
    """Luminous efficacy [cd/A].

        cd_per_A = L [cd/m²] / (J [A/m²] × 1e-4)

    Parameters
    ----------
    luminance_cd_m2 : float [cd/m²]
    J_Am2           : float [A/m²]

    Returns
    -------
    cd_A : float [cd/A];  0.0 when J_Am2 ≤ 0
    """
    if J_Am2 <= 0.0:
        return 0.0
    return luminance_cd_m2 / (J_Am2 * 1e-4)
