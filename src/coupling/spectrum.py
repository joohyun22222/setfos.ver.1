"""G-weighted optical emission spectrum computation.

Physical model
--------------
    emission[λ] = ∫_{EML} G_norm(z) × |E|²(z, λ) dz × PL(λ)

where G_norm(z) = G_singlet(z) / ∫G_singlet(z')dz'  (unit-integral normalisation).

Edge cases
----------
- All-zero G (dark/sub-threshold): falls back to uniform weighting → reproduces
  the existing "uniform" z_distribution behaviour of EmissionSolver.
- EML narrower than one grid pitch (N_eml == 1): single-point |E|² lookup.
- All-zero PL: returns zero array (no normalisation attempted).
"""

from __future__ import annotations

import numpy as np

_G_FLOOR = 1e-40   # threshold below which integral is treated as zero


def normalize_generation(
    G_singlet_eml: np.ndarray,
    z_eml_nm: np.ndarray,
) -> np.ndarray:
    """Normalise singlet generation to unit integral over the EML z-range.

    Parameters
    ----------
    G_singlet_eml : (N_eml,) float >= 0
    z_eml_nm      : (N_eml,) float, sorted ascending

    Returns
    -------
    G_norm : (N_eml,) float >= 0, ∫G_norm dz = 1 (in nm units)
             Returns uniform 1/thickness when total integral ≤ _G_FLOOR
             or N_eml == 1.
    """
    N = len(G_singlet_eml)

    if N == 1:
        return np.array([1.0])

    total = float(np.trapezoid(G_singlet_eml, z_eml_nm))

    if total <= _G_FLOOR:
        thickness = float(z_eml_nm[-1] - z_eml_nm[0])
        if thickness <= 0.0:
            return np.ones(N) / N
        return np.full(N, 1.0 / thickness)

    return G_singlet_eml / total


def weighted_emission_integral(
    G_norm: np.ndarray,
    z_eml_nm: np.ndarray,
    E_squared_eml: np.ndarray,
) -> np.ndarray:
    """Compute ∫ G_norm(z) × |E|²(z, λ) dz over the EML.

    Parameters
    ----------
    G_norm        : (N_eml,) float — normalised generation profile
    z_eml_nm      : (N_eml,) float — EML z positions [nm]
    E_squared_eml : (N_eml, N_wl) float — |E|² restricted to EML rows

    Returns
    -------
    integral : (N_wl,) float

    Notes
    -----
    Vectorised via broadcasting: G_norm[:,None] * E_squared_eml → (N_eml, N_wl),
    then np.trapezoid along axis=0.
    For N_eml == 1 (single-point EML), returns G_norm[0] * E_squared_eml[0] * 1 nm
    to avoid trapezoid returning zero over a degenerate interval.
    """
    N_eml = len(G_norm)

    if N_eml == 1:
        # Single point: treat as a 1 nm wide rectangle
        return G_norm[0] * E_squared_eml[0] * 1.0

    integrand = G_norm[:, np.newaxis] * E_squared_eml   # (N_eml, N_wl)
    return np.trapezoid(integrand, z_eml_nm, axis=0)    # (N_wl,)


def apply_pl_and_normalize(
    raw_integral: np.ndarray,
    pl_spectrum: np.ndarray,
) -> np.ndarray:
    """Multiply by PL spectrum and area-normalise.

    Parameters
    ----------
    raw_integral  : (N_wl,) float
    pl_spectrum   : (N_wl,) float — interpolated PL, max-normalised to 1

    Returns
    -------
    emission_norm : (N_wl,) float — emission spectrum, sums to 1 (if non-zero)
    """
    weighted = raw_integral * pl_spectrum
    total = float(np.sum(weighted))
    if total <= 0.0:
        return np.zeros_like(weighted)
    return weighted / total
