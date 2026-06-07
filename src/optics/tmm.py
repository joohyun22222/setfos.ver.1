"""Transfer Matrix Method (TMM) core solver for multilayer optical systems.

Implements the characteristic matrix formalism for normal-incidence plane waves.
All arrays are vectorised over wavelength for efficiency.

Reference:
  Born & Wolf, "Principles of Optics", ch. 1 — stratified media.
  Heavens, "Optical Properties of Thin Solid Films", ch. 4.
"""

from __future__ import annotations

import numpy as np


def _layer_matrix(n_cplx: np.ndarray, d_nm: float, wl_nm: np.ndarray) -> np.ndarray:
    """2×2 characteristic matrix for a single layer.

    Parameters
    ----------
    n_cplx : (N_wl,) complex  — refractive index at each wavelength
    d_nm   : float            — layer thickness in nm
    wl_nm  : (N_wl,) float   — wavelengths in nm

    Returns
    -------
    M : (N_wl, 2, 2) complex
    """
    delta = 2.0 * np.pi * n_cplx * d_nm / wl_nm   # complex phase thickness

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    M = np.zeros((len(wl_nm), 2, 2), dtype=complex)
    M[:, 0, 0] =  cos_d
    M[:, 0, 1] = -1j * sin_d / n_cplx
    M[:, 1, 0] = -1j * n_cplx * sin_d
    M[:, 1, 1] =  cos_d
    return M


def _stack_transfer_matrix(
    n_layers: list[np.ndarray],
    d_layers_nm: list[float],
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Product of characteristic matrices for the full stack.

    Returns
    -------
    M : (N_wl, 2, 2) complex
    """
    N_wl = len(wavelengths_nm)
    M = np.tile(np.eye(2, dtype=complex), (N_wl, 1, 1))
    for n_cplx, d in zip(n_layers, d_layers_nm):
        M = M @ _layer_matrix(n_cplx, d, wavelengths_nm)
    return M


def _r_amplitude(
    n_layers: list[np.ndarray],
    d_layers_nm: list[float],
    wavelengths_nm: np.ndarray,
    n_inc: float | complex = 1.0,
    n_sub: float | complex = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex reflection and transmission amplitudes (r, t).

    Used internally for field profile reconstruction.
    """
    M = _stack_transfer_matrix(n_layers, d_layers_nm, wavelengths_nm)
    eta_i = complex(n_inc)
    eta_s = complex(n_sub)
    B = M[:, 0, 0] + M[:, 0, 1] * eta_s
    C = M[:, 1, 0] + M[:, 1, 1] * eta_s
    denom = eta_i * B + C
    r = (eta_i * B - C) / denom
    t = 2.0 * eta_i / denom
    return r, t


def compute_rta(
    n_layers: list[np.ndarray],
    d_layers_nm: list[float],
    wavelengths_nm: np.ndarray,
    n_inc: float | complex = 1.0,
    n_sub: float | complex = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R, T, A spectra for a multilayer stack at normal incidence.

    Parameters
    ----------
    n_layers     : list of (N_wl,) complex — complex n for each layer
    d_layers_nm  : list of float           — thickness of each layer in nm
    wavelengths_nm : (N_wl,) float         — query wavelengths in nm
    n_inc        : refractive index of incident medium (default: air = 1.0)
    n_sub        : refractive index of substrate / exit medium (default: glass = 1.5)

    Returns
    -------
    R, T, A : each (N_wl,) float
        Reflectance, Transmittance, Absorptance (R + T + A ≈ 1).
    """
    if len(n_layers) != len(d_layers_nm):
        raise ValueError(
            f"n_layers length ({len(n_layers)}) must match d_layers_nm ({len(d_layers_nm)})"
        )

    M = _stack_transfer_matrix(n_layers, d_layers_nm, wavelengths_nm)
    eta_i = complex(n_inc)
    eta_s = complex(n_sub)

    B = M[:, 0, 0] * 1.0 + M[:, 0, 1] * eta_s
    C = M[:, 1, 0] * 1.0 + M[:, 1, 1] * eta_s

    denom = eta_i * B + C

    r = (eta_i * B - C) / denom          # reflection amplitude
    t = 2.0 * eta_i / denom              # transmission amplitude

    R = np.abs(r) ** 2
    T = (np.real(eta_s) / np.real(eta_i)) * np.abs(t) ** 2
    A = np.clip(1.0 - R - T, 0.0, 1.0)  # clip numerical noise below 0

    return R, T, A


def compute_rta_single(
    n_layers: list[complex],
    d_layers_nm: list[float],
    wavelength_nm: float,
    n_inc: float | complex = 1.0,
    n_sub: float | complex = 1.5,
) -> tuple[float, float, float]:
    """Scalar wrapper for a single wavelength — useful for debugging."""
    wl = np.array([wavelength_nm])
    n_arr = [np.array([n]) for n in n_layers]
    R, T, A = compute_rta(n_arr, d_layers_nm, wl, n_inc, n_sub)
    return float(R[0]), float(T[0]), float(A[0])
