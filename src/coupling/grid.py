"""Grid interpolation and EML masking for electro-optical coupling.

The electrical solver operates on its own 1D node mesh (z_elec_nm).
The optical field profile uses a separate z_opt_nm grid. Both share the
same physical device span but may differ at layer boundaries.

All functions are pure numpy — no scipy dependency.
"""

from __future__ import annotations

import numpy as np


def interpolate_to_optical_grid(
    z_elec_nm: np.ndarray,
    G_elec: np.ndarray,
    z_opt_nm: np.ndarray,
) -> np.ndarray:
    """Interpolate a generation profile from the electrical grid to the optical grid.

    Uses linear interpolation with zero extrapolation outside the electrical
    mesh range.

    Parameters
    ----------
    z_elec_nm : (N_elec,) sorted ascending — electrical mesh positions [nm]
    G_elec    : (N_elec,) float >= 0 — generation rate on electrical grid
    z_opt_nm  : (N_opt,) — optical mesh positions [nm]

    Returns
    -------
    G_opt : (N_opt,) float >= 0
    """
    G_opt = np.interp(z_opt_nm, z_elec_nm, G_elec, left=0.0, right=0.0)
    return np.maximum(G_opt, 0.0)


def eml_layer_bounds(
    layer_boundaries_nm: np.ndarray,
    layer_names: list[str],
    emitter_layer_name: str,
) -> tuple[float, float]:
    """Return (z_start, z_end) in nm for the named emitter layer.

    Parameters
    ----------
    layer_boundaries_nm : (N_layers+1,) — interface z positions [nm]
    layer_names         : list[str] — length N_layers
    emitter_layer_name  : str

    Returns
    -------
    (z_start_nm, z_end_nm)

    Raises
    ------
    ValueError if layer not found.
    """
    for i, name in enumerate(layer_names):
        if name == emitter_layer_name:
            return float(layer_boundaries_nm[i]), float(layer_boundaries_nm[i + 1])
    raise ValueError(
        f"Emitter layer '{emitter_layer_name}' not found. "
        f"Available layers: {layer_names}"
    )


def eml_mask(
    z_opt_nm: np.ndarray,
    layer_boundaries_nm: np.ndarray,
    layer_names: list[str],
    emitter_layer_name: str,
) -> np.ndarray:
    """Boolean mask selecting optical grid points inside the EML layer.

    Uses a closed interval [z_start, z_end].
    If the EML is thinner than one grid pitch, returns the single nearest point.

    Parameters
    ----------
    z_opt_nm            : (N_opt,) — optical mesh positions [nm]
    layer_boundaries_nm : (N_layers+1,)
    layer_names         : list[str]
    emitter_layer_name  : str

    Returns
    -------
    mask : (N_opt,) bool

    Raises
    ------
    ValueError if emitter_layer_name not in layer_names.
    """
    z_start, z_end = eml_layer_bounds(layer_boundaries_nm, layer_names, emitter_layer_name)
    mask = (z_opt_nm >= z_start) & (z_opt_nm <= z_end)

    if not np.any(mask):
        # Fallback: pick nearest single point when EML is sub-grid-pitch thin
        idx = int(np.argmin(np.abs(z_opt_nm - 0.5 * (z_start + z_end))))
        mask = np.zeros(len(z_opt_nm), dtype=bool)
        mask[idx] = True

    return mask
