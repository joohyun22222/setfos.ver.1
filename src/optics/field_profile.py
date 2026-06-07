"""Layer-resolved absorption and internal E-field profile via Transfer Matrix Method.

Physical convention (Born & Wolf, normal incidence):
  - E and H fields at each z: [E, H]^T
  - Characteristic matrix M takes [E_right, H_right] → [E_left, H_left]
  - H = n·E for a forward-travelling plane wave in medium with index n
  - Poynting: S(z) = (1/2) Re[E(z)·H*(z)]
  - Incident power S_inc = n_inc/2  (incident field amplitude = 1)
  - Layer absorptance: A_j = (S_left - S_right) / S_inc

The inverse propagation formula (left-to-right, from z=0 to z=ξ within layer j):
  [E(ξ)]   [[cos δ,   i sin δ / n_j]] [E_left]
  [H(ξ)] = [[i n_j sin δ,    cos δ ]] [H_left]
where δ = 2π n_j ξ / λ  (complex phase; evanescent decay encoded in Im[n]).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tmm import _r_amplitude


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerAbsorption:
    """Wavelength-resolved absorptance for a single layer."""

    layer_name: str
    material_name: str
    thickness_nm: float
    A_spectrum: np.ndarray          # (N_wl,) float in [0, 1]

    @property
    def A_mean(self) -> float:
        """Mean absorptance across the wavelength range."""
        return float(np.mean(self.A_spectrum))

    @property
    def A_peak(self) -> float:
        """Peak absorptance."""
        return float(np.max(self.A_spectrum))


@dataclass
class FieldProfileResult:
    """Electric field profile and layer-resolved absorption for a multilayer stack.

    Arrays
    ------
    wavelength_nm       : (N_wl,)
    z_nm                : (N_z,)       z coordinates from 0 (entrance) to total thickness
    E_field             : (N_z, N_wl)  complex; normalised to unit incident amplitude
    layer_boundaries_nm : (N_layers+1,) interface z positions [0, d1, d1+d2, ...]
    layer_names         : list[str]
    layer_absorption    : list[LayerAbsorption]

    Emission-coupling interface
    ---------------------------
    get_intensity_at_z(z)  → |E|²(λ)  — for placing a dipole at z and computing LDOS weight
    get_field_at_wavelength(wl) → E(z) — for mode-profile analysis at fixed λ
    """

    wavelength_nm: np.ndarray           # (N_wl,)
    z_nm: np.ndarray                    # (N_z,)
    E_field: np.ndarray                 # (N_z, N_wl) complex
    layer_boundaries_nm: np.ndarray     # (N_layers+1,)
    layer_names: list[str]
    layer_absorption: list[LayerAbsorption]

    # -------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------

    @property
    def E_squared(self) -> np.ndarray:
        """|E(z,λ)|² normalised to incident intensity."""
        return np.abs(self.E_field) ** 2

    @property
    def total_layer_absorption(self) -> np.ndarray:
        """Sum of all layer absorptances A_j(λ), shape (N_wl,)."""
        if not self.layer_absorption:
            return np.zeros_like(self.wavelength_nm)
        return np.sum([la.A_spectrum for la in self.layer_absorption], axis=0)

    # -------------------------------------------------------------------
    # Lookup helpers for emission coupling
    # -------------------------------------------------------------------

    def get_field_at_wavelength(self, wl_nm: float) -> np.ndarray:
        """Complex E-field profile E(z) at the wavelength nearest to wl_nm, shape (N_z,)."""
        idx = int(np.argmin(np.abs(self.wavelength_nm - wl_nm)))
        return self.E_field[:, idx]

    def get_intensity_at_z(self, z_nm: float) -> np.ndarray:
        """|E|² spectrum at the z position nearest to z_nm, shape (N_wl,)."""
        idx = int(np.argmin(np.abs(self.z_nm - z_nm)))
        return self.E_squared[idx, :]

    def layer_center_z(self, layer_index: int) -> float:
        """Z coordinate at the centre of layer `layer_index` (0-based)."""
        z0 = self.layer_boundaries_nm[layer_index]
        z1 = self.layer_boundaries_nm[layer_index + 1]
        return float(0.5 * (z0 + z1))

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["FieldProfile summary:"]
        lines.append(
            f"  λ: {self.wavelength_nm[0]:.0f}–{self.wavelength_nm[-1]:.0f} nm"
            f"  ({len(self.wavelength_nm)} points)"
        )
        lines.append(
            f"  z: 0–{self.z_nm[-1]:.1f} nm  ({len(self.z_nm)} points)"
        )
        lines.append("  Layer-resolved absorptance (mean / peak over λ):")
        for la in self.layer_absorption:
            lines.append(
                f"    {la.layer_name:<22s} A_mean={la.A_mean:.4f}  A_peak={la.A_peak:.4f}"
            )
        lines.append(
            f"  Total layer A_mean = {float(np.mean(self.total_layer_absorption)):.4f}"
        )
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save |E|²(z, λ) matrix as CSV.  Rows = z positions, columns = wavelengths."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "z_nm," + ",".join(f"{w:.1f}" for w in self.wavelength_nm)
        rows = [
            f"{z:.3f}," + ",".join(f"{v:.6f}" for v in row)
            for z, row in zip(self.z_nm, self.E_squared)
        ]
        path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    def layer_absorption_to_csv(self, path: str | Path) -> None:
        """Save per-layer A(λ) spectra as CSV.  Rows = wavelengths, columns = layers."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [la.layer_name for la in self.layer_absorption]
        header = "wavelength_nm," + ",".join(names)
        rows = []
        for i, wl in enumerate(self.wavelength_nm):
            vals = ",".join(f"{la.A_spectrum[i]:.8f}" for la in self.layer_absorption)
            rows.append(f"{wl:.1f},{vals}")
        path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Save full result as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wavelength_nm": self.wavelength_nm.tolist(),
            "z_nm": self.z_nm.tolist(),
            "E_squared": self.E_squared.tolist(),
            "layer_boundaries_nm": self.layer_boundaries_nm.tolist(),
            "layer_names": self.layer_names,
            "layer_absorption": [
                {
                    "layer_name": la.layer_name,
                    "material_name": la.material_name,
                    "thickness_nm": la.thickness_nm,
                    "A_spectrum": la.A_spectrum.tolist(),
                    "A_mean": la.A_mean,
                    "A_peak": la.A_peak,
                }
                for la in self.layer_absorption
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_field_profile(
    n_layers: list[np.ndarray],
    d_layers_nm: list[float],
    wavelengths_nm: np.ndarray,
    layer_names: list[str] | None = None,
    material_names: list[str] | None = None,
    n_inc: float = 1.0,
    n_sub: float = 1.5,
    z_resolution_nm: float = 1.0,
) -> FieldProfileResult:
    """Compute internal E-field profile and layer-resolved absorptance.

    Parameters
    ----------
    n_layers       : list of (N_wl,) complex — complex refractive index per layer
    d_layers_nm    : list of float — layer thicknesses in nm
    wavelengths_nm : (N_wl,) float — wavelengths in nm
    layer_names    : optional display names for each layer
    material_names : optional material names for each layer (defaults to layer_names)
    n_inc          : refractive index of incident medium (default: 1.0 = air)
    n_sub          : refractive index of exit substrate (default: 1.5 = glass)
    z_resolution_nm : spatial sampling pitch in nm (default: 1 nm)

    Returns
    -------
    FieldProfileResult with:
        - E_field (N_z, N_wl) complex — normalised to unit incident amplitude
        - layer_absorption — Poynting-flux absorptance for each layer
    """
    if len(n_layers) != len(d_layers_nm):
        raise ValueError(
            f"n_layers ({len(n_layers)}) and d_layers_nm ({len(d_layers_nm)}) must match"
        )

    N_layers = len(n_layers)
    if layer_names is None:
        layer_names = [f"Layer_{j + 1}" for j in range(N_layers)]
    if material_names is None:
        material_names = list(layer_names)

    # ---------------------------------------------------------------
    # 1. Get complex reflection amplitude r(λ)
    # ---------------------------------------------------------------
    r, _t = _r_amplitude(n_layers, d_layers_nm, wavelengths_nm, n_inc, n_sub)

    # ---------------------------------------------------------------
    # 2. Initialise field at left face of stack (z = 0)
    #    E = 1 + r,  H = n_inc·(1 − r)   [E and H in wave-impedance units]
    # ---------------------------------------------------------------
    eta_i = float(n_inc)               # real for lossless incident medium
    E_left = 1.0 + r                   # (N_wl,) complex
    H_left = eta_i * (1.0 - r)        # (N_wl,) complex

    z_segments: list[np.ndarray] = []
    E_segments: list[np.ndarray] = []  # each: (N_z_j, N_wl) complex
    layer_abs: list[LayerAbsorption] = []
    boundaries = [0.0]
    z_offset = 0.0

    # ---------------------------------------------------------------
    # 3. Propagate left → right through each layer
    # ---------------------------------------------------------------
    for j, (n_j, d_j, lname, mname) in enumerate(
        zip(n_layers, d_layers_nm, layer_names, material_names)
    ):
        # Sample positions within layer j (endpoint NOT included to avoid duplicates)
        z_local = np.arange(0.0, d_j, z_resolution_nm)
        if len(z_local) == 0:
            z_local = np.array([0.0])

        # Phase at each (z, λ): δ[z, λ] = 2π·n_j[λ]·z / λ
        # Shapes: n_j (N_wl,), z_local (N_z_j,), wavelengths_nm (N_wl,)
        delta = (
            2.0 * np.pi
            * n_j[np.newaxis, :]            # (1, N_wl)
            * z_local[:, np.newaxis]        # (N_z_j, 1)
            / wavelengths_nm[np.newaxis, :] # (1, N_wl)
        )  # → (N_z_j, N_wl) complex

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        # Inverse propagation: E(z) = cos(δ)·E_left + (i/n_j)·sin(δ)·H_left
        E_j = (
            cos_d * E_left[np.newaxis, :]
            + (1j / n_j[np.newaxis, :]) * sin_d * H_left[np.newaxis, :]
        )  # (N_z_j, N_wl)

        z_segments.append(z_local + z_offset)
        E_segments.append(E_j)

        # Poynting flux at left boundary: S_left = Re[E·H*] / (2·n_inc)
        # Normalised to incident power (S_inc = n_inc/2): A = ΔS / S_inc = ΔS·2/n_inc
        S_left = np.real(E_left * np.conj(H_left)) / n_inc   # (N_wl,)

        # Propagate through full layer to get [E_right, H_right]
        delta_full = 2.0 * np.pi * n_j * d_j / wavelengths_nm  # (N_wl,) complex
        cos_f = np.cos(delta_full)
        sin_f = np.sin(delta_full)

        E_right = cos_f * E_left + (1j / n_j) * sin_f * H_left
        H_right = 1j * n_j * sin_f * E_left + cos_f * H_left

        S_right = np.real(E_right * np.conj(H_right)) / n_inc  # (N_wl,)

        # Layer absorptance = net Poynting flux absorbed in this slab
        A_j = np.clip(S_left - S_right, 0.0, 1.0)

        layer_abs.append(LayerAbsorption(
            layer_name=lname,
            material_name=mname,
            thickness_nm=d_j,
            A_spectrum=A_j,
        ))

        # Advance to next layer
        E_left = E_right
        H_left = H_right
        z_offset += d_j
        boundaries.append(z_offset)

    # ---------------------------------------------------------------
    # 4. Append final boundary point (right face of last layer)
    # ---------------------------------------------------------------
    if N_layers > 0:
        z_segments.append(np.array([z_offset]))
        E_segments.append(E_left[np.newaxis, :])  # field at exit interface

    z_all = np.concatenate(z_segments) if z_segments else np.array([0.0])
    E_all = np.concatenate(E_segments, axis=0) if E_segments else np.ones(
        (1, len(wavelengths_nm)), dtype=complex
    )

    return FieldProfileResult(
        wavelength_nm=wavelengths_nm,
        z_nm=z_all,
        E_field=E_all,
        layer_boundaries_nm=np.array(boundaries),
        layer_names=layer_names,
        layer_absorption=layer_abs,
    )
