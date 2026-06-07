"""Data structures for optical emission computation.

Architecture for emission coupling (Steps 6-8):

  EmitterConfig (YAML schema, src/io/models.py)
      └─ EmitterProfile  (loaded PL spectrum, ready for computation)
             │
             ├─ StackContext   (optical stack + emitter position)
             │       │
             │       └─ OutcouplingBase.compute(ctx) → OutcouplingResult
             │
             └─ EmissionResult  (final output: weighted spectrum, η_out)

The EmitterProfile.pl_at() method interpolates the PL spectrum onto any
wavelength grid, enabling efficient re-use across different solver configs.
The StackContext bundles everything a future Purcell-factor calculator (Step 7+)
will need: n/k per layer, z_emitter, dipole orientation, and the pre-computed
FieldProfileResult (to avoid re-running TMM inside the outcoupling model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..optics.field_profile import FieldProfileResult


# ---------------------------------------------------------------------------
# Emitter description (with loaded PL data)
# ---------------------------------------------------------------------------

@dataclass
class EmitterProfile:
    """Emitter with loaded PL spectrum, ready for optical computation.

    Built by :class:`~src.emission.solver.EmissionSolver` from a raw
    :class:`~src.io.models.EmitterConfig`.
    """

    layer_name: str
    material: str
    wavelength_nm: np.ndarray       # (N_pl,) PL measurement wavelengths
    pl_spectrum: np.ndarray         # (N_pl,) normalised to max = 1
    horizontal_fraction: float      # fraction of horizontally oriented dipoles [0,1]
    z_distribution: str             # "center" | "uniform" | "front" | "back"
    emitter_type: str               # "phosphorescent" | "fluorescent"

    def pl_at(self, wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate PL onto *wavelengths* (nm).  Returns 0 outside PL range."""
        return np.interp(
            wavelengths, self.wavelength_nm, self.pl_spectrum,
            left=0.0, right=0.0,
        )


# ---------------------------------------------------------------------------
# Stack context passed to outcoupling calculators
# ---------------------------------------------------------------------------

@dataclass
class StackContext:
    """Everything an outcoupling calculator needs about the optical environment.

    Passed by :class:`~src.emission.solver.EmissionSolver` to any
    :class:`~src.emission.outcoupling.OutcouplingBase` implementation.
    Future Purcell-factor models (Step 7) will use ``n_layers``,
    ``d_layers_nm``, ``z_emitter_nm``, and ``horizontal_fraction``
    to integrate the dyadic Green's function over in-plane k-vectors.
    """

    n_layers: list[np.ndarray]      # (N_wl,) complex index per layer
    d_layers_nm: list[float]        # thickness per layer in nm
    wavelengths_nm: np.ndarray      # (N_wl,) wavelength grid
    z_emitter_nm: float             # emitter z-coordinate in device (nm)
    horizontal_fraction: float      # fraction of horizontal dipoles [0,1]
    n_inc: float                    # incident medium refractive index
    n_sub: float                    # substrate refractive index
    field_profile: FieldProfileResult  # pre-computed |E|² profile


# ---------------------------------------------------------------------------
# Outcoupling result
# ---------------------------------------------------------------------------

@dataclass
class OutcouplingResult:
    """Wavelength-resolved outcoupling efficiency from a given model."""

    wavelength_nm: np.ndarray       # (N_wl,)
    eta_spectrum: np.ndarray        # (N_wl,) outcoupling efficiency in [0, 1]
    model_name: str

    @property
    def eta_mean(self) -> float:
        """Spectrally averaged outcoupling efficiency."""
        return float(np.mean(self.eta_spectrum))


# ---------------------------------------------------------------------------
# Final emission result
# ---------------------------------------------------------------------------

@dataclass
class EmissionResult:
    """Output of the emission solver for a given OLED stack and emitter.

    Properties
    ----------
    emission_spectrum : PL × |E|² × η_out, area-normalised — the observable
                        EL spectrum shape (proportional to photon flux per nm).
    eta_out           : PL-weighted mean outcoupling efficiency.
    """

    wavelength_nm: np.ndarray           # (N_wl,)
    z_emitter_nm: float                 # emitter position in device
    layer_name: str                     # EML layer name
    pl_spectrum: np.ndarray             # (N_wl,) interpolated PL, max-normalised
    E_squared: np.ndarray               # (N_wl,) |E|² at emitter position
    weighted_spectrum: np.ndarray       # (N_wl,) PL × |E|², area-normalised
    outcoupling: OutcouplingResult | None = None

    # -------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------

    @property
    def emission_spectrum(self) -> np.ndarray:
        """PL-and-field-weighted emission spectrum, optionally outcoupling-modulated.

        Normalised so that the area under the spectrum sums to 1.
        This represents the spectral shape of the observable EL emission.
        """
        base = self.weighted_spectrum.copy()
        if self.outcoupling is not None:
            base = base * self.outcoupling.eta_spectrum
        total = base.sum()
        return base / total if total > 0 else base

    @property
    def eta_out(self) -> float | None:
        """PL-spectrum-weighted mean outcoupling efficiency.

        Returns *None* when no outcoupling model is attached.
        """
        if self.outcoupling is None:
            return None
        pl = self.pl_spectrum
        total_pl = pl.sum()
        if total_pl == 0:
            return None
        return float(np.sum(self.outcoupling.eta_spectrum * pl) / total_pl)

    @property
    def peak_emission_nm(self) -> float:
        """Wavelength of maximum emission_spectrum."""
        return float(self.wavelength_nm[np.argmax(self.emission_spectrum)])

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "EmissionResult summary:",
            f"  Emitter layer : {self.layer_name}",
            f"  Emitter z     : {self.z_emitter_nm:.1f} nm",
            f"  mean |E|²     : {float(np.mean(self.E_squared)):.4f}",
            f"  peak emission : {self.peak_emission_nm:.0f} nm",
        ]
        if self.outcoupling is not None:
            lines.append(
                f"  outcoupling   : {self.outcoupling.model_name}"
                f"  η_out = {self.outcoupling.eta_mean:.4f}"
            )
        if self.eta_out is not None:
            lines.append(f"  PL-weighted η_out = {self.eta_out:.4f}")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save per-wavelength emission data as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        has_oc = self.outcoupling is not None
        cols   = ["wavelength_nm", "pl_spectrum", "E_squared",
                  "weighted_emission", "emission_spectrum"]
        if has_oc:
            cols.append("outcoupling_eta")

        rows = []
        em = self.emission_spectrum
        for i, wl in enumerate(self.wavelength_nm):
            row = (f"{wl:.1f},{self.pl_spectrum[i]:.6f},{self.E_squared[i]:.6f},"
                   f"{self.weighted_spectrum[i]:.6f},{em[i]:.6f}")
            if has_oc:
                row += f",{self.outcoupling.eta_spectrum[i]:.6f}"
            rows.append(row)

        path.write_text(",".join(cols) + "\n" + "\n".join(rows), encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Save full result as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "wavelength_nm":     self.wavelength_nm.tolist(),
            "z_emitter_nm":      self.z_emitter_nm,
            "layer_name":        self.layer_name,
            "pl_spectrum":       self.pl_spectrum.tolist(),
            "E_squared":         self.E_squared.tolist(),
            "weighted_spectrum": self.weighted_spectrum.tolist(),
            "emission_spectrum": self.emission_spectrum.tolist(),
        }
        if self.outcoupling is not None:
            payload["outcoupling"] = {
                "model_name":   self.outcoupling.model_name,
                "eta_spectrum": self.outcoupling.eta_spectrum.tolist(),
                "eta_mean":     self.outcoupling.eta_mean,
            }
        if self.eta_out is not None:
            payload["eta_out"] = self.eta_out

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
