"""Data structures for electro-optical coupling results (Step 10)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ElectroOpticalResult:
    """Combined electro-optical result at one bias point.

    Parameters
    ----------
    voltage_V           : applied anode voltage [V]
    J_Am2               : current density [A/m²]
    wavelength_nm       : (N_wl,) wavelength grid [nm]
    emission_spectrum   : (N_wl,) G-weighted, PL-weighted, area-normalised
    luminance_cd_m2     : photometric luminance L [cd/m²]
    cd_per_A            : luminous efficacy [cd/A]
    G_norm_on_opt_grid  : (N_eml,) normalised singlet generation on EML sub-grid
    eml_z_nm            : (N_eml,) z positions of EML sub-grid [nm]
    """

    voltage_V: float
    J_Am2: float
    wavelength_nm: np.ndarray
    emission_spectrum: np.ndarray
    luminance_cd_m2: float
    cd_per_A: float
    G_norm_on_opt_grid: np.ndarray
    eml_z_nm: np.ndarray

    @property
    def peak_emission_nm(self) -> float:
        """Wavelength of peak emission [nm]."""
        return float(self.wavelength_nm[np.argmax(self.emission_spectrum)])

    def to_csv(self, path: str | Path) -> None:
        """Save per-wavelength emission data as CSV."""
        path = Path(path)
        rows = ["wavelength_nm,emission_spectrum"]
        for wl, em in zip(self.wavelength_nm, self.emission_spectrum):
            rows.append(f"{wl:.3f},{em:.6e}")
        path.write_text("\n".join(rows))

    def to_json(self, path: str | Path) -> None:
        """Save scalar metrics and spectrum arrays as JSON."""
        data = {
            "voltage_V": self.voltage_V,
            "J_Am2": self.J_Am2,
            "luminance_cd_m2": self.luminance_cd_m2,
            "cd_per_A": self.cd_per_A,
            "peak_emission_nm": self.peak_emission_nm,
            "wavelength_nm": self.wavelength_nm.tolist(),
            "emission_spectrum": self.emission_spectrum.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def summary(self) -> str:
        return (
            f"V={self.voltage_V:.2f}V  J={self.J_Am2:.3e}A/m²  "
            f"L={self.luminance_cd_m2:.1f}cd/m²  "
            f"cd/A={self.cd_per_A:.2f}  peak={self.peak_emission_nm:.1f}nm"
        )


@dataclass
class CoupledSweepResult:
    """Ordered list of ElectroOpticalResult from a voltage sweep.

    Produced by CoupledEmissionSolver.solve_sweep().
    """

    results: list[ElectroOpticalResult]

    @property
    def voltages(self) -> np.ndarray:
        """Applied voltages [V], shape (M,)."""
        return np.array([r.voltage_V for r in self.results])

    @property
    def luminance(self) -> np.ndarray:
        """Luminance [cd/m²], shape (M,)."""
        return np.array([r.luminance_cd_m2 for r in self.results])

    @property
    def cd_per_A(self) -> np.ndarray:
        """Luminous efficacy [cd/A], shape (M,)."""
        return np.array([r.cd_per_A for r in self.results])

    def to_csv(self, path: str | Path) -> None:
        """Save L-V and cd/A-V table as CSV."""
        path = Path(path)
        rows = ["voltage_V,J_Am2,luminance_cd_m2,cd_per_A"]
        for r in self.results:
            rows.append(
                f"{r.voltage_V:.4f},{r.J_Am2:.6e},"
                f"{r.luminance_cd_m2:.4f},{r.cd_per_A:.4f}"
            )
        path.write_text("\n".join(rows))
