"""J-V-L sweep result data structures (Steps 11-12)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class JVLPoint:
    """Electro-optical result at a single bias point.

    Parameters
    ----------
    voltage_V           : applied anode voltage [V]
    J_Am2               : current density [A/m²]
    luminance_cd_m2     : luminance [cd/m²]
    eqe                 : external quantum efficiency [0, 1]
    cd_per_A            : current efficiency [cd/A]  (CE)
    lm_per_W            : power efficiency [lm/W]    (PE)
    peak_wavelength_nm  : emission peak wavelength [nm]
    wavelength_nm       : (N_wl,) wavelength grid [nm]
    emission_spectrum   : (N_wl,) area-normalised EL spectrum
    """

    voltage_V: float
    J_Am2: float
    luminance_cd_m2: float
    eqe: float
    cd_per_A: float
    lm_per_W: float
    peak_wavelength_nm: float
    wavelength_nm: np.ndarray
    emission_spectrum: np.ndarray

    # ------------------------------------------------------------------
    # Aliases for standard metric names
    # ------------------------------------------------------------------

    @property
    def ce(self) -> float:
        """Current efficiency [cd/A] — alias for cd_per_A."""
        return self.cd_per_A

    @property
    def pe(self) -> float:
        """Power efficiency [lm/W] — alias for lm_per_W."""
        return self.lm_per_W

    @property
    def eqe_pct(self) -> float:
        """EQE in percent [%]."""
        return self.eqe * 100.0

    @property
    def J_mAcm2(self) -> float:
        """Current density [mA/cm²]."""
        return self.J_Am2 * 0.1

    def summary(self) -> str:
        return (
            f"V={self.voltage_V:.2f}V  J={self.J_mAcm2:.3f}mA/cm²  "
            f"L={self.luminance_cd_m2:.1f}cd/m²  "
            f"EQE={self.eqe_pct:.2f}%  CE={self.cd_per_A:.2f}cd/A  "
            f"PE={self.lm_per_W:.2f}lm/W  peak={self.peak_wavelength_nm:.1f}nm"
        )


@dataclass
class JVLResult:
    """Ordered J-V-L sweep result from :class:`JVLCalculator`.

    Bundles per-bias :class:`JVLPoint` objects and provides
    array-level access for plotting and export.
    """

    points: list[JVLPoint]

    # ------------------------------------------------------------------
    # Aggregated arrays
    # ------------------------------------------------------------------

    @property
    def voltages(self) -> np.ndarray:
        """Applied voltages [V], shape (M,)."""
        return np.array([p.voltage_V for p in self.points])

    @property
    def J_Am2(self) -> np.ndarray:
        """Current density [A/m²], shape (M,)."""
        return np.array([p.J_Am2 for p in self.points])

    @property
    def J_mAcm2(self) -> np.ndarray:
        """Current density [mA/cm²], shape (M,)."""
        return self.J_Am2 * 0.1

    @property
    def luminance(self) -> np.ndarray:
        """Luminance [cd/m²], shape (M,)."""
        return np.array([p.luminance_cd_m2 for p in self.points])

    @property
    def eqe(self) -> np.ndarray:
        """External quantum efficiency [0, 1], shape (M,)."""
        return np.array([p.eqe for p in self.points])

    @property
    def eqe_pct(self) -> np.ndarray:
        """EQE [%], shape (M,)."""
        return self.eqe * 100.0

    @property
    def cd_per_A(self) -> np.ndarray:
        """Current efficiency [cd/A], shape (M,)."""
        return np.array([p.cd_per_A for p in self.points])

    @property
    def ce(self) -> np.ndarray:
        """Current efficiency [cd/A] — alias for cd_per_A, shape (M,)."""
        return self.cd_per_A

    @property
    def lm_per_W(self) -> np.ndarray:
        """Power efficiency [lm/W], shape (M,)."""
        return np.array([p.lm_per_W for p in self.points])

    @property
    def pe(self) -> np.ndarray:
        """Power efficiency [lm/W] — alias for lm_per_W, shape (M,)."""
        return self.lm_per_W

    @property
    def peak_wavelength_nm(self) -> np.ndarray:
        """Peak emission wavelength [nm], shape (M,)."""
        return np.array([p.peak_wavelength_nm for p in self.points])

    # ------------------------------------------------------------------
    # Spectrum access
    # ------------------------------------------------------------------

    def spectrum_at(self, voltage_V: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (wavelength_nm, emission_spectrum) for the point nearest to voltage_V.

        Parameters
        ----------
        voltage_V : target voltage [V]

        Returns
        -------
        wavelength_nm    : (N_wl,) [nm]
        emission_spectrum : (N_wl,) area-normalised EL
        """
        idx = int(np.argmin(np.abs(self.voltages - voltage_V)))
        pt = self.points[idx]
        return pt.wavelength_nm.copy(), pt.emission_spectrum.copy()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save J-V-L-EQE summary table as CSV.

        Columns: voltage_V, J_Am2, J_mAcm2, luminance_cd_m2,
                 eqe, eqe_pct, cd_per_A, lm_per_W, peak_wavelength_nm
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "voltage_V,J_Am2,J_mAcm2,luminance_cd_m2,"
            "eqe,eqe_pct,cd_per_A,lm_per_W,peak_wavelength_nm"
        )
        rows = [header]
        for p in self.points:
            rows.append(
                f"{p.voltage_V:.4f},{p.J_Am2:.6e},{p.J_mAcm2:.6e},"
                f"{p.luminance_cd_m2:.6f},"
                f"{p.eqe:.6f},{p.eqe_pct:.4f},"
                f"{p.cd_per_A:.6f},{p.lm_per_W:.6f},{p.peak_wavelength_nm:.2f}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Save scalar metrics as JSON (without full spectrum arrays)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_points": len(self.points),
            "voltage_V": self.voltages.tolist(),
            "J_Am2": self.J_Am2.tolist(),
            "J_mAcm2": self.J_mAcm2.tolist(),
            "luminance_cd_m2": self.luminance.tolist(),
            "eqe": self.eqe.tolist(),
            "eqe_pct": self.eqe_pct.tolist(),
            "cd_per_A": self.cd_per_A.tolist(),
            "lm_per_W": self.lm_per_W.tolist(),
            "peak_wavelength_nm": self.peak_wavelength_nm.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        V   = self.voltages
        L   = self.luminance
        J   = self.J_mAcm2
        eqe = self.eqe_pct
        cd  = self.cd_per_A
        return (
            f"JVLResult: {len(self.points)} bias points  "
            f"V=[{V[0]:.2f}, {V[-1]:.2f}] V  "
            f"L_max={float(L.max()):.1f} cd/m²  "
            f"J_max={float(J.max()):.3f} mA/cm²  "
            f"EQE_max={float(eqe.max()):.2f}%  "
            f"CE_max={float(cd.max()):.2f} cd/A"
        )
