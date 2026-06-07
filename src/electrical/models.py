"""Data structures for electrical simulation state and results.

Hierarchy
---------
GummelConfig        — convergence / damping parameters
DeviceState         — φ(z), n(z), p(z), J_n(z), J_p(z) at one bias
BiasPoint           — DeviceState + convergence metadata
SweepResult         — ordered list of BiasPoint (J-V curve)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------

@dataclass
class GummelConfig:
    """Convergence and damping parameters for the Gummel iteration."""
    max_iterations: int = 100
    tolerance: float = 1e-6   # max |Δφ| convergence criterion [V]
    damping: float = 1.0      # mixing factor ∈ (0, 1]; 1.0 = no damping


# ---------------------------------------------------------------------------
# Per-bias device state
# ---------------------------------------------------------------------------

@dataclass
class DeviceState:
    """Complete electrical state of the device at a single bias voltage.

    Array units
    -----------
    phi_V   : V   (electrostatic potential)
    n_m3    : m⁻³ (electron density)
    p_m3    : m⁻³ (hole density)
    Jn_Am2  : A/m² (electron current density, defined at edges)
    Jp_Am2  : A/m² (hole current density, defined at edges)
    dz_m    : m   (edge spacings, optional — enables E_field_Vm)
    """

    voltage_V: float
    phi_V: np.ndarray        # (N,)
    n_m3: np.ndarray         # (N,)
    p_m3: np.ndarray         # (N,)
    Jn_Am2: np.ndarray       # (N-1,)
    Jp_Am2: np.ndarray       # (N-1,)
    dz_m: np.ndarray | None = None   # (N-1,) for E_field_Vm

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def J_total(self) -> float:
        """Total current density at device mid-plane [A/m²].

        Positive = conventional current from anode to cathode.
        """
        if len(self.Jn_Am2) == 0:
            return 0.0
        mid = len(self.Jn_Am2) // 2
        return float(self.Jn_Am2[mid] + self.Jp_Am2[mid])

    @property
    def J_total_mAcm2(self) -> float:
        """J_total converted to mA/cm²  (SI: ×0.1)."""
        return self.J_total * 0.1

    @property
    def E_field_Vm(self) -> np.ndarray | None:
        """Electric field E = −dφ/dz at each edge [V/m].

        Returns None when dz_m is not set.
        """
        if self.dz_m is None or len(self.dz_m) == 0:
            return None
        return -np.diff(self.phi_V) / self.dz_m

    @property
    def np_product(self) -> np.ndarray:
        """n·p product [m⁻⁶], diagnostic for recombination analysis."""
        return self.n_m3 * self.p_m3


# ---------------------------------------------------------------------------
# Bias-point result
# ---------------------------------------------------------------------------

@dataclass
class BiasPoint:
    """Simulation result at a single applied bias voltage."""
    voltage_V: float
    state: DeviceState
    converged: bool
    n_iterations: int


# ---------------------------------------------------------------------------
# J-V sweep result
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Ordered collection of :class:`BiasPoint` objects from a voltage scan."""
    bias_points: list[BiasPoint]

    # ------------------------------------------------------------------
    # Aggregated arrays
    # ------------------------------------------------------------------

    @property
    def voltages(self) -> np.ndarray:
        """Applied voltages [V], shape (M,)."""
        return np.array([bp.voltage_V for bp in self.bias_points])

    @property
    def J_total(self) -> np.ndarray:
        """Total mid-plane current density [A/m²], shape (M,)."""
        return np.array([bp.state.J_total for bp in self.bias_points])

    @property
    def J_total_mAcm2(self) -> np.ndarray:
        """J_total in mA/cm², shape (M,)."""
        return self.J_total * 0.1

    @property
    def all_converged(self) -> bool:
        """True if every bias point reached the Gummel convergence criterion."""
        return all(bp.converged for bp in self.bias_points)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save J-V table as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "voltage_V,J_Am2,J_mAcm2,converged,n_iter"
        rows = [
            f"{bp.voltage_V:.6f},{bp.state.J_total:.6e},"
            f"{bp.state.J_total_mAcm2:.6e},{bp.converged},{bp.n_iterations}"
            for bp in self.bias_points
        ]
        path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Save full sweep summary as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "voltages_V":    self.voltages.tolist(),
            "J_Am2":         self.J_total.tolist(),
            "J_mAcm2":       self.J_total_mAcm2.tolist(),
            "all_converged": self.all_converged,
            "n_points":      len(self.bias_points),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        return (
            f"SweepResult: {len(self.bias_points)} bias points  "
            f"V=[{self.voltages[0]:.2f}, {self.voltages[-1]:.2f}] V  "
            f"all_converged={self.all_converged}"
        )
