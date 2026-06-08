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


# ---------------------------------------------------------------------------
# Recombination profile (Step 9)
# ---------------------------------------------------------------------------

@dataclass
class RecombinationProfile:
    """바이어스별 위치 의존 재결합 프로파일.

    Array units
    -----------
    z_nm        : nm   (노드 위치)
    R_srh_m3s   : m⁻³/s  (SRH 재결합률)
    R_lan_m3s   : m⁻³/s  (Langevin 쌍극자 재결합률)
    R_total_m3s : m⁻³/s  (전체 재결합률 = SRH + Langevin)

    Exciton generation 연결
    -----------------------
    Langevin 재결합이 엑시톤을 생성한다고 가정 (스핀 통계 기준):
      G_singlet = 0.25 × R_lan  (단일항 S₁)
      G_triplet = 0.75 × R_lan  (삼중항 T₁)
    인광 OLED (Ir(ppy)₃ 등)는 삼중항 수확으로 이론 IQE = 100%.
    """

    voltage_V: float
    z_nm: np.ndarray          # (N,)
    R_srh_m3s: np.ndarray     # (N,)
    R_lan_m3s: np.ndarray     # (N,)
    R_total_m3s: np.ndarray   # (N,)

    # ------------------------------------------------------------------
    # 엑시톤 생성 속성
    # ------------------------------------------------------------------

    @property
    def G_exciton_m3s(self) -> np.ndarray:
        """엑시톤 생성률 ≡ Langevin 재결합률 [m⁻³/s]."""
        return self.R_lan_m3s

    @property
    def G_singlet_m3s(self) -> np.ndarray:
        """단일항 엑시톤 생성률 (스핀 통계 25%) [m⁻³/s]."""
        return 0.25 * self.R_lan_m3s

    @property
    def G_triplet_m3s(self) -> np.ndarray:
        """삼중항 엑시톤 생성률 (스핀 통계 75%) [m⁻³/s]."""
        return 0.75 * self.R_lan_m3s

    # ------------------------------------------------------------------
    # 진단 속성
    # ------------------------------------------------------------------

    @property
    def peak_z_nm(self) -> float:
        """최대 전체 재결합률 위치 [nm]."""
        return float(self.z_nm[np.argmax(self.R_total_m3s)])

    @property
    def total_recombination_m2s(self) -> float:
        """전체 장치 재결합률 적분 [m⁻²/s] (사다리꼴 규칙)."""
        return float(np.trapezoid(self.R_total_m3s, self.z_nm * 1e-9))

    # ------------------------------------------------------------------
    # 직렬화 (excitonics 모듈 연결용)
    # ------------------------------------------------------------------

    def exciton_generation_dict(self) -> dict:
        """Step 10 엑시톤 솔버에 전달 가능한 포맷으로 직렬화.

        반환 키
        -------
        voltage_V       : 인가 전압 [V]
        z_nm            : 위치 배열 [nm]
        G_total_m3s     : 전체 엑시톤 생성률 [m⁻³/s]
        G_singlet_m3s   : 단일항 생성률 [m⁻³/s]
        G_triplet_m3s   : 삼중항 생성률 [m⁻³/s]
        """
        return {
            "voltage_V":      self.voltage_V,
            "z_nm":           self.z_nm.tolist(),
            "G_total_m3s":    self.G_exciton_m3s.tolist(),
            "G_singlet_m3s":  self.G_singlet_m3s.tolist(),
            "G_triplet_m3s":  self.G_triplet_m3s.tolist(),
        }

    def to_csv(self, path: str | Path) -> None:
        """재결합 프로파일을 CSV 파일로 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "z_nm,R_srh_m3s,R_lan_m3s,R_total_m3s,G_singlet_m3s,G_triplet_m3s"
        rows = [
            f"{z:.3f},{rs:.6e},{rl:.6e},{rt:.6e},{gs:.6e},{gt:.6e}"
            for z, rs, rl, rt, gs, gt in zip(
                self.z_nm, self.R_srh_m3s, self.R_lan_m3s, self.R_total_m3s,
                self.G_singlet_m3s, self.G_triplet_m3s,
            )
        ]
        path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
