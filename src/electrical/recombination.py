"""재결합 프로파일 계산 모듈 (Step 9).

물리 모델
---------
SRH 재결합 (포획-방출, trap-assisted):
    R_srh = (np − ni²) / [τ_p(n+ni) + τ_n(p+ni)]

Langevin 재결합 (쌍극자, bimolecular):
    R_lan = k_L · max(np − ni², 0)
    k_L   = q(μ_n + μ_p) / (ε_r ε_0)   [m³/s]

엑시톤 생성 (스핀 통계, singlet-triplet ratio):
    G_singlet = 0.25 · R_lan
    G_triplet = 0.75 · R_lan
    → 인광 OLED는 삼중항(Ir(ppy)₃)까지 수확 → 이론 IQE 100% 가능
"""

from __future__ import annotations

import numpy as np

from .continuity import ContinuitySolver, _NI_DEFAULT
from .mesh import EPS0, Mesh1D, Q
from .models import DeviceState, RecombinationProfile, SweepResult


class RecombinationSolver:
    """위치별 재결합 프로파일 계산기.

    Parameters
    ----------
    mesh       : Mesh1D — 노드당 재료 물성 포함
    continuity : ContinuitySolver — SRH 계산 재사용
    """

    def __init__(self, mesh: Mesh1D, continuity: ContinuitySolver) -> None:
        self._mesh = mesh
        self._continuity = continuity

        # Langevin 계수 k_L = q(μ_n + μ_p) / (ε_r ε_0),  단위: [m³/s]
        eps = np.array([p.eps_r for p in mesh.node_props]) * EPS0
        mu_n = np.array([p.mu_n for p in mesh.node_props])
        mu_p = np.array([p.mu_p for p in mesh.node_props])
        self._k_lan: np.ndarray = Q * (mu_n + mu_p) / eps  # (N,)

        # 전극 노드 마스크 — 금속은 Langevin 재결합 없음
        self._organic_mask: np.ndarray = np.array(
            [not p.is_electrode for p in mesh.node_props], dtype=bool
        )

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def compute(
        self,
        state: DeviceState,
        voltage_V: float | None = None,
    ) -> RecombinationProfile:
        """수렴된 DeviceState에서 재결합 프로파일 계산.

        Parameters
        ----------
        state     : 수렴된 DeviceState (n_m3, p_m3 배열 포함)
        voltage_V : 기록용 전압 [V]; None 이면 state.voltage_V 사용

        Returns
        -------
        RecombinationProfile
        """
        n, p = state.n_m3, state.p_m3
        ni = _NI_DEFAULT

        # SRH — 유기물 노드에만 적용
        # 전극 노드는 접촉 BC 고정값(n=1e21 또는 p=1e21)에 의한
        # 인위적 대형 SRH를 물리적 재결합으로 오해하지 않도록 마스킹
        R_srh_raw = self._continuity.recombination_srh(n, p, ni)
        R_srh = np.maximum(R_srh_raw, 0.0) * self._organic_mask

        # Langevin — 유기물 노드에만 적용 (금속 전극 제외)
        np_excess = np.maximum(n * p - ni ** 2, 0.0)
        R_lan = self._k_lan * np_excess * self._organic_mask

        V = voltage_V if voltage_V is not None else state.voltage_V

        return RecombinationProfile(
            voltage_V=V,
            z_nm=self._mesh.z_nm.copy(),
            R_srh_m3s=R_srh,
            R_lan_m3s=R_lan,
            R_total_m3s=R_srh + R_lan,
        )


class RecombinationSweep:
    """전압 스윕 전체에 걸쳐 재결합 프로파일 일괄 계산.

    Parameters
    ----------
    rec_solver : RecombinationSolver
    """

    def __init__(self, rec_solver: RecombinationSolver) -> None:
        self._solver = rec_solver

    def compute_all(self, sweep: SweepResult) -> list[RecombinationProfile]:
        """모든 바이어스 포인트에서 재결합 프로파일 계산.

        Parameters
        ----------
        sweep : SweepResult — GummelSolver가 생성한 J-V 스윕 결과

        Returns
        -------
        list[RecombinationProfile], sweep.bias_points 와 1:1 대응
        """
        return [
            self._solver.compute(bp.state, bp.voltage_V)
            for bp in sweep.bias_points
        ]

    def peak_positions(self, profiles: list[RecombinationProfile]) -> np.ndarray:
        """바이어스별 최대 재결합 위치 [nm], shape (M,)."""
        return np.array([pr.peak_z_nm for pr in profiles])

    def total_recombinations(
        self, profiles: list[RecombinationProfile]
    ) -> np.ndarray:
        """바이어스별 전체 재결합률 적분 [m⁻²/s], shape (M,)."""
        return np.array([pr.total_recombination_m2s for pr in profiles])
