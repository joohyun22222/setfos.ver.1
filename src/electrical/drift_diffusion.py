"""Gummel 자기 일관 반복 — drift-diffusion 시스템.

Gummel 알고리즘
--------------
V_anode, V_cathode 가 주어지면 |Δφ|_∞ < tolerance 가 될 때까지 반복:

  1. Poisson 방정식으로 φ(z) 갱신  (n, p 고정)
  2. 전자 연속 방정식으로 n 갱신  (φ, p 고정)
  3. 정공 연속 방정식으로 p 갱신  (φ, n 고정)
  4. 수렴 판정: max|Δφ|, max|Δn/n|, max|Δp/p| < tolerance
  5. Jn, Jp 계산 (SG 공식)

경계 조건
--------
- 양극 (노드 0):  φ = V_anode,  p = p_contact,  n = ni
- 음극 (마지막 노드): φ = V_cathode, n = n_contact,  p = ni

접촉 캐리어 밀도는 전극 옆 유기물 층의 Nc/Nv 에서 Boltzmann 통계로 계산.
"""

from __future__ import annotations

import logging

import numpy as np

from .continuity import ContinuitySolver, _NI_DEFAULT
from .mesh import KB, Mesh1D, Q, T0
from .models import BiasPoint, DeviceState, GummelConfig
from .poisson import PoissonSolver

# 이상 오믹 접촉 가정:
#   양극(anode)  : 정공 주입 → p_high, n_low
#   음극(cathode): 전자 주입 → n_high, p_low
# 1e21 m⁻³ 는 Debye 길이 ≈ 17 nm >> 2 nm 메쉬로 수치적으로 안정
_ANODE_P    = 1e21    # 양극 정공 경계 조건 [m⁻³]
_CATHODE_N  = 1e21    # 음극 전자 경계 조건 [m⁻³]

logger = logging.getLogger(__name__)


class GummelSolver:
    """Poisson + drift-diffusion 자기 일관 솔버.

    Parameters
    ----------
    mesh        : 노드당 재료 물성이 담긴 1D 메쉬
    poisson     : 조립된 :class:`~src.electrical.poisson.PoissonSolver`
    continuity  : 조립된 :class:`~src.electrical.continuity.ContinuitySolver`
    config      : 수렴·감쇄 파라미터 (:class:`GummelConfig`)
    """

    def __init__(
        self,
        mesh: Mesh1D,
        poisson: PoissonSolver,
        continuity: ContinuitySolver,
        config: GummelConfig | None = None,
    ) -> None:
        self._mesh       = mesh
        self._poisson    = poisson
        self._continuity = continuity
        self._cfg        = config or GummelConfig()

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def equilibrium_state(self, V_anode: float = 0.0) -> DeviceState:
        """열평형 초기 상태 생성.

        φ: 선형 프로파일, n = p = ni (진성 캐리어 밀도).
        """
        N   = self._mesh.num_nodes
        phi = np.linspace(V_anode, 0.0, N)
        n0  = np.full(N, _NI_DEFAULT)
        p0  = np.full(N, _NI_DEFAULT)
        return DeviceState(
            voltage_V=V_anode,
            phi_V=phi,
            n_m3=n0,
            p_m3=p0,
            Jn_Am2=np.zeros(N - 1),
            Jp_Am2=np.zeros(N - 1),
            dz_m=self._mesh.dz_m,
        )

    def solve(
        self,
        V_anode: float,
        V_cathode: float = 0.0,
        initial_state: DeviceState | None = None,
    ) -> BiasPoint:
        """고정 바이어스에서 Gummel 반복 실행.

        Parameters
        ----------
        V_anode       : 양극 전위 [V]
        V_cathode     : 음극 기준 전위 [V] (기본값 0)
        initial_state : 이전 편향점에서의 warm-start 초기 상태

        Returns
        -------
        :class:`BiasPoint` — 수렴된 :class:`DeviceState` 포함.
        """
        V_bias = V_anode - V_cathode
        state  = initial_state or self.equilibrium_state(V_anode)

        # 접촉 캐리어 밀도 계산 (한 번만)
        n_bc_l, p_bc_l = self._contact_carriers(0)       # 양극 경계
        n_bc_r, p_bc_r = self._contact_carriers(-1)      # 음극 경계

        n_iter    = 0
        converged = False

        for it in range(self._cfg.max_iterations):
            n_iter = it + 1

            # ── 1단계: Poisson ──────────────────────────────────────────
            phi_new = self._update_phi(state, V_anode, V_cathode)

            # ── 2~3단계: 연속 방정식 (SG) ──────────────────────────────
            n_new, p_new, Jn_new, Jp_new = self._update_carriers(
                state, phi_new, n_bc_l, n_bc_r, p_bc_l, p_bc_r
            )

            # ── 4단계: 수렴 판정 ────────────────────────────────────────
            converged = self._check_convergence(state, phi_new, n_new, p_new)

            state = DeviceState(
                voltage_V=V_bias,
                phi_V=phi_new,
                n_m3=n_new,
                p_m3=p_new,
                Jn_Am2=Jn_new,
                Jp_Am2=Jp_new,
                dz_m=self._mesh.dz_m,
            )

            if converged:
                logger.debug(
                    "Gummel 수렴: V=%.3f V, %d 회 반복", V_bias, n_iter
                )
                break
        else:
            # max_iterations 초과 시 경고 로그
            J_mid = float(state.Jn_Am2[len(state.Jn_Am2) // 2]
                          + state.Jp_Am2[len(state.Jp_Am2) // 2])
            phi_delta = float(np.max(np.abs(phi_new - state.phi_V)))
            logger.warning(
                "Gummel 미수렴: V=%.3fV, max_iter=%d, |Δφ|_max=%.2e V, "
                "J_mid=%.3e A/m²",
                V_bias, self._cfg.max_iterations, phi_delta, J_mid,
            )

        return BiasPoint(
            voltage_V=V_bias,
            state=state,
            converged=converged,
            n_iterations=n_iter,
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _update_phi(
        self,
        state: DeviceState,
        phi_left: float,
        phi_right: float,
    ) -> np.ndarray:
        """Poisson 방정식 풀기 + 감쇄 혼합."""
        phi_new = self._poisson.solve(state.n_m3, state.p_m3,
                                      phi_left, phi_right)
        alpha = self._cfg.damping
        return alpha * phi_new + (1.0 - alpha) * state.phi_V

    def _update_carriers(
        self,
        state: DeviceState,
        phi_new: np.ndarray,
        n_bc_l: float,
        n_bc_r: float,
        p_bc_l: float,
        p_bc_r: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """전자·정공 연속 방정식 풀기 및 엣지 전류 계산.

        Returns
        -------
        n_new, p_new : 수렴된 캐리어 밀도
        Jn, Jp       : 엣지 전류 밀도 [A/m²]
        """
        n_new = self._continuity.solve_electrons(
            phi_new, state.p_m3, n_bc_l, n_bc_r
        )
        p_new = self._continuity.solve_holes(
            phi_new, n_new, p_bc_l, p_bc_r
        )
        Jn, Jp = self._continuity.compute_currents(phi_new, n_new, p_new)
        return n_new, p_new, Jn, Jp

    def _contact_carriers(self, node_idx: int) -> tuple[float, float]:
        """이상 오믹 접촉 경계 조건 (n, p) 반환.

        양극(node 0, ITO) : 정공 주입 → (n=ni, p=_ANODE_P)
        음극(마지막 노드, Al): 전자 주입 → (n=_CATHODE_N, p=ni)

        _ANODE_P = _CATHODE_N = 1e21 m⁻³ 선택 근거:
          - Debye 길이 ≈ 17 nm ≫ 2 nm 메쉬 → Poisson 수렴 안정
          - 확산 길이 ≈ 50 nm ≫ 2 nm → SG 수치 안정
        """
        # solve() 는 항상 0(양극)과 -1(음극)만 전달함
        is_anode = (node_idx == 0)
        if is_anode:
            return _NI_DEFAULT, _ANODE_P    # (n_low, p_high)
        else:
            return _CATHODE_N, _NI_DEFAULT  # (n_high, p_low)

    def _check_convergence(
        self,
        state: DeviceState,
        phi_new: np.ndarray,
        n_new: np.ndarray,
        p_new: np.ndarray,
    ) -> bool:
        """φ, n, p 모두 수렴 기준 충족 시 True 반환."""
        tol = self._cfg.tolerance

        # 절대 오차 (퍼텐셜)
        if float(np.max(np.abs(phi_new - state.phi_V))) >= tol:
            return False

        # 상대 오차 (캐리어 밀도, 0 나눗셈 방지를 위해 기준값 +1)
        n_ref = np.maximum(np.abs(state.n_m3), 1.0)
        p_ref = np.maximum(np.abs(state.p_m3), 1.0)
        if float(np.max(np.abs(n_new - state.n_m3) / n_ref)) >= tol:
            return False
        if float(np.max(np.abs(p_new - state.p_m3) / p_ref)) >= tol:
            return False

        return True


# ---------------------------------------------------------------------------
# 편의 팩토리
# ---------------------------------------------------------------------------

def build_solver(
    mesh: Mesh1D,
    config: GummelConfig | None = None,
) -> GummelSolver:
    """메쉬에서 바로 사용 가능한 :class:`GummelSolver` 조립.

    내부적으로 :class:`PoissonSolver` 와 :class:`ContinuitySolver` 생성.
    """
    return GummelSolver(
        mesh,
        PoissonSolver(mesh),
        ContinuitySolver(mesh),
        config,
    )
