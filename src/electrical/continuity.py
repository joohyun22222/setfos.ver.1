"""전자·정공 연속 방정식 — Scharfetter-Gummel 이산화.

정상 상태 drift-diffusion 공식:

    전자 전류:   Jn =  q μn n E + q Dn ∂n/∂z
    정공 전류:   Jp =  q μp p E − q Dp ∂p/∂z
    여기서  E = −∂φ/∂z,  D = μ kT/q  (아인슈타인 관계)

    연속 방정식 (정상):
        ∂Jn/∂z =  q R
        ∂Jp/∂z = −q R

Scharfetter-Gummel (SG) 이산화:
    엣지 i+½ 에서 전자 전류:
        Jn[i] = C[i] · ( B(u[i])·n[i+1] − B(−u[i])·n[i] )

    엣지 i+½ 에서 정공 전류:
        Jp[i] = C_p[i] · ( B(−u[i])·p[i+1] − B(u[i])·p[i] )

    u[i] = q(φ[i+1]−φ[i]) / kT,  C[i] = q μ_edge[i] / dz[i]

연립방정식 (전자, 내부 노드 i):
    A[i,i−1] =  C[i−1]·B(−u[i−1]) / dz_box
    A[i,i]   = −(C[i−1]·B(u[i−1]) + C[i]·B(−u[i])) / dz_box − q·a_n[i]
    A[i,i+1] =  C[i]·B(u[i]) / dz_box
    rhs[i]   = −q·c_n[i]

연립방정식 (정공, 내부 노드 i):
    A[i,i−1] =  C_p[i−1]·B(u[i−1]) / dz_box
    A[i,i]   = −(C_p[i−1]·B(−u[i−1]) + C_p[i]·B(u[i])) / dz_box + q·a_p[i]
    A[i,i+1] =  C_p[i]·B(−u[i]) / dz_box
    rhs[i]   =  q·c_p[i]

선형화된 SRH (Gummel 분리):
    전자: R ≈ a_n·n − c_n,  a_n = p/D,  c_n = ni²/D
          D = τ_p·(2ni) + τ_n·(p+ni)
    정공: R ≈ a_p·p − c_p,  a_p = n/D,  c_p = ni²/D
          D = τ_p·(n+ni) + τ_n·(2ni)
"""

from __future__ import annotations

import logging

import numpy as np

try:
    from scipy.linalg import solve_banded as _solve_banded
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from .mesh import KB, Mesh1D, Q, T0

_KT_J       = KB * T0      # kT at 300 K [J]
_NI_DEFAULT = 1e10         # 기본 진성 캐리어 밀도 [m⁻³]
_MAX_CARRIER = 1e30        # 캐리어 밀도 상한 (발산 방지)

logger = logging.getLogger(__name__)


class ContinuitySolver:
    """전자·정공 연속 방정식 솔버 (Scharfetter-Gummel).

    Parameters
    ----------
    mesh : 조립된 :class:`~src.electrical.mesh.Mesh1D`
    """

    def __init__(self, mesh: Mesh1D) -> None:
        self._mesh  = mesh
        self._mu_n  = mesh.mu_n_array()   # (N,)
        self._mu_p  = mesh.mu_p_array()   # (N,)

        # 엣지 조화평균 이동도 — 계면 불연속 처리
        mn, mp = self._mu_n, self._mu_p
        self._mu_n_edge = 2.0 * mn[:-1] * mn[1:] / (mn[:-1] + mn[1:] + 1e-300)
        self._mu_p_edge = 2.0 * mp[:-1] * mp[1:] / (mp[:-1] + mp[1:] + 1e-300)

        # SRH 수명 배열
        self._tau_n = np.array([pr.tau_n for pr in mesh.node_props])
        self._tau_p = np.array([pr.tau_p for pr in mesh.node_props])

        # Trap-state SRH statistics (Step 13); default = nᵢ = midgap
        self._trap_n1 = np.array([pr.trap_n1 for pr in mesh.node_props])
        self._trap_p1 = np.array([pr.trap_p1 for pr in mesh.node_props])

    # ------------------------------------------------------------------
    # 공개 API: 솔버
    # ------------------------------------------------------------------

    def solve_electrons(
        self,
        phi: np.ndarray,
        p: np.ndarray,
        n_left: float,
        n_right: float,
    ) -> np.ndarray:
        """Scharfetter-Gummel 이산화로 전자 연속 방정식 풀기.

        Parameters
        ----------
        phi     : (N,) 정전 퍼텐셜 [V]
        p       : (N,) 정공 밀도 [m⁻³]
        n_left  : 양극 전자 경계 조건 [m⁻³]
        n_right : 음극 전자 경계 조건 [m⁻³]

        Returns
        -------
        n : (N,) 전자 밀도 [m⁻³]
        """
        N  = self._mesh.num_nodes
        dz = self._mesh.dz_m                         # (N-1,)
        C  = Q * self._mu_n_edge / dz                # (N-1,) 전류 계수

        u   = Q * np.diff(phi) / _KT_J              # (N-1,) 정규화 전압 차이
        Bu  = self.bernoulli(u)                       # B(u)
        Bmu = self.bernoulli(-u)                      # B(-u)

        # 선형화된 SRH — trap-state 통계 적용 (Step 13)
        # D = τ_p·(2·n₁) + τ_n·(p + p₁);  midgap 기본: n₁=p₁=nᵢ
        ni  = _NI_DEFAULT
        n1  = self._trap_n1
        p1  = self._trap_p1
        D   = np.maximum(self._tau_p * (2.0 * n1) + self._tau_n * (p + p1), 1e-300)
        a_n = p / D       # R ≈ a_n·n − c_n
        c_n = ni**2 / D

        # 벡터화된 3중 대각 조립
        dz_box = 0.5 * (dz[:-1] + dz[1:])            # (N-2,) 내부 박스 폭

        sub  = C[:-1] * Bmu[:-1] / dz_box             # (N-2,) 하부 대각
        sup  = C[1:]  * Bu[1:]   / dz_box             # (N-2,) 상부 대각
        diag_int = (-(C[:-1] * Bu[:-1] + C[1:] * Bmu[1:]) / dz_box
                    - Q * a_n[1:N-1])                  # (N-2,) 내부 대각

        # 전체 대각 벡터 (경계 포함)
        diag = np.empty(N)
        diag[0]    = 1.0
        diag[1:-1] = diag_int
        diag[-1]   = 1.0

        lower = np.zeros(N)   # lower[i] = A[i, i-1]
        upper = np.zeros(N)   # upper[i] = A[i, i+1]
        lower[1:-1] = sub
        upper[1:-1] = sup

        rhs       = np.empty(N)
        rhs[0]    = n_left
        rhs[1:-1] = -Q * c_n[1:N-1]
        rhs[-1]   = n_right

        n = self._tridiag_solve(lower, diag, upper, rhs)
        return np.clip(n, 1.0, _MAX_CARRIER)

    def solve_holes(
        self,
        phi: np.ndarray,
        n: np.ndarray,
        p_left: float,
        p_right: float,
    ) -> np.ndarray:
        """Scharfetter-Gummel 이산화로 정공 연속 방정식 풀기.

        Parameters
        ----------
        phi     : (N,) 정전 퍼텐셜 [V]
        n       : (N,) 전자 밀도 [m⁻³]
        p_left  : 양극 정공 경계 조건 [m⁻³]
        p_right : 음극 정공 경계 조건 [m⁻³]

        Returns
        -------
        p : (N,) 정공 밀도 [m⁻³]
        """
        N  = self._mesh.num_nodes
        dz = self._mesh.dz_m
        Cp = Q * self._mu_p_edge / dz                 # (N-1,)

        u   = Q * np.diff(phi) / _KT_J
        Bu  = self.bernoulli(u)
        Bmu = self.bernoulli(-u)

        # 선형화된 SRH — trap-state 통계 적용 (Step 13)
        # D = τ_n·(2·p₁) + τ_p·(n + n₁);  midgap 기본: n₁=p₁=nᵢ
        ni  = _NI_DEFAULT
        n1  = self._trap_n1
        p1  = self._trap_p1
        D   = np.maximum(self._tau_n * (2.0 * p1) + self._tau_p * (n + n1), 1e-300)
        a_p = n / D
        c_p = ni**2 / D

        dz_box = 0.5 * (dz[:-1] + dz[1:])

        # 정공은 드리프트 방향이 반대 → B(u)/B(-u) 위치 교환
        sub  = Cp[:-1] * Bu[:-1]  / dz_box
        sup  = Cp[1:]  * Bmu[1:]  / dz_box
        diag_int = (-(Cp[:-1] * Bmu[:-1] + Cp[1:] * Bu[1:]) / dz_box
                    + Q * a_p[1:N-1])

        diag = np.empty(N)
        diag[0]    = 1.0
        diag[1:-1] = diag_int
        diag[-1]   = 1.0

        lower = np.zeros(N)
        upper = np.zeros(N)
        lower[1:-1] = sub
        upper[1:-1] = sup

        rhs       = np.empty(N)
        rhs[0]    = p_left
        rhs[1:-1] = Q * c_p[1:N-1]
        rhs[-1]   = p_right

        p = self._tridiag_solve(lower, diag, upper, rhs)
        return np.clip(p, 1.0, _MAX_CARRIER)

    def compute_currents(
        self,
        phi: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """수렴 후 엣지 전류 밀도 계산.

        Returns
        -------
        Jn : (N-1,) 전자 전류 밀도 [A/m²]
        Jp : (N-1,) 정공 전류 밀도 [A/m²]
        """
        dz  = self._mesh.dz_m
        Cn  = Q * self._mu_n_edge / dz
        Cp  = Q * self._mu_p_edge / dz

        u   = Q * np.diff(phi) / _KT_J
        Bu  = self.bernoulli(u)
        Bmu = self.bernoulli(-u)

        # Jn[i] = C[i]·(B(u)·n[i+1] − B(−u)·n[i])
        Jn = Cn * (Bu * n[1:] - Bmu * n[:-1])
        # Jp[i] = C_p[i]·(B(−u)·p[i+1] − B(u)·p[i])
        Jp = Cp * (Bmu * p[1:] - Bu  * p[:-1])

        return Jn, Jp

    # ------------------------------------------------------------------
    # 내부: 3중 대각 선형 시스템 풀기
    # ------------------------------------------------------------------

    @staticmethod
    def _tridiag_solve(
        lower: np.ndarray,
        diag: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """3중 대각 시스템 풀기.

        scipy 가 있으면 solve_banded 사용, 없으면 numpy 일반 풀이 사용.

        Parameters
        ----------
        lower : (N,) 하부 대각선, lower[i] = A[i, i-1]
        diag  : (N,) 주 대각선
        upper : (N,) 상부 대각선, upper[i] = A[i, i+1]
        rhs   : (N,) 우변 벡터
        """
        if _HAVE_SCIPY:
            # scipy 형식: ab[0,1:]=upper, ab[1,:]=diag, ab[2,:-1]=lower
            N  = len(diag)
            ab = np.zeros((3, N))
            ab[0, 1:]  = upper[:-1]  # 상부 대각 (인덱스 오프셋)
            ab[1, :]   = diag
            ab[2, :-1] = lower[1:]   # 하부 대각 (인덱스 오프셋)
            return _solve_banded((1, 1), ab, rhs)
        else:
            # scipy 없을 때 일반 풀이 (느리지만 호환 가능)
            N = len(diag)
            A = (np.diag(diag)
                 + np.diag(lower[1:], -1)
                 + np.diag(upper[:-1], 1))
            return np.linalg.solve(A, rhs)

    # ------------------------------------------------------------------
    # 수학적 보조 함수
    # ------------------------------------------------------------------

    def bernoulli(self, x: float | np.ndarray) -> float | np.ndarray:
        """Bernoulli 함수 B(x) = x / (e^x − 1), 수치 안정 버전.

        |x| < 1e-8: B(x) ≈ 1 − x/2 (테일러 전개).
        x ≫ 1    : B(x) ≈ 0.
        x ≪ −1  : B(x) ≈ |x|.
        """
        x = np.asarray(x, dtype=float)
        # 대규모 overflow 방지: |x| > 500 은 B(x)→0 또는 B(x)→|x|
        x_clamped = np.clip(x, -500.0, 500.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expm1_val = np.expm1(x_clamped)
            safe      = np.where(expm1_val != 0.0,
                                 x_clamped / expm1_val,
                                 0.0)
        taylor = 1.0 - 0.5 * x
        result = np.where(np.abs(x) < 1e-8, taylor, safe)
        result = result if result.ndim > 0 else float(result)
        return result

    def recombination_srh(
        self,
        n: np.ndarray,
        p: np.ndarray,
        ni: float = 1e10,
    ) -> np.ndarray:
        """Shockley-Read-Hall 재결합률 R(z) [m⁻³/s].

            R = (np − ni²) / [τ_p·(n + n₁) + τ_n·(p + p₁)]

        trap-state 통계(n₁, p₁)를 사용; 기본값 n₁=p₁=nᵢ (midgap).
        열평형 (np = ni²) 에서 R = 0.
        """
        ni2   = ni ** 2
        denom = self._tau_p * (n + self._trap_n1) + self._tau_n * (p + self._trap_p1)
        return np.where(denom > 0, (n * p - ni2) / denom, 0.0)

    def diffusion_coefficient(self, mu: np.ndarray) -> np.ndarray:
        """아인슈타인 관계: D = μ · kT/q [m²/s]."""
        return mu * _KT_J / Q

    def sg_current_edge(
        self,
        phi_i: float,
        phi_j: float,
        n_i: float,
        n_j: float,
        mu_edge: float,
        dz: float,
    ) -> float:
        """단일 엣지에서 Scharfetter-Gummel 전자 전류 [A/m²].

        J_{i→j} = (q μ / dz) [B(u) n_j − B(−u) n_i],  u = q(φ_j−φ_i)/kT.
        """
        u = Q * (phi_j - phi_i) / _KT_J
        return float(Q * mu_edge / dz
                     * (self.bernoulli(u) * n_j - self.bernoulli(-u) * n_i))
