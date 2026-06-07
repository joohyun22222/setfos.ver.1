"""1D finite-difference Poisson solver.

Equation (1D, normalised by ε₀):

    d/dz [ ε_r(z) · dφ/dz ] = −q(p − n + N_D) / ε₀

Discretised with box integration and harmonic-mean ε_r at layer
interfaces, yielding a tridiagonal linear system A·φ = b.
Dirichlet BCs (Ohmic metal contacts) are applied to rows 0 and N−1.

Reference
---------
Selberherr, S. *Analysis and Simulation of Semiconductor Devices*, 1984.
"""

from __future__ import annotations

import numpy as np

from .mesh import EPS0, Mesh1D, Q


class PoissonSolver:
    """Finite-difference Poisson solver on a :class:`~src.electrical.mesh.Mesh1D`.

    The assembled matrix A is dense (N × N); for N ≲ 1000 this is fast
    with ``numpy.linalg.solve``.  Step 8 can migrate to scipy sparse.

    Usage
    -----
    solver = PoissonSolver(mesh)
    phi    = solver.solve(n, p, phi_left=V_anode, phi_right=0.0)

    For matrix inspection:
    A, b = solver.assemble(n, p)
    A, b = solver.apply_dirichlet(A, b, phi_left, phi_right)
    """

    def __init__(self, mesh: Mesh1D) -> None:
        self._mesh = mesh
        self._N    = mesh.num_nodes
        self._dz   = mesh.dz_m          # (N-1,) edge spacings [m]
        self._Nd   = mesh.N_doping_array()  # (N,) [m⁻³]

        # Harmonic-mean ε_r at each edge — correct for discontinuous ε
        eps_r = mesh.eps_array()  # (N,)
        eps_l = eps_r[:-1]
        eps_r_ = eps_r[1:]
        self._eps_edge = 2.0 * eps_l * eps_r_ / (eps_l + eps_r_ + 1e-30)  # (N-1,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        n: np.ndarray,
        p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (A, b) for A·φ = b (interior rows only).

        First and last rows are left zeroed; call :meth:`apply_dirichlet`
        before solving.

        Parameters
        ----------
        n, p : (N,) carrier densities [m⁻³]

        Returns
        -------
        A : (N, N) dense ndarray
        b : (N,)   rhs vector
        """
        N  = self._N
        dz = self._dz       # (N-1,)
        ee = self._eps_edge  # (N-1,) harmonic-mean ε_r at edges

        A = np.zeros((N, N))
        b = np.zeros(N)

        for i in range(1, N - 1):
            dz_l = dz[i - 1]            # left edge spacing
            dz_r = dz[i]               # right edge spacing
            dz_c = 0.5 * (dz_l + dz_r) # box half-width for node i

            coeff_l = ee[i - 1] / dz_l
            coeff_r = ee[i]     / dz_r

            A[i, i - 1] =  coeff_l / dz_c
            A[i, i]     = -(coeff_l + coeff_r) / dz_c
            A[i, i + 1] =  coeff_r / dz_c

            # Charge density normalised by ε₀
            rho = Q * (p[i] - n[i] + self._Nd[i])   # [C/m³]
            b[i] = -rho / EPS0                        # [V/m²]

        return A, b

    def apply_dirichlet(
        self,
        A: np.ndarray,
        b: np.ndarray,
        phi_left: float,
        phi_right: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impose Dirichlet BCs on the system A·φ = b (in-place).

        Sets row 0 → φ₀ = phi_left, row N−1 → φ_{N-1} = phi_right.
        """
        A[0, :]  = 0.0;  A[0, 0]   = 1.0;  b[0]  = phi_left
        A[-1, :] = 0.0;  A[-1, -1] = 1.0;  b[-1] = phi_right
        return A, b

    def solve(
        self,
        n: np.ndarray,
        p: np.ndarray,
        phi_left: float,
        phi_right: float,
    ) -> np.ndarray:
        """Solve the Poisson equation for φ(z).

        Parameters
        ----------
        n, p      : (N,) carrier densities [m⁻³]
        phi_left  : Dirichlet potential at anode [V]
        phi_right : Dirichlet potential at cathode [V]

        Returns
        -------
        phi : (N,) electrostatic potential [V]
        """
        A, b = self.assemble(n, p)
        A, b = self.apply_dirichlet(A, b, phi_left, phi_right)
        return np.linalg.solve(A, b)
