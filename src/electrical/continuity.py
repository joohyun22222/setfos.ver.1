"""Electron and hole continuity equations — Step 8 target.

Steady-state drift-diffusion formulation:

    Electron current:   Jn =  q μn n E + q Dn ∂n/∂z
    Hole current:       Jp =  q μp p E − q Dp ∂p/∂z
    where  E = −∂φ/∂z,  D = μ kT/q  (Einstein relation)

    Continuity (steady-state):
        ∂Jn/∂z =  q(R − G)
       −∂Jp/∂z =  q(R − G)

Scharfetter-Gummel (SG) discretisation ensures numerical stability
for all ratios of drift to diffusion.  The SG current at edge i+½ is:

    J_{i+½} = (q μ / dz) [B(u) n_{i+1} − B(−u) n_i]

where  u = q(φ_{i+1} − φ_i) / kT  and  B(x) = x / (e^x − 1).

Mathematical building blocks (Bernoulli function, SRH recombination,
Einstein relation) are fully implemented here so Step 8 can assemble
the SG matrix directly.  :meth:`solve_electrons` / :meth:`solve_holes`
raise :exc:`NotImplementedError` until the assembly is written.
"""

from __future__ import annotations

import numpy as np

from .mesh import KB, Mesh1D, Q, T0


_KT_J = KB * T0    # kT at 300 K [J]


class ContinuitySolver:
    """Electron and hole continuity equation solver.

    Parameters
    ----------
    mesh : assembled :class:`~src.electrical.mesh.Mesh1D`
    """

    def __init__(self, mesh: Mesh1D) -> None:
        self._mesh  = mesh
        self._mu_n  = mesh.mu_n_array()   # (N,)
        self._mu_p  = mesh.mu_p_array()   # (N,)

    # ------------------------------------------------------------------
    # Solver interface (stubs — Step 8)
    # ------------------------------------------------------------------

    def solve_electrons(
        self,
        phi: np.ndarray,
        p: np.ndarray,
        n_left: float,
        n_right: float,
    ) -> np.ndarray:
        """Return n(z) satisfying the electron continuity equation.

        Parameters
        ----------
        phi     : (N,) electrostatic potential [V]
        p       : (N,) hole density [m⁻³] (for recombination term)
        n_left  : electron density at anode contact [m⁻³]
        n_right : electron density at cathode contact [m⁻³]

        Raises
        ------
        NotImplementedError
            Step 8 will implement the Scharfetter-Gummel matrix assembly.
        """
        raise NotImplementedError(
            "Electron continuity solver not yet implemented. "
            "Step 8 will add the Scharfetter-Gummel discretisation."
        )

    def solve_holes(
        self,
        phi: np.ndarray,
        n: np.ndarray,
        p_left: float,
        p_right: float,
    ) -> np.ndarray:
        """Return p(z) satisfying the hole continuity equation.

        Parameters
        ----------
        phi     : (N,) electrostatic potential [V]
        n       : (N,) electron density [m⁻³] (for recombination term)
        p_left  : hole density at anode contact [m⁻³]
        p_right : hole density at cathode contact [m⁻³]

        Raises
        ------
        NotImplementedError
            Step 8 will implement the Scharfetter-Gummel matrix assembly.
        """
        raise NotImplementedError(
            "Hole continuity solver not yet implemented. "
            "Step 8 will add the Scharfetter-Gummel discretisation."
        )

    # ------------------------------------------------------------------
    # Mathematical building blocks (available for Step 8)
    # ------------------------------------------------------------------

    def bernoulli(self, x: float | np.ndarray) -> float | np.ndarray:
        """Bernoulli function B(x) = x / (e^x − 1), numerically stable.

        For |x| < 1e-8: B(x) ≈ 1 − x/2 (Taylor series, avoids 0/0).
        Used in the SG discretisation as:
            J_{i+½} ∝ B(u) n_{i+1} − B(−u) n_i,  u = q·Δφ/kT.
        """
        x = np.asarray(x, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            exact = np.where(np.expm1(x) != 0, x / np.expm1(x), 0.0)
        result = np.where(np.abs(x) < 1e-8, 1.0 - 0.5 * x, exact)
        return float(result) if result.ndim == 0 else result

    def recombination_srh(
        self,
        n: np.ndarray,
        p: np.ndarray,
        ni: float = 1e10,
    ) -> np.ndarray:
        """Shockley-Read-Hall recombination rate R(z) [m⁻³/s].

            R = (n·p − ni²) / [τ_p·(n + ni) + τ_n·(p + ni)]

        At thermal equilibrium (np = ni²) → R = 0.

        Parameters
        ----------
        n, p : carrier densities [m⁻³]
        ni   : intrinsic carrier density [m⁻³]
        """
        tau_n = np.array([pr.tau_n for pr in self._mesh.node_props])
        tau_p = np.array([pr.tau_p for pr in self._mesh.node_props])
        ni2   = ni ** 2
        denom = tau_p * (n + ni) + tau_n * (p + ni)
        return np.where(denom > 0, (n * p - ni2) / denom, 0.0)

    def diffusion_coefficient(self, mu: np.ndarray) -> np.ndarray:
        """Einstein relation D = μ · kT/q [m²/s].

        Parameters
        ----------
        mu : mobility array [m²/V/s], shape (N,) or scalar
        """
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
        """Scharfetter-Gummel electron current at a single edge [A/m²].

        J_{i→j} = (q μ / dz) [B(u) n_j − B(−u) n_i]
        where u = q(φ_j − φ_i) / kT.

        Sign convention: positive J flows from node i to node j.
        """
        u = Q * (phi_j - phi_i) / _KT_J
        return float(Q * mu_edge / dz * (self.bernoulli(u) * n_j - self.bernoulli(-u) * n_i))
