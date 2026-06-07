"""Self-consistent Gummel iteration for the drift-diffusion system.

Gummel Algorithm
----------------
Given V_anode and V_cathode, repeat until |Δφ|_∞ < tolerance:

  1. Solve Poisson for φ(z)           given n, p
  2. Solve electron continuity for n  given φ, p
  3. Solve hole continuity for p      given φ, n
  4. Check convergence

Until Step 8 implements the continuity solvers, the loop terminates
after the first Poisson update (continuity stubs raise NotImplementedError)
and returns the equilibrium carrier profiles unchanged.

Convenience factory
-------------------
    mesh   = build_mesh(stack, material_db)
    solver = build_solver(mesh)          # creates Poisson + Continuity internally
    bp     = solver.solve(V_anode=3.0)
"""

from __future__ import annotations

import numpy as np

from .continuity import ContinuitySolver
from .mesh import Mesh1D
from .models import BiasPoint, DeviceState, GummelConfig
from .poisson import PoissonSolver


_NI_DEFAULT = 1e10   # intrinsic carrier density placeholder [m⁻³]


class GummelSolver:
    """Self-consistent Poisson + drift-diffusion solver.

    Parameters
    ----------
    mesh        : 1D mesh with per-node material properties
    poisson     : assembled :class:`~src.electrical.poisson.PoissonSolver`
    continuity  : assembled :class:`~src.electrical.continuity.ContinuitySolver`
    config      : convergence / damping parameters (default :class:`GummelConfig`)
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
    # Public API
    # ------------------------------------------------------------------

    def equilibrium_state(self, V_anode: float = 0.0) -> DeviceState:
        """Construct a thermal-equilibrium initial state.

        Uses a linear potential profile as initial guess for φ and
        intrinsic carrier densities (n = p = ni) throughout the device.

        Parameters
        ----------
        V_anode : anode contact potential [V] (reference: cathode = 0 V)
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
        """Run the Gummel iteration at a fixed bias voltage.

        Parameters
        ----------
        V_anode       : anode contact potential [V]
        V_cathode     : cathode reference potential [V] (default 0)
        initial_state : warm-start from the previous bias point;
                        defaults to :meth:`equilibrium_state`

        Returns
        -------
        :class:`BiasPoint` with converged :class:`DeviceState`.

        Notes
        -----
        When the continuity solvers are not yet implemented (Step 8),
        the loop terminates after one Poisson iteration.  The
        ``converged`` flag then reflects only the Poisson update.
        """
        V_bias = V_anode - V_cathode
        state  = initial_state or self.equilibrium_state(V_anode)

        n_iter = 0
        converged = False

        for it in range(self._cfg.max_iterations):
            n_iter = it + 1

            # ── Step 1: Poisson ─────────────────────────────────────
            phi_new = self._update_phi(state, V_anode, V_cathode)

            # ── Steps 2–3: Continuity (stubs until Step 8) ──────────
            n_new, p_new, cont_solved = self._update_carriers(
                state, phi_new, V_anode, V_cathode
            )

            # ── Convergence check ────────────────────────────────────
            converged = self._check_convergence(state.phi_V, phi_new)

            state = DeviceState(
                voltage_V=V_bias,
                phi_V=phi_new,
                n_m3=n_new,
                p_m3=p_new,
                Jn_Am2=np.zeros(self._mesh.num_edges),
                Jp_Am2=np.zeros(self._mesh.num_edges),
                dz_m=self._mesh.dz_m,
            )

            if converged or not cont_solved:
                break

        return BiasPoint(
            voltage_V=V_bias,
            state=state,
            converged=converged,
            n_iterations=n_iter,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_phi(
        self,
        state: DeviceState,
        phi_left: float,
        phi_right: float,
    ) -> np.ndarray:
        phi_new = self._poisson.solve(state.n_m3, state.p_m3, phi_left, phi_right)
        alpha = self._cfg.damping
        return alpha * phi_new + (1.0 - alpha) * state.phi_V

    def _update_carriers(
        self,
        state: DeviceState,
        phi_new: np.ndarray,
        V_anode: float,
        V_cathode: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Attempt to solve continuity equations.

        Returns (n, p, solved_flag).  solved_flag=False if stubs raised
        NotImplementedError (Step 8 not yet implemented).
        """
        try:
            n_bc_l = self._ohmic_n(0)
            n_bc_r = self._ohmic_n(-1)
            p_bc_l = self._ohmic_p(0)
            p_bc_r = self._ohmic_p(-1)
            n_new = self._continuity.solve_electrons(phi_new, state.p_m3, n_bc_l, n_bc_r)
            p_new = self._continuity.solve_holes(phi_new, n_new, p_bc_l, p_bc_r)
            return n_new, p_new, True
        except NotImplementedError:
            return state.n_m3, state.p_m3, False

    def _ohmic_n(self, node_idx: int) -> float:
        """Electron density at an Ohmic contact (charge-neutral)."""
        Nd = max(self._mesh.node_props[node_idx].N_doping, 0.0)
        return Nd if Nd > 0 else _NI_DEFAULT

    def _ohmic_p(self, node_idx: int) -> float:
        """Hole density at an Ohmic contact (charge-neutral)."""
        Na = max(-self._mesh.node_props[node_idx].N_doping, 0.0)
        return Na if Na > 0 else _NI_DEFAULT

    def _check_convergence(
        self, phi_old: np.ndarray, phi_new: np.ndarray
    ) -> bool:
        return float(np.max(np.abs(phi_new - phi_old))) < self._cfg.tolerance


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_solver(
    mesh: Mesh1D,
    config: GummelConfig | None = None,
) -> GummelSolver:
    """Assemble a ready-to-use :class:`GummelSolver` from a :class:`Mesh1D`.

    Creates :class:`~src.electrical.poisson.PoissonSolver` and
    :class:`~src.electrical.continuity.ContinuitySolver` internally.
    """
    return GummelSolver(
        mesh,
        PoissonSolver(mesh),
        ContinuitySolver(mesh),
        config,
    )
