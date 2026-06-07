"""Bias sweep orchestrator for J-V characterisation.

:class:`BiasSweepRunner` linearly scans the applied anode voltage from
``v_start`` to ``v_end``, solving the drift-diffusion system at each
bias point via the attached :class:`~src.electrical.drift_diffusion.GummelSolver`.
Warm-starting (initial state from the previous converged point) is used
by default to accelerate convergence across consecutive bias steps.
"""

from __future__ import annotations

import numpy as np

from .drift_diffusion import GummelSolver
from .models import BiasPoint, SweepResult


class BiasSweepRunner:
    """Voltage sweep driver.

    Parameters
    ----------
    gummel    : assembled Gummel solver
    v_start   : first applied anode voltage [V]
    v_end     : last applied anode voltage [V]
    n_points  : number of uniformly-spaced bias points
    v_cathode : cathode reference potential [V] (default 0)
    """

    def __init__(
        self,
        gummel: GummelSolver,
        v_start: float = 0.0,
        v_end: float = 5.0,
        n_points: int = 6,
        v_cathode: float = 0.0,
    ) -> None:
        self._gummel   = gummel
        self.v_start   = v_start
        self.v_end     = v_end
        self.n_points  = n_points
        self.v_cathode = v_cathode

    @property
    def voltages(self) -> np.ndarray:
        """Planned anode voltage array [V], shape (n_points,)."""
        return np.linspace(self.v_start, self.v_end, self.n_points)

    def run(self) -> SweepResult:
        """Execute the voltage sweep and return a :class:`SweepResult`.

        Iterates from *v_start* to *v_end*, passing the previous
        :class:`~src.electrical.models.DeviceState` as the warm-start
        initial guess for the next bias point.
        """
        bias_points: list[BiasPoint] = []
        prev_state = None

        for V in self.voltages:
            bp = self._gummel.solve(
                V_anode=V,
                V_cathode=self.v_cathode,
                initial_state=prev_state,
            )
            bias_points.append(bp)
            prev_state = bp.state   # warm start

        return SweepResult(bias_points)
