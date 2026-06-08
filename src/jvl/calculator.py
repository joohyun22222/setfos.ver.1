"""J-V-L calculator — full electro-optical sweep orchestrator (Steps 11-12).

Pipeline
--------
  BiasSweepRunner          : electrical J-V sweep (Gummel)
    │ SweepResult
  RecombinationSweep       : SRH + Langevin recombination profiles
    │ list[RecombinationProfile]
  CoupledEmissionSolver    : G-weighted EL spectra + luminance
    │ CoupledSweepResult
  JVLCalculator._build     : assemble JVLResult with EQE/CE/PE
"""

from __future__ import annotations

from ..coupling.solver import CoupledEmissionSolver
from ..electrical.bias_sweep import BiasSweepRunner
from ..electrical.drift_diffusion import GummelSolver
from ..electrical.recombination import RecombinationSolver, RecombinationSweep
from ..emission.models import EmitterProfile
from ..optics.field_profile import FieldProfileResult
from .efficiency import compute_ce, compute_eqe, compute_pe
from .models import JVLPoint, JVLResult


class JVLCalculator:
    """Full J-V-L sweep calculator with EQE, CE, PE.

    Orchestrates the electrical solver, recombination solver, and
    electro-optical coupling solver to produce a complete J-V-L
    characterisation of an OLED device.

    Parameters
    ----------
    gummel          : assembled :class:`~src.electrical.drift_diffusion.GummelSolver`
    rec_solver      : :class:`~src.electrical.recombination.RecombinationSolver`
    field_profile   : pre-computed :class:`~src.optics.field_profile.FieldProfileResult`
    emitter_profile : :class:`~src.emission.models.EmitterProfile` with PL spectrum
    emitter_layer   : name of the EML layer in field_profile
    eta_rad         : radiative efficiency in [0, 1].
                      Absorbs η_spin: PLQY×1.0 (phosphorescent)
                      or PLQY×0.25 (fluorescent).
    eta_out         : outcoupling efficiency in [0, 1]
    """

    def __init__(
        self,
        gummel: GummelSolver,
        rec_solver: RecombinationSolver,
        field_profile: FieldProfileResult,
        emitter_profile: EmitterProfile,
        emitter_layer: str,
        eta_rad: float = 1.0,
        eta_out: float = 1.0,
    ) -> None:
        self._gummel          = gummel
        self._rec_solver      = rec_solver
        self._field_profile   = field_profile
        self._emitter_profile = emitter_profile
        self._emitter_layer   = emitter_layer
        self._eta_rad         = eta_rad
        self._eta_out         = eta_out

        # Build coupling solver once (expensive pre-computations in __init__)
        self._coupling = CoupledEmissionSolver(
            field_profile   = field_profile,
            emitter_profile = emitter_profile,
            emitter_layer   = emitter_layer,
            eta_rad         = eta_rad,
            eta_out         = eta_out,
        )

    def run(
        self,
        v_start: float = 0.0,
        v_end: float = 5.0,
        n_points: int = 11,
        v_cathode: float = 0.0,
    ) -> JVLResult:
        """Execute a full J-V-L sweep with EQE, CE, PE.

        Parameters
        ----------
        v_start   : first anode voltage [V]
        v_end     : last anode voltage [V]
        n_points  : number of uniformly-spaced bias points
        v_cathode : cathode reference potential [V] (default 0)

        Returns
        -------
        :class:`JVLResult` with per-point J, L, EQE, CE, PE and EL spectra.
        """
        # Step 1 — Electrical J-V sweep
        runner = BiasSweepRunner(
            self._gummel,
            v_start=v_start,
            v_end=v_end,
            n_points=n_points,
            v_cathode=v_cathode,
        )
        sweep = runner.run()

        # Step 2 — Recombination profiles
        rec_sweep = RecombinationSweep(self._rec_solver)
        rec_profiles = rec_sweep.compute_all(sweep)

        # Step 3 — G-weighted EL spectra + luminance
        coupled = self._coupling.solve_sweep(rec_profiles, sweep)

        # Step 4 — Assemble JVLResult with EQE/CE/PE
        points: list[JVLPoint] = []
        for eo in coupled.results:
            eqe = compute_eqe(
                eo.wavelength_nm, eo.emission_spectrum,
                eo.luminance_cd_m2, eo.J_Am2,
            )
            ce = compute_ce(eo.luminance_cd_m2, eo.J_Am2)
            pe = compute_pe(eo.luminance_cd_m2, eo.J_Am2, eo.voltage_V)
            points.append(
                JVLPoint(
                    voltage_V          = eo.voltage_V,
                    J_Am2              = eo.J_Am2,
                    luminance_cd_m2    = eo.luminance_cd_m2,
                    eqe                = eqe,
                    cd_per_A           = ce,
                    lm_per_W           = pe,
                    peak_wavelength_nm = eo.peak_emission_nm,
                    wavelength_nm      = eo.wavelength_nm,
                    emission_spectrum  = eo.emission_spectrum,
                )
            )

        return JVLResult(points)
