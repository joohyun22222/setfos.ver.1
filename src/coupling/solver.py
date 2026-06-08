"""Electro-optical coupling solver (Step 10).

CoupledEmissionSolver bridges the Step 8-9 electrical outputs
(RecombinationProfile, SweepResult) with the Step 1-6 optical outputs
(FieldProfileResult, EmitterProfile) to produce physically consistent
G-weighted EL spectra and luminance values.

Data flow
---------
  RecombinationProfile.G_singlet_m3s   (electrical grid)
    │  grid.interpolate_to_optical_grid()
    ▼
  G_opt   (optical grid, full device)
    │  grid.eml_mask()
    ▼
  G_singlet_eml, z_eml_nm   (EML sub-grid)
    │  spectrum.normalize_generation()
    ▼
  G_norm                                        FieldProfileResult
    │  spectrum.weighted_emission_integral()  ◄─  .E_squared[eml_rows, :]
    ▼
  raw_integral (N_wl,)
    │  spectrum.apply_pl_and_normalize(·, PL)
    ▼
  emission_norm (N_wl,)
    ├─► luminance.compute_luminance()  → L [cd/m²]
    └─► luminance.compute_cd_per_A()   → cd/A

  All → ElectroOpticalResult
"""

from __future__ import annotations

import numpy as np

from ..electrical.models import RecombinationProfile, SweepResult
from ..emission.models import EmitterProfile
from ..optics.field_profile import FieldProfileResult

from .grid import eml_mask, interpolate_to_optical_grid
from .luminance import compute_cd_per_A, compute_luminance
from .models import CoupledSweepResult, ElectroOpticalResult
from .spectrum import apply_pl_and_normalize, normalize_generation, weighted_emission_integral


class CoupledEmissionSolver:
    """Electro-optical coupling solver.

    Combines a pre-computed FieldProfileResult and EmitterProfile with
    RecombinationProfile objects to compute G-weighted EL spectra and
    photometric quantities at each bias point.

    Parameters
    ----------
    field_profile   : pre-computed FieldProfileResult (Step 5)
    emitter_profile : EmitterProfile with PL spectrum + layer name
    emitter_layer   : name of the EML layer (must exist in field_profile.layer_names)
    eta_rad         : radiative quantum efficiency in [0, 1].
                      Absorbs η_spin: pass PLQY×1.0 (phosphorescent) or
                      PLQY×0.25 (fluorescent).
    eta_out         : outcoupling efficiency in [0, 1]

    Notes
    -----
    This class holds no mutable state after construction; solve() and
    solve_sweep() are stateless over their inputs.
    """

    def __init__(
        self,
        field_profile: FieldProfileResult,
        emitter_profile: EmitterProfile,
        emitter_layer: str,
        eta_rad: float = 1.0,
        eta_out: float = 1.0,
    ) -> None:
        self._fp        = field_profile
        self._emitter   = emitter_profile
        self._eml_name  = emitter_layer
        self._eta_rad   = eta_rad
        self._eta_out   = eta_out

        # Pre-compute EML mask (same for all bias points)
        self._mask = eml_mask(
            field_profile.z_nm,
            field_profile.layer_boundaries_nm,
            field_profile.layer_names,
            emitter_layer,
        )
        self._eml_z_nm = field_profile.z_nm[self._mask]

        # Pre-interpolate PL onto optical wavelength grid
        self._pl = emitter_profile.pl_at(field_profile.wavelength_nm)

        # Pre-extract |E|² sub-matrix restricted to EML rows
        self._E2_eml = field_profile.E_squared[self._mask, :]   # (N_eml, N_wl)

    def solve(
        self,
        rec_profile: RecombinationProfile,
        J_Am2: float,
    ) -> ElectroOpticalResult:
        """Compute electro-optical result for one bias point.

        Parameters
        ----------
        rec_profile : RecombinationProfile at the target bias
        J_Am2       : current density [A/m²] (from BiasPoint.state.J_total)

        Returns
        -------
        ElectroOpticalResult
        """
        # 1. Interpolate G_singlet from electrical mesh to optical mesh
        G_opt = interpolate_to_optical_grid(
            rec_profile.z_nm,
            rec_profile.G_singlet_m3s,
            self._fp.z_nm,
        )

        # 2. Restrict to EML
        G_eml = G_opt[self._mask]

        # 3. Normalise G within EML
        G_norm = normalize_generation(G_eml, self._eml_z_nm)

        # 4. Weighted emission integral: ∫G_norm × |E|² dz  →  (N_wl,)
        raw = weighted_emission_integral(G_norm, self._eml_z_nm, self._E2_eml)

        # 5. Apply PL and area-normalise
        emission_norm = apply_pl_and_normalize(raw, self._pl)

        # 6. Photometrics
        L = compute_luminance(
            self._fp.wavelength_nm,
            emission_norm,
            J_Am2,
            self._eta_rad,
            self._eta_out,
        )
        cd_A = compute_cd_per_A(L, J_Am2)

        return ElectroOpticalResult(
            voltage_V=rec_profile.voltage_V,
            J_Am2=J_Am2,
            wavelength_nm=self._fp.wavelength_nm.copy(),
            emission_spectrum=emission_norm,
            luminance_cd_m2=L,
            cd_per_A=cd_A,
            G_norm_on_opt_grid=G_norm,
            eml_z_nm=self._eml_z_nm.copy(),
        )

    def solve_sweep(
        self,
        profiles: list[RecombinationProfile],
        sweep: SweepResult,
    ) -> CoupledSweepResult:
        """Batch-process a full voltage sweep.

        Parameters
        ----------
        profiles : list[RecombinationProfile] — one per bias point,
                   same order as sweep.bias_points.
                   Typically from RecombinationSweep.compute_all(sweep).
        sweep    : SweepResult from BiasSweepRunner.run()

        Returns
        -------
        CoupledSweepResult

        Raises
        ------
        ValueError if len(profiles) != len(sweep.bias_points).
        """
        if len(profiles) != len(sweep.bias_points):
            raise ValueError(
                f"Length mismatch: {len(profiles)} profiles vs "
                f"{len(sweep.bias_points)} bias points."
            )

        results: list[ElectroOpticalResult] = []
        for rec, bp in zip(profiles, sweep.bias_points):
            J = bp.state.J_total
            results.append(self.solve(rec, J))

        return CoupledSweepResult(results)
