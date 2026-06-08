"""Tests for Step 10 — Electro-Optical Coupling.

Coverage
--------
TestGridInterpolation        (5) — interpolation correctness, EML masking
TestGenerationNormalization  (4) — unit integral, fallback, edge cases
TestWeightedEmissionIntegral (4) — shape, uniform-G, zero-G, delta-G
TestApplyPLAndNormalize      (3) — normalisation, zero PL
TestCIEPhotopic              (2) — peak at 555 nm, non-negative
TestLuminanceFormula         (4) — dark, forward, units, linearity
TestCoupledEmissionSolver    (5) — end-to-end coupling at one bias point
TestSolveSweep               (3) — batch sweep, length check, mismatch
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
NK_ROOT = ROOT / "data" / "nk"

from src.io import load_project, load_emitter_config
from src.emission import EmissionSolver
from src.optics.field_profile import compute_field_profile, FieldProfileResult
from src.electrical import (
    build_mesh, build_solver, GummelConfig,
    RecombinationProfile, SweepResult, BiasPoint,
    BiasSweepRunner,
)
from src.electrical.recombination import RecombinationSolver, RecombinationSweep
from src.electrical.continuity import ContinuitySolver

from src.coupling import CoupledEmissionSolver, ElectroOpticalResult, CoupledSweepResult
from src.coupling.grid import (
    interpolate_to_optical_grid, eml_mask, eml_layer_bounds,
)
from src.coupling.spectrum import (
    normalize_generation, weighted_emission_integral, apply_pl_and_normalize,
)
from src.coupling.luminance import (
    cie_photopic_response, compute_luminance, compute_cd_per_A,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (shared, expensive)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def emitter_cfg():
    return load_emitter_config(SAMPLES / "emitter_irppy3.yaml")


@pytest.fixture(scope="module")
def field_profile(project) -> FieldProfileResult:
    """Pre-compute optical field profile for the OLED sample stack."""
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
    # Access the internal field profile via the solve path
    result = solver.solve(
        project.device_stack,
        project.material_db,
        project.solver_config,
        load_emitter_config(SAMPLES / "emitter_irppy3.yaml"),
    )
    # Rebuild field profile directly to get FieldProfileResult
    from src.emission.solver import EmissionSolver as _ES
    import numpy as _np
    _wl = _np.arange(
        project.solver_config.wavelength_grid.start_nm,
        project.solver_config.wavelength_grid.end_nm + 0.5,
        project.solver_config.wavelength_grid.step_nm,
    )
    stack = project.device_stack
    mat_db = project.material_db
    n_layers, d_layers, layer_names, _ = _ES(nk_root=NK_ROOT, project_root=ROOT)._build_optical_data(
        stack, mat_db, _wl
    )
    return compute_field_profile(
        n_layers, d_layers, _wl,
        layer_names=layer_names,
        n_inc=1.0, n_sub=1.5,
    )


@pytest.fixture(scope="module")
def mesh(project):
    return build_mesh(project.device_stack, project.material_db, z_resolution_nm=1.0)


@pytest.fixture(scope="module")
def gummel(mesh):
    return build_solver(mesh, GummelConfig(max_iterations=5, tolerance=1e-4))


@pytest.fixture(scope="module")
def sweep_result(gummel) -> SweepResult:
    runner = BiasSweepRunner(gummel, v_start=0.0, v_end=2.0, n_points=3)
    return runner.run()


@pytest.fixture(scope="module")
def rec_profiles(mesh, gummel, sweep_result) -> list[RecombinationProfile]:
    cont = ContinuitySolver(mesh)
    rec_solver = RecombinationSolver(mesh, cont)
    sweep = RecombinationSweep(rec_solver)
    return sweep.compute_all(sweep_result)


@pytest.fixture(scope="module")
def emitter_profile(project, emitter_cfg, field_profile):
    """Build EmitterProfile with PL interpolated onto optical wavelength grid."""
    from src.emission.models import EmitterProfile
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
    return solver._load_emitter(emitter_cfg, field_profile.wavelength_nm)


@pytest.fixture(scope="module")
def coupled_solver(field_profile, emitter_profile) -> CoupledEmissionSolver:
    return CoupledEmissionSolver(
        field_profile=field_profile,
        emitter_profile=emitter_profile,
        emitter_layer="EML",
        eta_rad=1.0,
        eta_out=1.0,
    )


@pytest.fixture(scope="module")
def rec_profile_gaussian(mesh) -> RecombinationProfile:
    """Synthetic RecombinationProfile with Gaussian G_singlet centred in EML."""
    z_nm   = mesh.z_nm
    z_mid  = 0.5 * (z_nm[0] + z_nm[-1])
    sigma  = 5.0
    R_lan  = np.exp(-0.5 * ((z_nm - z_mid) / sigma) ** 2) * 1e24   # [m⁻³/s]
    R_srh  = np.zeros_like(R_lan)
    return RecombinationProfile(
        voltage_V=3.0,
        z_nm=z_nm,
        R_srh_m3s=R_srh,
        R_lan_m3s=R_lan,
        R_total_m3s=R_lan,
    )


@pytest.fixture(scope="module")
def rec_profile_zero(mesh) -> RecombinationProfile:
    """RecombinationProfile with all-zero recombination (dark conditions)."""
    z_nm = mesh.z_nm
    zeros = np.zeros_like(z_nm)
    return RecombinationProfile(
        voltage_V=0.0,
        z_nm=z_nm,
        R_srh_m3s=zeros,
        R_lan_m3s=zeros,
        R_total_m3s=zeros,
    )


# ---------------------------------------------------------------------------
# TestGridInterpolation
# ---------------------------------------------------------------------------

class TestGridInterpolation:

    def test_interpolate_same_grid_is_identity(self):
        z = np.arange(0.0, 10.0)
        G = np.random.default_rng(0).random(len(z))
        G_out = interpolate_to_optical_grid(z, G, z)
        np.testing.assert_allclose(G_out, G, atol=1e-12)

    def test_interpolate_zero_outside_range(self):
        z_elec = np.array([5.0, 6.0, 7.0])
        G_elec = np.ones(3)
        z_opt  = np.array([0.0, 1.0, 5.5, 9.0, 10.0])
        G_out  = interpolate_to_optical_grid(z_elec, G_elec, z_opt)
        assert G_out[0] == pytest.approx(0.0)
        assert G_out[1] == pytest.approx(0.0)
        assert G_out[2] == pytest.approx(1.0)
        assert G_out[4] == pytest.approx(0.0)

    def test_interpolate_shape_matches_opt_grid(self):
        z_elec = np.linspace(0, 100, 50)
        G_elec = np.ones(50)
        z_opt  = np.linspace(0, 100, 101)
        assert interpolate_to_optical_grid(z_elec, G_elec, z_opt).shape == (101,)

    def test_eml_mask_includes_interior_points(self, field_profile):
        boundaries = field_profile.layer_boundaries_nm
        names      = field_profile.layer_names
        z_opt      = field_profile.z_nm
        mask       = eml_mask(z_opt, boundaries, names, "EML")
        z_start, z_end = eml_layer_bounds(boundaries, names, "EML")
        assert np.all(z_opt[mask] >= z_start)
        assert np.all(z_opt[mask] <= z_end)

    def test_eml_mask_raises_unknown_layer(self, field_profile):
        with pytest.raises(ValueError):
            eml_mask(
                field_profile.z_nm,
                field_profile.layer_boundaries_nm,
                field_profile.layer_names,
                "NONEXISTENT_LAYER",
            )


# ---------------------------------------------------------------------------
# TestGenerationNormalization
# ---------------------------------------------------------------------------

class TestGenerationNormalization:

    def test_unit_integral(self):
        z = np.linspace(0.0, 20.0, 21)
        G = np.exp(-0.5 * ((z - 10.0) / 3.0) ** 2)
        G_norm = normalize_generation(G, z)
        integral = float(np.trapezoid(G_norm, z))
        assert integral == pytest.approx(1.0, rel=1e-6)

    def test_uniform_when_all_zero(self):
        z = np.linspace(0.0, 10.0, 11)
        G = np.zeros(11)
        G_norm = normalize_generation(G, z)
        thickness = z[-1] - z[0]
        np.testing.assert_allclose(G_norm, 1.0 / thickness, rtol=1e-10)

    def test_shape_preserved(self):
        z = np.linspace(0.0, 5.0, 6)
        G = np.ones(6)
        assert normalize_generation(G, z).shape == (6,)

    def test_single_point_eml(self):
        G_norm = normalize_generation(np.array([1e22]), np.array([50.0]))
        assert G_norm.shape == (1,)
        assert G_norm[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestWeightedEmissionIntegral
# ---------------------------------------------------------------------------

class TestWeightedEmissionIntegral:

    def test_output_shape(self):
        N_eml, N_wl = 10, 20
        G_norm = np.ones(N_eml) / 9.0   # not truly normalised, just shape test
        z_eml  = np.linspace(0, 9, N_eml)
        E2     = np.ones((N_eml, N_wl))
        result = weighted_emission_integral(G_norm, z_eml, E2)
        assert result.shape == (N_wl,)

    def test_uniform_G_equals_mean_E_squared_times_thickness(self):
        """For constant G_norm = 1/L, integral = (1/L)×∫|E|²dz ≈ mean(|E|²)."""
        z_eml  = np.linspace(0.0, 10.0, 11)   # L = 10 nm
        G_norm = np.full(11, 1.0 / 10.0)
        E2     = np.random.default_rng(42).random((11, 5))
        result = weighted_emission_integral(G_norm, z_eml, E2)
        # Expected: trapezoid(E2, z) / 10
        for j in range(5):
            expected = float(np.trapezoid(E2[:, j], z_eml)) / 10.0
            assert result[j] == pytest.approx(expected, rel=1e-6)

    def test_all_zero_G_produces_zero(self):
        z_eml = np.linspace(0, 5, 6)
        G_norm = np.zeros(6)
        E2     = np.ones((6, 4))
        result = weighted_emission_integral(G_norm, z_eml, E2)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_single_point_eml_returns_product(self):
        """Single-point EML: result = G_norm[0] × E2[0] × 1 nm."""
        G_norm = np.array([2.0])
        z_eml  = np.array([30.0])
        E2     = np.array([[3.0, 4.0]])
        result = weighted_emission_integral(G_norm, z_eml, E2)
        np.testing.assert_allclose(result, [6.0, 8.0])


# ---------------------------------------------------------------------------
# TestApplyPLAndNormalize
# ---------------------------------------------------------------------------

class TestApplyPLAndNormalize:

    def test_emission_sums_to_one(self):
        raw = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        pl  = np.array([0.5, 1.0, 1.0, 1.0, 0.5])
        em  = apply_pl_and_normalize(raw, pl)
        assert em.sum() == pytest.approx(1.0, rel=1e-10)

    def test_zero_pl_returns_zero(self):
        raw = np.array([1.0, 2.0, 3.0])
        pl  = np.zeros(3)
        em  = apply_pl_and_normalize(raw, pl)
        np.testing.assert_allclose(em, 0.0, atol=1e-20)

    def test_shape_preserved(self):
        raw = np.ones(10)
        pl  = np.ones(10)
        assert apply_pl_and_normalize(raw, pl).shape == (10,)


# ---------------------------------------------------------------------------
# TestCIEPhotopic
# ---------------------------------------------------------------------------

class TestCIEPhotopic:

    def test_peak_at_555nm(self):
        wl = np.array([400.0, 555.0, 700.0])
        V  = cie_photopic_response(wl)
        assert V[1] == pytest.approx(1.0, rel=1e-10)
        assert V[1] > V[0]
        assert V[1] > V[2]

    def test_non_negative(self):
        wl = np.arange(380.0, 781.0, 1.0)
        assert np.all(cie_photopic_response(wl) >= 0.0)


# ---------------------------------------------------------------------------
# TestLuminanceFormula
# ---------------------------------------------------------------------------

class TestLuminanceFormula:

    _WL = np.arange(450.0, 700.0, 10.0)
    _EM = np.ones(len(_WL)) / len(_WL)   # flat normalised spectrum

    def test_luminance_zero_at_dark(self):
        assert compute_luminance(self._WL, self._EM, 0.0, 1.0, 1.0) == 0.0

    def test_luminance_positive_at_forward_bias(self):
        L = compute_luminance(self._WL, self._EM, 100.0, 1.0, 1.0)
        assert L > 0.0

    def test_cd_per_A_unit_conversion(self):
        J = 50.0   # A/m²
        L = 1000.0  # cd/m²
        assert compute_cd_per_A(L, J) == pytest.approx(L / (J * 1e-4), rel=1e-10)

    def test_luminance_linear_in_J(self):
        L1 = compute_luminance(self._WL, self._EM, 100.0, 1.0, 1.0)
        L2 = compute_luminance(self._WL, self._EM, 200.0, 1.0, 1.0)
        assert L2 == pytest.approx(2.0 * L1, rel=1e-9)


# ---------------------------------------------------------------------------
# TestCoupledEmissionSolver
# ---------------------------------------------------------------------------

class TestCoupledEmissionSolver:

    def test_solve_returns_correct_type(self, coupled_solver, rec_profile_gaussian):
        result = coupled_solver.solve(rec_profile_gaussian, J_Am2=100.0)
        assert isinstance(result, ElectroOpticalResult)

    def test_emission_spectrum_sums_to_one(self, coupled_solver, rec_profile_gaussian):
        result = coupled_solver.solve(rec_profile_gaussian, J_Am2=100.0)
        assert result.emission_spectrum.sum() == pytest.approx(1.0, rel=1e-6)

    def test_luminance_positive_at_nonzero_J(self, coupled_solver, rec_profile_gaussian):
        result = coupled_solver.solve(rec_profile_gaussian, J_Am2=100.0)
        assert result.luminance_cd_m2 > 0.0

    def test_zero_recombination_gives_zero_luminance(self, coupled_solver, rec_profile_zero):
        result = coupled_solver.solve(rec_profile_zero, J_Am2=0.0)
        assert result.luminance_cd_m2 == pytest.approx(0.0)

    def test_eml_z_within_layer_bounds(self, coupled_solver, field_profile):
        z_start, z_end = eml_layer_bounds(
            field_profile.layer_boundaries_nm,
            field_profile.layer_names,
            "EML",
        )
        assert np.all(coupled_solver._eml_z_nm >= z_start - 1e-9)
        assert np.all(coupled_solver._eml_z_nm <= z_end + 1e-9)


# ---------------------------------------------------------------------------
# TestSolveSweep
# ---------------------------------------------------------------------------

class TestSolveSweep:

    def test_sweep_result_count_matches_profiles(
        self, coupled_solver, rec_profiles, sweep_result
    ):
        csweep = coupled_solver.solve_sweep(rec_profiles, sweep_result)
        assert isinstance(csweep, CoupledSweepResult)
        assert len(csweep.results) == len(rec_profiles)

    def test_voltages_non_decreasing(
        self, coupled_solver, rec_profiles, sweep_result
    ):
        csweep = coupled_solver.solve_sweep(rec_profiles, sweep_result)
        v = csweep.voltages
        assert np.all(np.diff(v) >= -1e-9)

    def test_mismatch_raises_value_error(
        self, coupled_solver, rec_profiles, sweep_result
    ):
        truncated = rec_profiles[:1]
        with pytest.raises(ValueError):
            coupled_solver.solve_sweep(truncated, sweep_result)
