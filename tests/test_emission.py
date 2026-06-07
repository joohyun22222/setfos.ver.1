"""Tests for Step 6 — optical emission weighting and outcoupling interface."""

import json

import numpy as np
import pytest
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
NK_ROOT = ROOT / "data" / "nk"
PL_ROOT = ROOT / "data" / "pl"

from src.io import load_emitter_config, load_project
from src.io.models import EmitterConfig
from src.emission import (
    EmissionResult,
    EmissionSolver,
    EmitterProfile,
    FarFieldOutcoupling,
    NullOutcoupling,
    OutcouplingBase,
    OutcouplingResult,
    StackContext,
)
from src.optics.field_profile import compute_field_profile


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

WL = np.arange(380.0, 785.0, 5.0)


@pytest.fixture(scope="module")
def emitter_cfg():
    return load_emitter_config(SAMPLES / "emitter_irppy3.yaml")


@pytest.fixture(scope="module")
def oled_project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def emission_result(oled_project, emitter_cfg):
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
    return solver.solve(
        oled_project.device_stack,
        oled_project.material_db,
        oled_project.solver_config,
        emitter_cfg,
    )


@pytest.fixture(scope="module")
def emission_result_farfield(oled_project, emitter_cfg):
    solver = EmissionSolver(
        nk_root=NK_ROOT,
        project_root=ROOT,
        outcoupling=FarFieldOutcoupling(),
    )
    return solver.solve(
        oled_project.device_stack,
        oled_project.material_db,
        oled_project.solver_config,
        emitter_cfg,
    )


# ---------------------------------------------------------------------------
# TestEmitterConfig — YAML loading and validation
# ---------------------------------------------------------------------------

class TestEmitterConfig:

    def test_load_returns_emitter_config(self, emitter_cfg):
        assert isinstance(emitter_cfg, EmitterConfig)

    def test_layer_name(self, emitter_cfg):
        assert emitter_cfg.layer_name == "EML"

    def test_horizontal_fraction_value(self, emitter_cfg):
        assert emitter_cfg.horizontal_fraction == pytest.approx(0.78)

    def test_z_distribution(self, emitter_cfg):
        assert emitter_cfg.z_distribution == "center"

    def test_emitter_type(self, emitter_cfg):
        assert emitter_cfg.emitter_type == "phosphorescent"

    def test_invalid_horizontal_fraction_raises(self, tmp_path):
        bad = tmp_path / "bad_emitter.yaml"
        bad.write_text(
            "layer_name: EML\npl_spectrum_file: x.csv\nhorizontal_fraction: 1.5\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="horizontal_fraction"):
            load_emitter_config(bad)

    def test_invalid_z_distribution_raises(self, tmp_path):
        bad = tmp_path / "bad_z.yaml"
        bad.write_text(
            "layer_name: EML\npl_spectrum_file: x.csv\nz_distribution: diagonal\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="z_distribution"):
            load_emitter_config(bad)

    def test_invalid_emitter_type_raises(self, tmp_path):
        bad = tmp_path / "bad_type.yaml"
        bad.write_text(
            "layer_name: EML\npl_spectrum_file: x.csv\nemitter_type: triplet\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="emitter_type"):
            load_emitter_config(bad)


# ---------------------------------------------------------------------------
# TestPLSpectrum — Ir(ppy)3 PL data file
# ---------------------------------------------------------------------------

class TestPLSpectrum:

    def test_pl_csv_has_81_rows(self):
        import csv
        with open(PL_ROOT / "Irppy3_pl.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 81

    def test_pl_spectrum_max_is_one(self):
        import csv
        with open(PL_ROOT / "Irppy3_pl.csv", "r") as f:
            intensities = [float(r["intensity"]) for r in csv.DictReader(f)]
        assert max(intensities) == pytest.approx(1.0, abs=1e-6)

    def test_pl_spectrum_non_negative(self):
        import csv
        with open(PL_ROOT / "Irppy3_pl.csv", "r") as f:
            intensities = [float(r["intensity"]) for r in csv.DictReader(f)]
        assert all(v >= 0 for v in intensities)

    def test_pl_peak_in_green_band(self):
        """Ir(ppy)3 PL should peak between 500 and 560 nm."""
        import csv
        wls, ints = [], []
        with open(PL_ROOT / "Irppy3_pl.csv", "r") as f:
            for r in csv.DictReader(f):
                wls.append(float(r["wavelength_nm"]))
                ints.append(float(r["intensity"]))
        peak_wl = wls[ints.index(max(ints))]
        assert 500 <= peak_wl <= 560, f"PL peak at {peak_wl} nm, expected 500–560 nm"


# ---------------------------------------------------------------------------
# TestOutcouplingInterface — ABC and concrete models
# ---------------------------------------------------------------------------

class TestOutcouplingInterface:

    def test_outcoupling_base_is_abstract(self):
        """Cannot instantiate OutcouplingBase directly."""
        with pytest.raises(TypeError):
            OutcouplingBase()

    def test_null_outcoupling_eta_is_one(self):
        ctx = _dummy_context(WL)
        oc = NullOutcoupling()
        result = oc.compute(ctx)
        assert isinstance(result, OutcouplingResult)
        assert np.allclose(result.eta_spectrum, 1.0)

    def test_null_outcoupling_name(self):
        assert NullOutcoupling().name == "null"

    def test_farfield_outcoupling_eta_value(self):
        """η_out = 1/(2·n²) for n_sub=1.5 → 1/4.5 ≈ 0.2222."""
        ctx = _dummy_context(WL, n_sub=1.5)
        oc = FarFieldOutcoupling()
        result = oc.compute(ctx)
        expected = 1.0 / (2.0 * 1.5**2)
        assert np.allclose(result.eta_spectrum, expected, rtol=1e-6)

    def test_farfield_outcoupling_name(self):
        assert FarFieldOutcoupling().name == "far_field_geometric"

    def test_outcoupling_result_eta_mean(self):
        ctx = _dummy_context(WL)
        result = FarFieldOutcoupling().compute(ctx)
        expected_mean = float(np.mean(result.eta_spectrum))
        assert result.eta_mean == pytest.approx(expected_mean)

    def test_custom_outcoupling_plugin(self):
        """Plugging in a custom OutcouplingBase subclass must work."""
        class HalfOutcoupling(OutcouplingBase):
            @property
            def name(self):
                return "half"
            def compute(self, ctx):
                return OutcouplingResult(ctx.wavelengths_nm, 0.5*np.ones_like(ctx.wavelengths_nm), "half")

        ctx = _dummy_context(WL)
        result = HalfOutcoupling().compute(ctx)
        assert np.allclose(result.eta_spectrum, 0.5)


# ---------------------------------------------------------------------------
# TestEmissionSolver — end-to-end on OLED sample
# ---------------------------------------------------------------------------

class TestEmissionSolver:

    def test_returns_emission_result(self, emission_result):
        assert isinstance(emission_result, EmissionResult)

    def test_z_emitter_within_eml_bounds(self, emission_result):
        """EML spans z = 190–220 nm in the OLED sample stack."""
        assert 190.0 <= emission_result.z_emitter_nm <= 220.0

    def test_E_squared_non_negative(self, emission_result):
        assert np.all(emission_result.E_squared >= 0)

    def test_E_squared_non_zero_at_emitter(self, emission_result):
        """Light must reach the EML — |E|² cannot be zero."""
        assert np.mean(emission_result.E_squared) > 0.01

    def test_pl_spectrum_non_negative(self, emission_result):
        assert np.all(emission_result.pl_spectrum >= 0)

    def test_weighted_spectrum_sums_to_one(self, emission_result):
        """weighted_spectrum is area-normalised."""
        total = emission_result.weighted_spectrum.sum()
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_emission_spectrum_non_negative(self, emission_result):
        assert np.all(emission_result.emission_spectrum >= 0)

    def test_emission_spectrum_sums_to_one(self, emission_result):
        total = emission_result.emission_spectrum.sum()
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_peak_emission_in_green_band(self, emission_result):
        """PL-weighted peak should stay in 480–560 nm (green OLED with Ir(ppy)3)."""
        peak = emission_result.peak_emission_nm
        assert 480 <= peak <= 570, f"Peak emission at {peak:.0f} nm"

    def test_null_outcoupling_attached_by_default(self, emission_result):
        """Default EmissionSolver uses NullOutcoupling."""
        oc = emission_result.outcoupling
        assert oc is not None
        assert oc.model_name == "null"

    def test_null_outcoupling_eta_is_one(self, emission_result):
        assert np.allclose(emission_result.outcoupling.eta_spectrum, 1.0)

    def test_eta_out_none_without_outcoupling(self):
        """EmissionResult.eta_out returns None when outcoupling is None."""
        result = EmissionResult(
            wavelength_nm=WL,
            z_emitter_nm=205.0,
            layer_name="EML",
            pl_spectrum=np.ones_like(WL),
            E_squared=np.ones_like(WL),
            weighted_spectrum=np.ones_like(WL) / len(WL),
            outcoupling=None,
        )
        assert result.eta_out is None


# ---------------------------------------------------------------------------
# TestEmissionWithFarField — outcoupling model swap
# ---------------------------------------------------------------------------

class TestEmissionWithFarField:

    def test_returns_emission_result(self, emission_result_farfield):
        assert isinstance(emission_result_farfield, EmissionResult)

    def test_farfield_eta_out_approx(self, emission_result_farfield):
        """PL-weighted η_out ≈ 1/(2·1.5²) ≈ 0.222 for far-field model."""
        eta = emission_result_farfield.eta_out
        assert eta is not None
        assert eta == pytest.approx(1.0 / (2.0 * 1.5**2), rel=1e-3)

    def test_emission_spectrum_normalised(self, emission_result_farfield):
        total = emission_result_farfield.emission_spectrum.sum()
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestEmissionResultIO — CSV, JSON, summary
# ---------------------------------------------------------------------------

class TestEmissionResultIO:

    def test_to_csv_creates_file(self, emission_result, tmp_path):
        out = tmp_path / "emission.csv"
        emission_result.to_csv(out)
        assert out.exists()

    def test_to_csv_header(self, emission_result, tmp_path):
        out = tmp_path / "emission.csv"
        emission_result.to_csv(out)
        header = out.read_text().splitlines()[0]
        assert "wavelength_nm" in header
        assert "emission_spectrum" in header

    def test_to_csv_row_count(self, emission_result, tmp_path):
        out = tmp_path / "emission.csv"
        emission_result.to_csv(out)
        lines = out.read_text().splitlines()
        assert len(lines) == len(WL) + 1  # header + 81 data rows

    def test_to_json_creates_file(self, emission_result, tmp_path):
        out = tmp_path / "emission.json"
        emission_result.to_json(out)
        assert out.exists()

    def test_to_json_has_required_keys(self, emission_result, tmp_path):
        out = tmp_path / "emission.json"
        emission_result.to_json(out)
        data = json.loads(out.read_text())
        for key in ["wavelength_nm", "pl_spectrum", "E_squared",
                    "weighted_spectrum", "emission_spectrum"]:
            assert key in data, f"Missing key: {key}"

    def test_to_json_has_outcoupling_section(self, emission_result_farfield, tmp_path):
        out = tmp_path / "emission_ff.json"
        emission_result_farfield.to_json(out)
        data = json.loads(out.read_text())
        assert "outcoupling" in data
        assert "eta_spectrum" in data["outcoupling"]

    def test_summary_contains_emitter_info(self, emission_result):
        s = emission_result.summary()
        assert "EmissionResult" in s
        assert "EML" in s


# ---------------------------------------------------------------------------
# TestZDistributions — different emitter position modes
# ---------------------------------------------------------------------------

class TestZDistributions:

    @pytest.fixture(scope="class")
    def project(self):
        return load_project(SAMPLES / "oled_input.yaml")

    def _solve_with_zdist(self, project, z_dist):
        cfg = EmitterConfig(
            layer_name="EML",
            material="Ir(ppy)3",
            pl_spectrum_file=str(ROOT / "data" / "pl" / "Irppy3_pl.csv"),
            horizontal_fraction=0.78,
            z_distribution=z_dist,
            emitter_type="phosphorescent",
        )
        solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
        return solver.solve(
            project.device_stack,
            project.material_db,
            project.solver_config,
            cfg,
        )

    def test_z_front_lt_center(self, project):
        """front emitter z < center emitter z."""
        z_front  = self._solve_with_zdist(project, "front").z_emitter_nm
        z_center = self._solve_with_zdist(project, "center").z_emitter_nm
        assert z_front < z_center

    def test_z_back_gt_center(self, project):
        """back emitter z > center emitter z."""
        z_back   = self._solve_with_zdist(project, "back").z_emitter_nm
        z_center = self._solve_with_zdist(project, "center").z_emitter_nm
        assert z_back > z_center

    def test_uniform_returns_emission_result(self, project):
        result = self._solve_with_zdist(project, "uniform")
        assert isinstance(result, EmissionResult)
        assert result.weighted_spectrum.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _dummy_context(wavelengths, n_sub=1.5):
    """Minimal StackContext for unit-testing outcoupling models."""
    n_j = np.full(len(wavelengths), 1.5 + 0j)
    from src.optics.field_profile import compute_field_profile
    fp = compute_field_profile([n_j], [100.0], wavelengths)
    return StackContext(
        n_layers=[n_j],
        d_layers_nm=[100.0],
        wavelengths_nm=wavelengths,
        z_emitter_nm=50.0,
        horizontal_fraction=1.0,
        n_inc=1.0,
        n_sub=n_sub,
        field_profile=fp,
    )
