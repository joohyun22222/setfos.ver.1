"""Tests for layer-resolved absorption and internal E-field profile (Step 5)."""

import json

import numpy as np
import pytest
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
NK_ROOT = ROOT / "data" / "nk"

from src.optics.field_profile import (
    LayerAbsorption,
    FieldProfileResult,
    compute_field_profile,
)
from src.optics.tmm import compute_rta
from src.optics.rta import RTASolver, RTAResult
from src.io import load_project


# ---------------------------------------------------------------------------
# Helpers / shared setup
# ---------------------------------------------------------------------------

WL = np.arange(380.0, 785.0, 5.0)   # 81 wavelengths


def _const_n(n_val: complex, size: int = len(WL)) -> np.ndarray:
    return np.full(size, n_val, dtype=complex)


@pytest.fixture(scope="module")
def oled_project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def oled_rta_and_profile(oled_project):
    solver = RTASolver(nk_root=NK_ROOT)
    return solver.solve_with_profile(
        oled_project.device_stack,
        oled_project.material_db,
        oled_project.solver_config,
        z_resolution_nm=1.0,
    )


@pytest.fixture(scope="module")
def oled_rta(oled_rta_and_profile):
    return oled_rta_and_profile[0]


@pytest.fixture(scope="module")
def oled_fp(oled_rta_and_profile):
    return oled_rta_and_profile[1]


# ---------------------------------------------------------------------------
# TestLayerAbsorptionCore — analytical cases
# ---------------------------------------------------------------------------

class TestLayerAbsorptionCore:

    def test_single_lossless_layer_zero_absorption(self):
        """A dielectric layer with k=0 must absorb nothing."""
        n_j = _const_n(1.5 + 0j)
        fp = compute_field_profile([n_j], [100.0], WL)
        assert len(fp.layer_absorption) == 1
        A_j = fp.layer_absorption[0].A_spectrum
        assert np.all(A_j >= 0)
        assert np.allclose(A_j, 0.0, atol=1e-10), f"Lossless layer: max A = {A_j.max():.2e}"

    def test_single_absorbing_layer_positive_absorption(self):
        """An absorbing layer must have A > 0."""
        n_j = _const_n(1.5 + 0.1j)
        fp = compute_field_profile([n_j], [100.0], WL)
        A_j = fp.layer_absorption[0].A_spectrum
        assert np.all(A_j > 0)

    def test_metal_layer_high_absorption(self):
        """Thick Al: should absorb a significant fraction of the light that enters."""
        n_al = _const_n(0.77 + 6.08j)
        fp = compute_field_profile([n_al], [100.0], WL)
        A_al = fp.layer_absorption[0].A_spectrum
        assert np.mean(A_al) > 0.05, f"Al mean A = {np.mean(A_al):.4f}"

    def test_layer_absorption_sums_to_total_A(self):
        """Sum of per-layer absorptances must equal total A from compute_rta."""
        n_org   = _const_n(1.75 + 0.02j)
        n_metal = _const_n(0.77 + 6.08j)
        n_layers = [n_org, n_metal]
        d_layers = [50.0, 100.0]

        _R, _T, A_total = compute_rta(n_layers, d_layers, WL)
        fp = compute_field_profile(n_layers, d_layers, WL)

        A_layer_sum = fp.total_layer_absorption
        err = np.abs(A_layer_sum - A_total)
        assert np.max(err) < 1e-6, f"Layer A sum differs from total A; max err = {err.max():.2e}"

    def test_layer_absorption_non_negative(self):
        """All per-layer A_j must be >= 0."""
        layers = [_const_n(1.8 + 0.05j), _const_n(0.77 + 6.0j), _const_n(1.5 + 0.01j)]
        fp = compute_field_profile(layers, [60.0, 100.0, 40.0], WL)
        for la in fp.layer_absorption:
            assert np.all(la.A_spectrum >= 0), f"Negative A in {la.layer_name}"

    def test_layer_absorption_leq_one(self):
        """No layer can absorb more than 100% of incident light."""
        layers = [_const_n(0.5 + 10.0j)]
        fp = compute_field_profile(layers, [200.0], WL)
        assert np.all(fp.layer_absorption[0].A_spectrum <= 1.0 + 1e-10)

    def test_layer_count_matches_input(self):
        """FieldProfileResult must have one LayerAbsorption per input layer."""
        layers = [_const_n(1.5), _const_n(2.0), _const_n(1.8 + 0.05j)]
        fp = compute_field_profile(layers, [40.0, 80.0, 30.0], WL)
        assert len(fp.layer_absorption) == 3

    def test_layer_names_propagated(self):
        """Custom layer names must appear in LayerAbsorption entries."""
        names = ["ITO", "TCTA", "Al"]
        layers = [_const_n(1.9), _const_n(1.73), _const_n(0.77 + 6.08j)]
        fp = compute_field_profile(layers, [150.0, 40.0, 100.0], WL, layer_names=names)
        assert [la.layer_name for la in fp.layer_absorption] == names

    def test_mismatched_inputs_raise(self):
        """Mismatched n/d list lengths must raise ValueError."""
        with pytest.raises(ValueError):
            compute_field_profile(
                [_const_n(1.5), _const_n(2.0)], [10.0], WL
            )

    def test_energy_conservation_multilayer(self):
        """R + sum(A_j) + T = 1 at every wavelength for a complex stack."""
        n_layers = [
            _const_n(1.9 + 0.02j),
            _const_n(1.73 + 0.005j),
            _const_n(0.77 + 6.08j),
        ]
        d_layers = [150.0, 40.0, 100.0]
        R, T, _A = compute_rta(n_layers, d_layers, WL)
        fp = compute_field_profile(n_layers, d_layers, WL)

        balance = R + fp.total_layer_absorption + T
        err = np.abs(balance - 1.0)
        assert np.max(err) < 1e-6, f"Energy not conserved; max err = {err.max():.2e}"


# ---------------------------------------------------------------------------
# TestFieldProfileShape — geometric and shape checks
# ---------------------------------------------------------------------------

class TestFieldProfileShape:

    def test_E_field_shape(self):
        """E_field must be (N_z, N_wl)."""
        n = [_const_n(1.5)]
        fp = compute_field_profile(n, [100.0], WL, z_resolution_nm=5.0)
        assert fp.E_field.ndim == 2
        assert fp.E_field.shape[1] == len(WL)

    def test_E_squared_non_negative(self):
        """|E|² must be non-negative everywhere."""
        n = [_const_n(1.5 + 0.1j)]
        fp = compute_field_profile(n, [100.0], WL)
        assert np.all(fp.E_squared >= 0)

    def test_z_array_length_matches_E_field_rows(self):
        n = [_const_n(1.8), _const_n(1.5 + 0.05j)]
        fp = compute_field_profile(n, [50.0, 80.0], WL, z_resolution_nm=2.0)
        assert len(fp.z_nm) == fp.E_field.shape[0]

    def test_layer_boundaries_cumulative_sum(self):
        """layer_boundaries_nm[j] = sum of d_layers[0..j-1]."""
        d = [150.0, 40.0, 30.0, 50.0, 100.0]
        n_layers = [_const_n(1.5)] * len(d)
        fp = compute_field_profile(n_layers, d, WL)

        expected = np.concatenate(([0.0], np.cumsum(d)))
        assert np.allclose(fp.layer_boundaries_nm, expected), (
            f"boundaries: {fp.layer_boundaries_nm}, expected: {expected}"
        )

    def test_z_starts_at_zero(self):
        n = [_const_n(1.5)]
        fp = compute_field_profile(n, [50.0], WL)
        assert fp.z_nm[0] == pytest.approx(0.0)

    def test_z_ends_at_total_thickness(self):
        d = [150.0, 40.0, 100.0]
        n_layers = [_const_n(1.5)] * len(d)
        fp = compute_field_profile(n_layers, d, WL)
        assert fp.z_nm[-1] == pytest.approx(sum(d))

    def test_wavelength_array_preserved(self):
        n = [_const_n(1.5)]
        fp = compute_field_profile(n, [100.0], WL)
        assert np.array_equal(fp.wavelength_nm, WL)

    def test_get_field_at_wavelength_shape(self):
        n = [_const_n(1.5)]
        fp = compute_field_profile(n, [100.0], WL)
        profile = fp.get_field_at_wavelength(500.0)
        assert profile.shape == (len(fp.z_nm),)
        assert profile.dtype == complex

    def test_get_intensity_at_z_shape(self):
        n = [_const_n(1.5 + 0.05j)]
        fp = compute_field_profile(n, [100.0], WL)
        spectrum = fp.get_intensity_at_z(50.0)
        assert spectrum.shape == (len(WL),)
        assert np.all(spectrum >= 0)

    def test_layer_center_z(self):
        d = [150.0, 40.0, 100.0]
        n_layers = [_const_n(1.5)] * len(d)
        fp = compute_field_profile(n_layers, d, WL)
        # centre of layer 1 (TCTA-equivalent, d=40nm starting at z=150)
        assert fp.layer_center_z(1) == pytest.approx(150.0 + 20.0)


# ---------------------------------------------------------------------------
# TestFieldProfileOLED — end-to-end on sample OLED stack
# ---------------------------------------------------------------------------

class TestFieldProfileOLED:

    def test_solve_with_profile_returns_tuple(self, oled_rta, oled_fp):
        assert isinstance(oled_rta, RTAResult)
        assert isinstance(oled_fp, FieldProfileResult)

    def test_oled_layer_count(self, oled_fp):
        """OLED sample has 5 layers."""
        assert len(oled_fp.layer_absorption) == 5

    def test_oled_layer_absorption_non_negative(self, oled_fp):
        for la in oled_fp.layer_absorption:
            assert np.all(la.A_spectrum >= -1e-12), f"Negative A in {la.layer_name}"

    def test_oled_energy_conservation(self, oled_rta, oled_fp):
        """R + sum(A_j) + T = 1 within 1e-5 for the OLED stack."""
        balance = oled_rta.R + oled_fp.total_layer_absorption + oled_rta.T
        err = np.abs(balance - 1.0)
        assert np.max(err) < 1e-5, f"Energy balance error = {err.max():.2e}"

    def test_al_has_highest_mean_absorption(self, oled_fp):
        """Al cathode should dominate layer-resolved absorption."""
        al_la = next(la for la in oled_fp.layer_absorption if "Al" in la.material_name)
        al_A = al_la.A_mean
        for la in oled_fp.layer_absorption:
            if la.material_name != al_la.material_name:
                assert al_A >= la.A_mean, (
                    f"Al A_mean={al_A:.4f} < {la.material_name} A_mean={la.A_mean:.4f}"
                )

    def test_eml_field_non_zero(self, oled_fp):
        """The EML layer must have non-zero |E|² — a necessary condition for emission."""
        eml_idx = None
        for i, la in enumerate(oled_fp.layer_absorption):
            if la.layer_name.lower().startswith("eml") or "eml" in la.layer_name.lower():
                eml_idx = i
                break
        if eml_idx is None:
            pytest.skip("EML layer not identified by name in this stack")
        z_eml = oled_fp.layer_center_z(eml_idx)
        intensity = oled_fp.get_intensity_at_z(z_eml)
        assert np.mean(intensity) > 0.01, f"EML mean |E|² = {np.mean(intensity):.4f}"

    def test_oled_z_resolution_coverage(self, oled_fp):
        """z grid should span the full device thickness."""
        total_d = sum(la.thickness_nm for la in oled_fp.layer_absorption)
        assert oled_fp.z_nm[-1] == pytest.approx(total_d)

    def test_oled_profile_csv(self, oled_fp, tmp_path):
        out = tmp_path / "field_profile.csv"
        oled_fp.to_csv(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        # header + one row per z point
        assert len(lines) == len(oled_fp.z_nm) + 1
        assert lines[0].startswith("z_nm,")

    def test_oled_layer_absorption_csv(self, oled_fp, tmp_path):
        out = tmp_path / "layer_absorption.csv"
        oled_fp.layer_absorption_to_csv(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == len(WL) + 1
        assert lines[0].startswith("wavelength_nm,")

    def test_oled_profile_json(self, oled_fp, tmp_path):
        out = tmp_path / "field_profile.json"
        oled_fp.to_json(out)
        data = json.loads(out.read_text())
        assert "z_nm" in data
        assert "E_squared" in data
        assert "layer_absorption" in data
        assert len(data["layer_absorption"]) == 5

    def test_oled_summary_string(self, oled_fp):
        s = oled_fp.summary()
        assert "FieldProfile summary" in s
        assert "A_mean" in s

    def test_rta_consistent_with_profile(self, oled_rta, oled_fp):
        """RTAResult.A and sum of FieldProfileResult layer absorptances must agree."""
        A_from_rta    = oled_rta.A
        A_from_layers = oled_fp.total_layer_absorption
        err = np.abs(A_from_rta - A_from_layers)
        assert np.max(err) < 1e-5, (
            f"RTAResult.A vs layer sum: max discrepancy = {err.max():.2e}"
        )
