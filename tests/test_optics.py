"""Tests for the TMM optical solver and RTA calculation."""

import numpy as np
import pytest
from pathlib import Path

ROOT     = Path(__file__).parent.parent
SAMPLES  = ROOT / "configs" / "samples"
NK_ROOT  = ROOT / "data" / "nk"

from src.optics.tmm import compute_rta, compute_rta_single
from src.optics.nk_data import NKDataProvider, _read_nk_csv
from src.optics.rta import RTASolver, RTAResult
from src.io import load_project
from src.io.models import MaterialEntry, ThermalProperties


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

WL = np.arange(380.0, 785.0, 5.0)   # 81 wavelengths matching solver config


def _const_n(n_val: complex, size: int = len(WL)) -> np.ndarray:
    return np.full(size, n_val, dtype=complex)


@pytest.fixture(scope="module")
def project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def rta_result(project):
    solver = RTASolver(nk_root=NK_ROOT)
    return solver.solve(project.device_stack, project.material_db, project.solver_config)


# ---------------------------------------------------------------------------
# TMM core — analytical checks
# ---------------------------------------------------------------------------

class TestTMMCore:

    def test_empty_stack_energy_conservation(self):
        """No layers: result should equal Fresnel at air/glass interface."""
        R, T, A = compute_rta([], [], WL, n_inc=1.0, n_sub=1.5)
        assert np.allclose(R + T + A, 1.0, atol=1e-12)

    def test_empty_stack_no_absorption(self):
        """Air/glass with no absorbing layers → A = 0 exactly."""
        R, T, A = compute_rta([], [], WL, n_inc=1.0, n_sub=1.0)
        assert np.allclose(A, 0.0, atol=1e-12)

    def test_lossless_layer_no_absorption(self):
        """Lossless dielectric layer: k=0 → A should be ~0."""
        n = _const_n(1.5 + 0j)
        R, T, A = compute_rta([n], [100.0], WL, n_inc=1.0, n_sub=1.5)
        # Energy conservation
        assert np.allclose(R + T, 1.0, atol=1e-10), "R+T must equal 1 for lossless layer"
        # No absorption
        assert np.all(A <= 1e-10)

    def test_lossless_energy_conservation(self):
        """Three lossless layers: R + T + A = 1 everywhere."""
        layers_n = [_const_n(1.8), _const_n(1.5), _const_n(2.0)]
        layers_d = [50.0, 80.0, 30.0]
        R, T, A = compute_rta(layers_n, layers_d, WL)
        assert np.allclose(R + T + A, 1.0, atol=1e-10)

    def test_opaque_metal_near_zero_transmission(self):
        """Thick Al (100 nm): T should be essentially zero."""
        n_al = _const_n(0.77 + 6.08j)  # Al at ~500 nm
        R, T, A = compute_rta([n_al], [100.0], WL, n_inc=1.0, n_sub=1.0)
        assert np.all(T < 1e-4), f"Expected T≈0 for thick Al, got max T={T.max():.4f}"

    def test_opaque_metal_high_reflectance(self):
        """Al film: reflectance should be high (>0.8)."""
        n_al = _const_n(0.77 + 6.08j)
        R, T, A = compute_rta([n_al], [100.0], WL, n_inc=1.0, n_sub=1.0)
        assert np.all(R > 0.6), f"Al reflectance unexpectedly low: min={R.min():.3f}"

    def test_energy_conservation_absorbing_stack(self):
        """Absorbing multilayer: R + T + A = 1 at every wavelength."""
        n_org  = _const_n(1.75 + 0.01j)
        n_metal= _const_n(0.7  + 6.0j)
        R, T, A = compute_rta([n_org, n_metal], [50.0, 100.0], WL)
        err = np.abs(R + T + A - 1.0)
        assert np.max(err) < 1e-10, f"Energy not conserved; max error={err.max():.2e}"

    def test_scalar_wrapper_matches_vectorised(self):
        """compute_rta_single must return the same value as the vectorised form."""
        n_val = 1.8 + 0.05j
        d     = 50.0
        wl0   = 500.0

        R_s, T_s, A_s = compute_rta_single([n_val], [d], wl0)

        wl_arr = np.array([wl0])
        R_v, T_v, A_v = compute_rta([np.array([n_val])], [d], wl_arr)

        assert pytest.approx(R_s, abs=1e-12) == R_v[0]
        assert pytest.approx(T_s, abs=1e-12) == T_v[0]
        assert pytest.approx(A_s, abs=1e-12) == A_v[0]

    def test_output_shapes(self):
        """R, T, A must have same length as wavelength array."""
        n = [_const_n(1.5)]
        R, T, A = compute_rta(n, [50.0], WL)
        assert R.shape == WL.shape
        assert T.shape == WL.shape
        assert A.shape == WL.shape

    def test_mismatched_inputs_raise(self):
        """Mismatched n/d list lengths must raise ValueError."""
        with pytest.raises(ValueError):
            compute_rta([_const_n(1.5), _const_n(2.0)], [10.0], WL)

    def test_rta_values_in_unit_interval(self):
        """R, T, A must each be in [0, 1]."""
        n = [_const_n(1.5 + 0.1j)]
        R, T, A = compute_rta(n, [100.0], WL)
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))
        assert np.all((A >= 0) & (A <= 1))


# ---------------------------------------------------------------------------
# n/k data loading
# ---------------------------------------------------------------------------

class TestNKData:

    def test_al_csv_readable(self):
        wl, n, k = _read_nk_csv(NK_ROOT / "Al_nk.csv")
        assert len(wl) == 81
        assert np.all(n > 0)
        assert np.all(k > 0)

    def test_ito_csv_readable(self):
        wl, n, k = _read_nk_csv(NK_ROOT / "ITO_nk.csv")
        assert len(wl) == 81
        assert np.all(n > 1.0)  # ITO n > 1.5

    def test_organic_csv_nonnegative_k(self):
        for fname in ["TCTA_nk.csv", "TPBi_nk.csv", "Irppy3_nk.csv"]:
            wl, n, k = _read_nk_csv(NK_ROOT / fname)
            assert np.all(k >= 0), f"Negative k in {fname}"

    def test_provider_returns_correct_length(self):
        dummy_mat = MaterialEntry(
            material_name="Al",
            nk_reference="data/nk/Al_nk.csv",
            homo=None, lumo=None,
            electron_mobility=None, hole_mobility=None,
            dielectric_constant=None, electrode=None,
            thermal=ThermalProperties(None, None, None),
            notes="",
        )
        provider = NKDataProvider(NK_ROOT)
        result = provider.get_nk(dummy_mat, WL)
        assert result.shape == WL.shape
        assert result.dtype == complex

    def test_provider_fallback_on_missing(self, tmp_path):
        mat = MaterialEntry(
            material_name="FakeMat",
            nk_reference=None,
            homo=None, lumo=None,
            electron_mobility=None, hole_mobility=None,
            dielectric_constant=None, electrode=None,
            thermal=ThermalProperties(None, None, None),
            notes="",
        )
        provider = NKDataProvider(tmp_path)
        nk = provider.get_nk(mat, WL)
        assert np.allclose(np.real(nk), 1.5)
        assert np.allclose(np.imag(nk), 0.0)


# ---------------------------------------------------------------------------
# RTASolver — end-to-end on OLED sample
# ---------------------------------------------------------------------------

class TestRTASolverOLED:

    def test_returns_rta_result(self, rta_result):
        assert isinstance(rta_result, RTAResult)

    def test_wavelength_length(self, rta_result):
        assert len(rta_result.wavelength_nm) == 81

    def test_energy_conservation_oled(self, rta_result):
        err = rta_result.max_conservation_error
        assert err < 1e-8, f"Energy conservation violated: max error = {err:.2e}"

    def test_rta_arrays_in_unit_interval(self, rta_result):
        assert np.all(rta_result.R >= 0) and np.all(rta_result.R <= 1)
        assert np.all(rta_result.T >= 0) and np.all(rta_result.T <= 1)
        assert np.all(rta_result.A >= 0) and np.all(rta_result.A <= 1)

    def test_oled_near_zero_transmission(self, rta_result):
        """Al cathode should block nearly all transmission."""
        assert np.all(rta_result.T < 0.01), \
            f"Expected T<0.01 for OLED with Al cathode; got max T={rta_result.T.max():.4f}"

    def test_oled_high_absorptance(self, rta_result):
        """OLED stack absorbs non-trivially (Al dominates at UV; ITO adds visible absorption).
        Mean A ~0.20 is physically correct for this stack geometry."""
        assert np.mean(rta_result.A) > 0.1
        assert np.max(rta_result.A) > 0.3   # peak absorption should exceed 30 %

    def test_summary_string(self, rta_result):
        s = rta_result.summary()
        assert "RTA summary" in s
        assert "max|ΔE|" in s

    def test_to_csv(self, rta_result, tmp_path):
        out = tmp_path / "rta_out.csv"
        rta_result.to_csv(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert lines[0] == "wavelength_nm,R,T,A,RTA_sum"
        assert len(lines) == 82  # header + 81 data rows

    def test_to_json(self, rta_result, tmp_path):
        import json
        out = tmp_path / "rta_out.json"
        rta_result.to_json(out)
        data = json.loads(out.read_text())
        assert "wavelength_nm" in data
        assert len(data["R"]) == 81
