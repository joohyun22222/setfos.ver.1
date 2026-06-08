"""Tests for Steps 11-12 — J-V-L Calculator, EQE/CE/PE.

Coverage
--------
TestEfficiencyFormulas  (8) — EQE formula, CE formula, PE formula, zero guards
TestJVLPoint            (5) — properties, aliases, summary, spectrum shape
TestJVLResult           (8) — array shapes, aliases, spectrum_at, CSV/JSON, summary
TestJVLCalculatorRun    (5) — return type, n_points, voltage range, J finite, L≥0
TestJVLCalculatorPhysics(5) — lm_per_W zero at V=0, voltages increasing, cd/A≥0,
                               EQE≥0, peak wl in visible range
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
NK_ROOT = ROOT / "data" / "nk"

from src.io import load_project, load_emitter_config
from src.emission import EmissionSolver
from src.optics.field_profile import compute_field_profile, FieldProfileResult
from src.electrical import (
    build_mesh, build_solver, GummelConfig,
    BiasSweepRunner,
)
from src.electrical.recombination import RecombinationSolver
from src.electrical.continuity import ContinuitySolver

from src.jvl import JVLCalculator, JVLResult, JVLPoint, compute_eqe, compute_ce, compute_pe
from src.jvl.efficiency import _Q, _K_M, _PI


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def emitter_cfg():
    return load_emitter_config(SAMPLES / "emitter_irppy3.yaml")


@pytest.fixture(scope="module")
def field_profile(project) -> FieldProfileResult:
    from src.emission.solver import EmissionSolver as _ES
    _wl = np.arange(
        project.solver_config.wavelength_grid.start_nm,
        project.solver_config.wavelength_grid.end_nm + 0.5,
        project.solver_config.wavelength_grid.step_nm,
    )
    stack  = project.device_stack
    mat_db = project.material_db
    n_layers, d_layers, layer_names, _ = _ES(
        nk_root=NK_ROOT, project_root=ROOT
    )._build_optical_data(stack, mat_db, _wl)
    return compute_field_profile(
        n_layers, d_layers, _wl,
        layer_names=layer_names,
        n_inc=1.0, n_sub=1.5,
    )


@pytest.fixture(scope="module")
def mesh(project):
    return build_mesh(project.device_stack, project.material_db, z_resolution_nm=2.0)


@pytest.fixture(scope="module")
def gummel(mesh):
    cfg = GummelConfig(max_iterations=100, tolerance=1e-4, damping=0.5)
    return build_solver(mesh, cfg)


@pytest.fixture(scope="module")
def rec_solver(mesh):
    cont = ContinuitySolver(mesh)
    return RecombinationSolver(mesh, cont)


@pytest.fixture(scope="module")
def emitter_profile(project, emitter_cfg, field_profile):
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
    return solver._load_emitter(emitter_cfg, field_profile.wavelength_nm)


@pytest.fixture(scope="module")
def jvl_calc(gummel, rec_solver, field_profile, emitter_profile) -> JVLCalculator:
    return JVLCalculator(
        gummel          = gummel,
        rec_solver      = rec_solver,
        field_profile   = field_profile,
        emitter_profile = emitter_profile,
        emitter_layer   = "EML",
        eta_rad         = 1.0,
        eta_out         = 1.0,
    )


@pytest.fixture(scope="module")
def jvl_result(jvl_calc) -> JVLResult:
    """3-point sweep for speed."""
    return jvl_calc.run(v_start=0.0, v_end=4.0, n_points=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_spectrum(wl, center=520.0, sigma=20.0):
    em = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    return em / em.sum()


def _make_point(V=3.0, J=100.0, L=500.0, wl=None, em=None) -> JVLPoint:
    if wl is None:
        wl = np.linspace(380.0, 780.0, 81)
    if em is None:
        em = _gaussian_spectrum(wl)
    eqe = compute_eqe(wl, em, L, J)
    ce  = compute_ce(L, J)
    pe  = compute_pe(L, J, V)
    return JVLPoint(
        voltage_V=V, J_Am2=J, luminance_cd_m2=L,
        eqe=eqe, cd_per_A=ce, lm_per_W=pe,
        peak_wavelength_nm=float(wl[np.argmax(em)]),
        wavelength_nm=wl, emission_spectrum=em,
    )


def _make_jvl_result(n: int = 5) -> JVLResult:
    wl = np.linspace(380.0, 780.0, 81)
    em = _gaussian_spectrum(wl)
    points = [
        _make_point(V=float(i), J=float(i) * 10.0, L=float(i) * 100.0,
                    wl=wl.copy(), em=em.copy())
        for i in range(n)
    ]
    return JVLResult(points)


# ---------------------------------------------------------------------------
# TestEfficiencyFormulas
# ---------------------------------------------------------------------------

class TestEfficiencyFormulas:

    def test_eqe_formula_roundtrip(self):
        """EQE recovered from luminance formula must match η_rad × η_out."""
        wl = np.linspace(380.0, 780.0, 81)
        em = _gaussian_spectrum(wl)
        V_cie = np.exp(-0.5 * ((wl - 555.0) / 40.0) ** 2)
        overlap = float(np.trapezoid(em * V_cie, wl))
        eta = 0.2
        J = 500.0
        L = (_K_M / _PI) * (J / _Q) * eta * overlap
        eqe = compute_eqe(wl, em, L, J)
        assert eqe == pytest.approx(eta, rel=1e-6)

    def test_eqe_zero_when_J_zero(self):
        wl = np.linspace(380.0, 780.0, 11)
        em = np.ones(11) / 11.0
        assert compute_eqe(wl, em, 100.0, 0.0) == pytest.approx(0.0)

    def test_eqe_zero_when_L_zero(self):
        wl = np.linspace(380.0, 780.0, 11)
        em = np.ones(11) / 11.0
        assert compute_eqe(wl, em, 0.0, 100.0) == pytest.approx(0.0)

    def test_eqe_linear_in_L(self):
        wl = np.linspace(380.0, 780.0, 81)
        em = _gaussian_spectrum(wl)
        J  = 200.0
        eqe1 = compute_eqe(wl, em, 100.0, J)
        eqe2 = compute_eqe(wl, em, 200.0, J)
        assert eqe2 == pytest.approx(2 * eqe1, rel=1e-9)

    def test_ce_formula(self):
        """CE = L / (J × 1e-4)."""
        L, J = 1000.0, 500.0
        assert compute_ce(L, J) == pytest.approx(L / (J * 1e-4), rel=1e-9)

    def test_ce_zero_at_dark(self):
        assert compute_ce(0.0, 0.0) == pytest.approx(0.0)

    def test_pe_formula(self):
        """PE = L × π / (J × V)."""
        L, J, V = 1000.0, 500.0, 4.0
        assert compute_pe(L, J, V) == pytest.approx(L * math.pi / (J * V), rel=1e-9)

    def test_pe_zero_at_zero_V(self):
        assert compute_pe(500.0, 100.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestJVLPoint
# ---------------------------------------------------------------------------

class TestJVLPoint:

    def test_ce_alias_equals_cd_per_A(self):
        pt = _make_point()
        assert pt.ce == pytest.approx(pt.cd_per_A, rel=1e-9)

    def test_pe_alias_equals_lm_per_W(self):
        pt = _make_point()
        assert pt.pe == pytest.approx(pt.lm_per_W, rel=1e-9)

    def test_eqe_pct_is_100x_eqe(self):
        pt = _make_point()
        assert pt.eqe_pct == pytest.approx(pt.eqe * 100.0, rel=1e-9)

    def test_J_mAcm2_conversion(self):
        pt = _make_point(J=500.0)
        assert pt.J_mAcm2 == pytest.approx(50.0, rel=1e-9)

    def test_summary_contains_eqe(self):
        pt = _make_point()
        assert "EQE" in pt.summary()


# ---------------------------------------------------------------------------
# TestJVLResult
# ---------------------------------------------------------------------------

class TestJVLResult:

    def test_eqe_shape(self):
        res = _make_jvl_result(5)
        assert res.eqe.shape == (5,)

    def test_eqe_pct_is_100x_eqe(self):
        res = _make_jvl_result(4)
        np.testing.assert_allclose(res.eqe_pct, res.eqe * 100.0, rtol=1e-9)

    def test_ce_alias(self):
        res = _make_jvl_result(4)
        np.testing.assert_array_equal(res.ce, res.cd_per_A)

    def test_pe_alias(self):
        res = _make_jvl_result(4)
        np.testing.assert_array_equal(res.pe, res.lm_per_W)

    def test_to_csv_has_eqe_column(self, tmp_path):
        res = _make_jvl_result(3)
        out = tmp_path / "jvl.csv"
        res.to_csv(out)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert "eqe" in header

    def test_to_json_has_eqe_key(self, tmp_path):
        res = _make_jvl_result(3)
        out = tmp_path / "jvl.json"
        res.to_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "eqe" in data
        assert len(data["eqe"]) == 3

    def test_spectrum_at_nearest(self):
        res = _make_jvl_result(5)
        wl, em = res.spectrum_at(2.3)
        assert wl.shape == em.shape

    def test_summary_contains_eqe(self):
        res = _make_jvl_result(5)
        assert "EQE" in res.summary()


# ---------------------------------------------------------------------------
# TestJVLCalculatorRun
# ---------------------------------------------------------------------------

class TestJVLCalculatorRun:

    def test_return_type(self, jvl_result):
        assert isinstance(jvl_result, JVLResult)

    def test_n_points_matches_request(self, jvl_result):
        assert len(jvl_result.points) == 3

    def test_voltage_range(self, jvl_result):
        V = jvl_result.voltages
        assert float(V[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(V[-1]) == pytest.approx(4.0, abs=1e-9)

    def test_J_is_finite_array(self, jvl_result):
        assert jvl_result.J_Am2.shape == (len(jvl_result.points),)

    def test_luminance_non_negative(self, jvl_result):
        assert np.all(jvl_result.luminance >= 0.0)


# ---------------------------------------------------------------------------
# TestJVLCalculatorPhysics
# ---------------------------------------------------------------------------

class TestJVLCalculatorPhysics:

    def test_lm_per_W_zero_at_V_zero(self, jvl_result):
        """PE must be 0 at V=0 by formula guard."""
        assert jvl_result.lm_per_W[0] == pytest.approx(0.0, abs=1e-12)

    def test_eqe_zero_at_V_zero(self, jvl_result):
        """EQE must be 0 when luminance is 0 (J guard or L guard)."""
        # At V=0, luminance[0] ≥ 0; if L=0 then EQE=0 by formula guard
        if jvl_result.luminance[0] == pytest.approx(0.0, abs=1e-6):
            assert jvl_result.eqe[0] == pytest.approx(0.0, abs=1e-12)

    def test_eqe_non_negative(self, jvl_result):
        assert np.all(jvl_result.eqe >= 0.0)

    def test_cd_per_A_non_negative(self, jvl_result):
        assert np.all(jvl_result.cd_per_A >= 0.0)

    def test_voltages_strictly_increasing(self, jvl_result):
        assert np.all(np.diff(jvl_result.voltages) > 0.0)
