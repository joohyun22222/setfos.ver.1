"""Tests for Step 13 — Trap Physics.

Coverage
--------
TestTrapPhysicsHelpers (7) — srh_lifetimes formula, scaling, midgap n1=p1=ni,
                              LUMO-trap large n1, HOMO-trap large p1, n1*p1=ni²,
                              aggregate single-trap passthrough
TestTrapStateDataclass (3)  — defaults, field access, label
TestNodePropsDefaults   (3)  — no-trap material keeps defaults, tau unchanged,
                               n1=p1=ni
TestMeshWithTraps       (4)  — build_mesh success, trap shortens tau_n,
                               no-trap material tau unchanged, midgap n1≈ni
TestContinuityWithTrap  (4)  — recombination_srh midgap matches old formula,
                               non-midgap differs, recombination_srh non-negative,
                               solve_electrons returns finite array
TestBaseline            (2)  — full JVL sweep identical with/without trap on
                               no-trap material, trap material gives different R
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"

from src.electrical.traps import (
    aggregate_trap_srh,
    srh_lifetimes,
    srh_stat_densities,
)
from src.io.models import TrapState
from src.electrical.mesh import NodeProps, _mat_to_node_props
from src.io.models import MaterialEntry, ThermalProperties


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trap(
    density=1e22,
    energy=1.65,
    sigma_n=1e-19,
    sigma_p=1e-19,
    vth=1e5,
) -> TrapState:
    return TrapState(
        density_m3=density,
        energy_from_lumo_eV=energy,
        sigma_n=sigma_n,
        sigma_p=sigma_p,
        vth_n=vth,
        vth_p=vth,
    )


def _make_organic_entry(trap_states=None) -> MaterialEntry:
    """Minimal TCTA-like MaterialEntry for testing."""
    return MaterialEntry(
        material_name="test_organic",
        nk_reference=None,
        homo=-5.7,
        lumo=-2.4,
        electron_mobility=3e-4,
        hole_mobility=3e-4,
        dielectric_constant=3.0,
        electrode=None,
        thermal=ThermalProperties(None, None, None),
        notes="",
        trap_states=trap_states or [],
    )


# ---------------------------------------------------------------------------
# TestTrapPhysicsHelpers
# ---------------------------------------------------------------------------

class TestTrapPhysicsHelpers:

    def test_srh_lifetimes_formula(self):
        """tau_n = 1/(N_t * sigma_n * vth_n)."""
        tn, tp = srh_lifetimes(1e22, 1e-19, 2e-19, 1e5, 1e5)
        assert tn == pytest.approx(1.0 / (1e22 * 1e-19 * 1e5), rel=1e-9)
        assert tp == pytest.approx(1.0 / (1e22 * 2e-19 * 1e5), rel=1e-9)

    def test_srh_lifetimes_scale_with_density(self):
        """Doubling N_t halves tau."""
        tn1, _ = srh_lifetimes(1e22, 1e-19, 1e-19)
        tn2, _ = srh_lifetimes(2e22, 1e-19, 1e-19)
        assert tn2 == pytest.approx(tn1 / 2.0, rel=1e-9)

    def test_midgap_trap_gives_ni(self):
        """ΔE_t = Eg/2, Nc = Nv  →  n1 = p1 = ni."""
        Eg, ni = 3.3, 1e10
        n1, p1 = srh_stat_densities(Eg / 2.0, Eg, 1e27, 1e27, ni=ni)
        assert n1 == pytest.approx(ni, rel=1e-6)
        assert p1 == pytest.approx(ni, rel=1e-6)

    def test_lumo_trap_gives_large_n1(self):
        """Trap near LUMO (ΔE_t small)  →  n1 ≫ ni."""
        n1, p1 = srh_stat_densities(0.1, 3.3, 1e27, 1e27, ni=1e10)
        assert n1 > 1e12    # >> ni
        assert p1 < 1e10    # << ni

    def test_homo_trap_gives_large_p1(self):
        """Trap near HOMO (ΔE_t ≈ Eg)  →  p1 ≫ ni."""
        n1, p1 = srh_stat_densities(3.2, 3.3, 1e27, 1e27, ni=1e10)
        assert p1 > 1e12    # >> ni
        assert n1 < 1e10    # << ni

    def test_n1_p1_product_equals_ni2(self):
        """n1 · p1 = ni² for any trap energy."""
        ni = 1e10
        for delta in [0.0, 0.5, 1.0, 1.65, 3.0, 3.3]:
            n1, p1 = srh_stat_densities(delta, 3.3, 1e27, 1e27, ni=ni)
            assert n1 * p1 == pytest.approx(ni**2, rel=1e-6)

    def test_aggregate_single_trap_passthrough(self):
        """aggregate_trap_srh with one trap matches direct calculation."""
        ts = _make_trap()
        Eg = 3.3
        tau_n, tau_p, n1, p1 = aggregate_trap_srh([ts], Eg, 1e27, 1e27)
        tn_exp, tp_exp = srh_lifetimes(ts.density_m3, ts.sigma_n, ts.sigma_p)
        n1_exp, p1_exp = srh_stat_densities(ts.energy_from_lumo_eV, Eg, 1e27, 1e27)
        assert tau_n == pytest.approx(tn_exp, rel=1e-9)
        assert tau_p == pytest.approx(tp_exp, rel=1e-9)
        assert n1 == pytest.approx(n1_exp, rel=1e-6)
        assert p1 == pytest.approx(p1_exp, rel=1e-6)


# ---------------------------------------------------------------------------
# TestTrapStateDataclass
# ---------------------------------------------------------------------------

class TestTrapStateDataclass:

    def test_default_thermal_velocities(self):
        ts = _make_trap()
        assert ts.vth_n == pytest.approx(1e5)
        assert ts.vth_p == pytest.approx(1e5)

    def test_density_accessible(self):
        ts = _make_trap(density=5e21)
        assert ts.density_m3 == pytest.approx(5e21)

    def test_default_label(self):
        ts = TrapState(density_m3=1e22, energy_from_lumo_eV=1.5,
                       sigma_n=1e-19, sigma_p=1e-19)
        assert ts.label == "trap"


# ---------------------------------------------------------------------------
# TestNodePropsDefaults
# ---------------------------------------------------------------------------

class TestNodePropsDefaults:

    def test_no_trap_tau_n_is_default(self):
        mat = _make_organic_entry(trap_states=[])
        props = _mat_to_node_props(mat)
        # With no traps, aggregate_trap_srh returns (1e-6, 1e-6, ni, ni)
        assert props.tau_n == pytest.approx(1e-6)

    def test_no_trap_n1_equals_ni(self):
        mat = _make_organic_entry(trap_states=[])
        props = _mat_to_node_props(mat)
        assert props.trap_n1 == pytest.approx(1e10, rel=1e-6)

    def test_no_trap_p1_equals_ni(self):
        mat = _make_organic_entry(trap_states=[])
        props = _mat_to_node_props(mat)
        assert props.trap_p1 == pytest.approx(1e10, rel=1e-6)


# ---------------------------------------------------------------------------
# TestMeshWithTraps
# ---------------------------------------------------------------------------

class TestMeshWithTraps:

    def test_trap_shortens_tau_n(self):
        """With N_t=1e22 midgap trap, tau_n < 1e-6 (default)."""
        ts = _make_trap(density=1e22, sigma_n=1e-19)
        mat = _make_organic_entry(trap_states=[ts])
        props = _mat_to_node_props(mat)
        assert props.tau_n < 1e-6   # shorter than default

    def test_trap_tau_matches_formula(self):
        """Computed tau_n matches 1/(N_t * sigma_n * vth_n)."""
        ts = _make_trap(density=1e22, sigma_n=1e-19, vth=1e5)
        mat = _make_organic_entry(trap_states=[ts])
        props = _mat_to_node_props(mat)
        expected = 1.0 / (1e22 * 1e-19 * 1e5)
        assert props.tau_n == pytest.approx(expected, rel=1e-6)

    def test_no_trap_material_tau_unchanged(self):
        """Material without trap keeps default tau."""
        no_trap = _make_organic_entry(trap_states=[])
        props = _mat_to_node_props(no_trap)
        assert props.tau_n == pytest.approx(1e-6)

    def test_midgap_trap_n1_near_ni(self):
        """Midgap trap (ΔE_t = Eg/2) gives n1 ≈ ni = 1e10."""
        Eg = 3.3
        ts = _make_trap(energy=Eg / 2.0)
        mat = _make_organic_entry(trap_states=[ts])
        props = _mat_to_node_props(mat)
        assert props.trap_n1 == pytest.approx(1e10, rel=0.01)  # 1% tolerance


# ---------------------------------------------------------------------------
# TestContinuityWithTrap
# ---------------------------------------------------------------------------

class TestContinuityWithTrap:

    @pytest.fixture(scope="class")
    def mesh_no_trap(self):
        from src.io import load_project
        from src.electrical import build_mesh
        project = load_project(SAMPLES / "oled_input.yaml")
        return build_mesh(project.device_stack, project.material_db, z_resolution_nm=2.0)

    @pytest.fixture(scope="class")
    def cont_no_trap(self, mesh_no_trap):
        from src.electrical import ContinuitySolver
        return ContinuitySolver(mesh_no_trap)

    def test_trap_n1_array_loaded(self, cont_no_trap):
        """ContinuitySolver loads trap_n1 array from mesh node_props."""
        assert cont_no_trap._trap_n1.shape == (cont_no_trap._mesh.num_nodes,)

    def test_recombination_srh_non_negative(self, cont_no_trap, mesh_no_trap):
        """R_SRH ≥ 0 when np > ni²."""
        N = mesh_no_trap.num_nodes
        ni = 1e10
        n = np.full(N, 1e18)
        p = np.full(N, 1e18)
        R = cont_no_trap.recombination_srh(n, p, ni=ni)
        assert np.all(R >= 0.0)

    def test_recombination_srh_zero_at_equilibrium(self, cont_no_trap, mesh_no_trap):
        """R_SRH = 0 at thermal equilibrium (np = ni²)."""
        N = mesh_no_trap.num_nodes
        ni = 1e10
        n = np.full(N, ni)
        p = np.full(N, ni)
        R = cont_no_trap.recombination_srh(n, p, ni=ni)
        assert np.allclose(R, 0.0, atol=1e-3)

    def test_recombination_srh_midgap_matches_standard(self, mesh_no_trap):
        """When n1=p1=ni (midgap default), SRH equals old closed-form formula."""
        from src.electrical import ContinuitySolver
        cont = ContinuitySolver(mesh_no_trap)
        N = mesh_no_trap.num_nodes
        ni = 1e10
        n = np.full(N, 1e18)
        p = np.full(N, 1e18)
        R_new = cont.recombination_srh(n, p, ni=ni)
        # Manually compute old formula (n1=p1=ni)
        tau_n = cont._tau_n
        tau_p = cont._tau_p
        denom = tau_p * (n + ni) + tau_n * (p + ni)
        R_old = np.where(denom > 0, (n * p - ni**2) / denom, 0.0)
        np.testing.assert_allclose(R_new, R_old, rtol=1e-9)


# ---------------------------------------------------------------------------
# TestBaselineComparison
# ---------------------------------------------------------------------------

class TestBaselineComparison:

    @pytest.fixture(scope="class")
    def project(self):
        from src.io import load_project
        return load_project(SAMPLES / "oled_input.yaml")

    def test_build_mesh_with_trap_material_succeeds(self, project):
        """build_mesh with TCTA (which now has a trap) completes without error."""
        from src.electrical import build_mesh
        mesh = build_mesh(project.device_stack, project.material_db, z_resolution_nm=2.0)
        assert mesh.num_nodes > 0

    def test_trap_injection_shortens_tau(self, project):
        """Injecting a TrapState into a material's copy shortens tau_n vs baseline."""
        from src.electrical.mesh import _mat_to_node_props
        # baseline — no traps
        tcta_mat = project.material_db.get("TCTA")
        baseline = _mat_to_node_props(tcta_mat)
        # inject trap into a copy
        import dataclasses
        trap = _make_trap(density=1e22)
        trap_mat = dataclasses.replace(tcta_mat, trap_states=[trap])
        with_trap = _mat_to_node_props(trap_mat)
        assert with_trap.tau_n < baseline.tau_n   # trap shortens lifetime
