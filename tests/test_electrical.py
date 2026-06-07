"""Tests for Step 7 — 1D electrical solver skeleton.

Coverage
--------
TestMesh1D          (10) — node grid, layer assignment, property arrays
TestNodeProps        (5) — organic vs electrode property derivation
TestDeviceState      (7) — state container, derived quantities, E_field
TestPoissonSolver    (9) — matrix structure, BCs, linear-φ correctness
TestContinuity       (6) — interface contract, Bernoulli, SRH
TestGummelSolver     (7) — equilibrium state, Gummel iteration
TestBiasSweep        (6) — voltage sweep orchestration
TestSweepResult      (5) — aggregated J-V arrays, I/O
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
NK_ROOT = ROOT / "data" / "nk"

from src.io import load_project
from src.electrical import (
    BiasSweepRunner,
    BiasPoint,
    ContinuitySolver,
    DeviceState,
    GummelConfig,
    GummelSolver,
    Mesh1D,
    NodeProps,
    PoissonSolver,
    SweepResult,
    build_mesh,
    build_solver,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def mesh(project):
    return build_mesh(project.device_stack, project.material_db, z_resolution_nm=1.0)


@pytest.fixture(scope="module")
def poisson(mesh):
    return PoissonSolver(mesh)


@pytest.fixture(scope="module")
def continuity(mesh):
    return ContinuitySolver(mesh)


@pytest.fixture(scope="module")
def gummel(mesh):
    return build_solver(mesh, GummelConfig(max_iterations=10, tolerance=1e-6))


@pytest.fixture(scope="module")
def sweep(gummel):
    return BiasSweepRunner(gummel, v_start=0.0, v_end=2.0, n_points=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_mesh(N: int = 7, eps_r: float = 3.0, dz_nm: float = 1.0) -> Mesh1D:
    """Minimal uniform single-layer mesh for isolated unit tests."""
    z_nm = np.arange(N, dtype=float) * dz_nm
    z_m  = z_nm * 1e-9
    dz_m = np.full(N - 1, dz_nm * 1e-9)
    props = [
        NodeProps(
            eps_r=eps_r, chi_eV=2.0, Eg_eV=3.0,
            Nc=1e27, Nv=1e27, mu_n=1e-10, mu_p=1e-10,
            tau_n=1e-6, tau_p=1e-6, N_doping=0.0,
            is_electrode=False, work_function_eV=0.0,
        )
        for _ in range(N)
    ]
    return Mesh1D(
        z_nm=z_nm, z_m=z_m, dz_m=dz_m,
        node_layer=np.zeros(N, dtype=int),
        layer_names=["test"],
        layer_boundaries_nm=np.array([0.0, float(N - 1) * dz_nm]),
        node_props=props,
    )


# ---------------------------------------------------------------------------
# TestMesh1D
# ---------------------------------------------------------------------------

class TestMesh1D:

    def test_build_returns_mesh1d(self, mesh):
        assert isinstance(mesh, Mesh1D)

    def test_num_nodes_positive(self, mesh):
        assert mesh.num_nodes > 0

    def test_z_nm_starts_at_zero(self, mesh):
        assert mesh.z_nm[0] == pytest.approx(0.0)

    def test_z_nm_ends_at_total_thickness(self, project, mesh):
        total = project.device_stack.total_thickness_nm()
        assert mesh.z_nm[-1] == pytest.approx(total, abs=1.0)

    def test_num_edges_equals_nodes_minus_one(self, mesh):
        assert mesh.num_edges == mesh.num_nodes - 1

    def test_dz_m_all_positive(self, mesh):
        assert np.all(mesh.dz_m > 0)

    def test_layer_boundaries_count(self, project, mesh):
        n_layers = len(project.device_stack.layers)
        assert len(mesh.layer_boundaries_nm) == n_layers + 1

    def test_node_props_length_matches_nodes(self, mesh):
        assert len(mesh.node_props) == mesh.num_nodes

    def test_layer_mask_sums_to_num_nodes(self, mesh):
        total = sum(mesh.layer_mask(name).sum() for name in mesh.layer_names)
        assert total == mesh.num_nodes

    def test_layer_slice_returns_valid_slice(self, mesh):
        sl = mesh.layer_slice("EML")
        assert isinstance(sl, slice)
        assert sl.stop > sl.start


# ---------------------------------------------------------------------------
# TestNodeProps
# ---------------------------------------------------------------------------

class TestNodeProps:

    def test_organic_bandgap_positive(self, project, mesh):
        """TCTA HOMO=−5.7, LUMO=−2.4 → Eg = LUMO−HOMO = 3.3 eV."""
        tcta_sl = mesh.layer_slice("TCTA")
        props   = mesh.node_props[tcta_sl.start]
        assert props.Eg_eV > 0

    def test_organic_chi_positive(self, mesh):
        """Electron affinity (χ = −LUMO) must be positive for organics."""
        sl = mesh.layer_slice("TCTA")
        assert mesh.node_props[sl.start].chi_eV > 0

    def test_electrode_is_flagged(self, mesh):
        """ITO and Al layers must be flagged as electrodes."""
        ito_sl = mesh.layer_slice("ITO")
        al_sl  = mesh.layer_slice("Al")
        assert mesh.node_props[ito_sl.start].is_electrode
        assert mesh.node_props[al_sl.start].is_electrode

    def test_organic_not_electrode(self, mesh):
        eml_sl = mesh.layer_slice("EML")
        assert not mesh.node_props[eml_sl.start].is_electrode

    def test_mobilities_are_positive(self, mesh):
        for p in mesh.node_props:
            assert p.mu_n > 0 and p.mu_p > 0


# ---------------------------------------------------------------------------
# TestDeviceState
# ---------------------------------------------------------------------------

class TestDeviceState:

    def _zero_state(self, N: int = 10) -> DeviceState:
        return DeviceState(
            voltage_V=0.0,
            phi_V=np.zeros(N),
            n_m3=np.ones(N) * 1e10,
            p_m3=np.ones(N) * 1e10,
            Jn_Am2=np.zeros(N - 1),
            Jp_Am2=np.zeros(N - 1),
            dz_m=np.full(N - 1, 1e-9),
        )

    def test_can_construct(self):
        state = self._zero_state()
        assert isinstance(state, DeviceState)

    def test_j_total_is_float(self):
        state = self._zero_state()
        assert isinstance(state.J_total, float)

    def test_j_total_zero_for_zero_currents(self):
        state = self._zero_state()
        assert state.J_total == pytest.approx(0.0)

    def test_j_total_mAcm2_scaling(self):
        state = self._zero_state(10)
        state.Jn_Am2[:] = 100.0   # 100 A/m² = 10 mA/cm²
        assert state.J_total_mAcm2 == pytest.approx(10.0)

    def test_e_field_shape(self):
        N = 10
        state = self._zero_state(N)
        state.phi_V = np.linspace(1.0, 0.0, N)
        E = state.E_field_Vm
        assert E is not None and len(E) == N - 1

    def test_e_field_none_without_dz(self):
        N = 5
        state = DeviceState(0.0, np.zeros(N), np.zeros(N), np.zeros(N),
                            np.zeros(N-1), np.zeros(N-1), dz_m=None)
        assert state.E_field_Vm is None

    def test_np_product_shape(self):
        state = self._zero_state(10)
        assert state.np_product.shape == (10,)


# ---------------------------------------------------------------------------
# TestPoissonSolver
# ---------------------------------------------------------------------------

class TestPoissonSolver:

    def test_can_instantiate(self, poisson):
        assert isinstance(poisson, PoissonSolver)

    def test_assemble_returns_correct_shapes(self, poisson, mesh):
        N  = mesh.num_nodes
        n  = np.zeros(N)
        p  = np.zeros(N)
        A, b = poisson.assemble(n, p)
        assert A.shape == (N, N)
        assert b.shape == (N,)

    def test_a_matrix_is_square(self, poisson, mesh):
        N = mesh.num_nodes
        A, _ = poisson.assemble(np.zeros(N), np.zeros(N))
        assert A.shape[0] == A.shape[1]

    def test_interior_diagonal_negative(self, poisson, mesh):
        """Interior diagonal must be negative for positive ε_r."""
        N = mesh.num_nodes
        A, _ = poisson.assemble(np.zeros(N), np.zeros(N))
        for i in range(1, N - 1):
            assert A[i, i] < 0, f"A[{i},{i}] = {A[i,i]:.4e} is not negative"

    def test_apply_dirichlet_sets_boundary_rows(self, poisson, mesh):
        N = mesh.num_nodes
        A, b = poisson.assemble(np.zeros(N), np.zeros(N))
        A, b = poisson.apply_dirichlet(A, b, 1.0, 0.0)
        assert A[0, 0]   == pytest.approx(1.0) and np.all(A[0, 1:] == 0)
        assert A[-1, -1] == pytest.approx(1.0) and np.all(A[-1, :-1] == 0)
        assert b[0]  == pytest.approx(1.0)
        assert b[-1] == pytest.approx(0.0)

    def test_solve_returns_correct_length(self, poisson, mesh):
        N   = mesh.num_nodes
        phi = poisson.solve(np.zeros(N), np.zeros(N), 1.0, 0.0)
        assert len(phi) == N

    def test_solve_boundary_values_applied(self, poisson, mesh):
        N   = mesh.num_nodes
        phi = poisson.solve(np.zeros(N), np.zeros(N), 2.0, 0.5)
        assert phi[0]  == pytest.approx(2.0, abs=1e-10)
        assert phi[-1] == pytest.approx(0.5, abs=1e-10)

    def test_solve_linear_phi_uniform_eps_zero_charge(self):
        """Uniform ε, zero charge, linear BCs → φ must be linear."""
        mesh_u = _uniform_mesh(N=7, eps_r=3.0)
        solver = PoissonSolver(mesh_u)
        N      = mesh_u.num_nodes
        phi    = solver.solve(np.zeros(N), np.zeros(N), 1.0, 0.0)
        expected = np.linspace(1.0, 0.0, N)
        np.testing.assert_allclose(phi, expected, atol=1e-10)

    def test_solve_symmetric_bc_gives_uniform_phi(self):
        """Equal BCs with zero charge → φ is constant."""
        mesh_u = _uniform_mesh(N=5)
        solver = PoissonSolver(mesh_u)
        N = mesh_u.num_nodes
        phi = solver.solve(np.zeros(N), np.zeros(N), 0.5, 0.5)
        np.testing.assert_allclose(phi, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# TestContinuity
# ---------------------------------------------------------------------------

class TestContinuity:

    def test_can_instantiate(self, continuity):
        assert isinstance(continuity, ContinuitySolver)

    def test_solve_electrons_raises_not_implemented(self, continuity, mesh):
        N   = mesh.num_nodes
        phi = np.zeros(N)
        p   = np.full(N, 1e10)
        with pytest.raises(NotImplementedError):
            continuity.solve_electrons(phi, p, 1e10, 1e10)

    def test_solve_holes_raises_not_implemented(self, continuity, mesh):
        N   = mesh.num_nodes
        phi = np.zeros(N)
        n   = np.full(N, 1e10)
        with pytest.raises(NotImplementedError):
            continuity.solve_holes(phi, n, 1e10, 1e10)

    def test_bernoulli_at_zero(self, continuity):
        """B(0) = 1 from the Taylor expansion."""
        assert continuity.bernoulli(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_bernoulli_array_input(self, continuity):
        x   = np.array([-1.0, 0.0, 1.0])
        out = continuity.bernoulli(x)
        assert out.shape == (3,)

    def test_recombination_at_equilibrium_is_zero(self, continuity):
        """SRH rate must vanish when n·p = ni²."""
        N  = _uniform_mesh().num_nodes
        ni = 1e10
        n  = np.full(N, ni)
        p  = np.full(N, ni)
        # Build a ContinuitySolver on the uniform mesh
        cont = ContinuitySolver(_uniform_mesh(N))
        R = cont.recombination_srh(n, p, ni=ni)
        np.testing.assert_allclose(R, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# TestGummelSolver
# ---------------------------------------------------------------------------

class TestGummelSolver:

    def test_can_instantiate(self, gummel):
        assert isinstance(gummel, GummelSolver)

    def test_equilibrium_state_returns_device_state(self, gummel):
        state = gummel.equilibrium_state()
        assert isinstance(state, DeviceState)

    def test_equilibrium_phi_shape(self, gummel, mesh):
        state = gummel.equilibrium_state()
        assert state.phi_V.shape == (mesh.num_nodes,)

    def test_equilibrium_n_positive(self, gummel):
        state = gummel.equilibrium_state()
        assert np.all(state.n_m3 > 0)

    def test_solve_returns_bias_point(self, gummel):
        bp = gummel.solve(V_anode=0.0)
        assert isinstance(bp, BiasPoint)

    def test_solve_voltage_stored_correctly(self, gummel):
        bp = gummel.solve(V_anode=1.0, V_cathode=0.0)
        assert bp.voltage_V == pytest.approx(1.0)

    def test_solve_state_phi_shape(self, gummel, mesh):
        bp = gummel.solve(V_anode=0.0)
        assert bp.state.phi_V.shape == (mesh.num_nodes,)


# ---------------------------------------------------------------------------
# TestBiasSweep
# ---------------------------------------------------------------------------

class TestBiasSweep:

    def test_can_instantiate(self, sweep):
        assert isinstance(sweep, BiasSweepRunner)

    def test_voltages_property_length(self, sweep):
        assert len(sweep.voltages) == sweep.n_points

    def test_voltages_start_value(self, sweep):
        assert sweep.voltages[0] == pytest.approx(sweep.v_start)

    def test_voltages_end_value(self, sweep):
        assert sweep.voltages[-1] == pytest.approx(sweep.v_end)

    def test_run_returns_sweep_result(self, sweep):
        result = sweep.run()
        assert isinstance(result, SweepResult)

    def test_run_produces_correct_number_of_points(self, sweep):
        result = sweep.run()
        assert len(result.bias_points) == sweep.n_points


# ---------------------------------------------------------------------------
# TestSweepResult
# ---------------------------------------------------------------------------

class TestSweepResult:

    @pytest.fixture(scope="class")
    def result(self, gummel):
        runner = BiasSweepRunner(gummel, v_start=0.0, v_end=1.0, n_points=3)
        return runner.run()

    def test_voltages_array_shape(self, result):
        assert result.voltages.shape == (3,)

    def test_j_total_array_shape(self, result):
        assert result.J_total.shape == (3,)

    def test_all_bias_points_have_state(self, result):
        for bp in result.bias_points:
            assert isinstance(bp.state, DeviceState)

    def test_to_csv_creates_file(self, result, tmp_path):
        out = tmp_path / "jv.csv"
        result.to_csv(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert "voltage_V" in lines[0]
        assert len(lines) == 4   # header + 3 data rows

    def test_to_json_creates_file(self, result, tmp_path):
        out = tmp_path / "jv.json"
        result.to_json(out)
        data = json.loads(out.read_text())
        assert "voltages_V" in data
        assert "J_Am2" in data
        assert data["n_points"] == 3
