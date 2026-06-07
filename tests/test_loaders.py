"""Tests for src/io loaders and internal data structures."""

import pytest
from pathlib import Path

# Paths relative to project root
ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"
DATA = ROOT / "data"

from src.io import (
    load_material_db,
    load_device_stack,
    load_measurement,
    load_solver_config,
    load_project,
)
from src.io.models import (
    MaterialDB, MaterialEntry,
    DeviceStack, Layer,
    SolverConfig,
    Measurement,
    Project,
)


# ---------------------------------------------------------------------------
# Material DB
# ---------------------------------------------------------------------------

class TestMaterialDB:
    def setup_method(self):
        self.db = load_material_db(DATA / "materials" / "material_db.yaml")

    def test_returns_MaterialDB(self):
        assert isinstance(self.db, MaterialDB)

    def test_all_oled_materials_present(self):
        for name in ["ITO", "TCTA", "TPBi", "Al", "Ir(ppy)3"]:
            assert name in self.db.materials, f"Missing material: {name}"

    def test_material_entry_type(self):
        assert isinstance(self.db.materials["TCTA"], MaterialEntry)

    def test_homo_lumo_organic(self):
        tcta = self.db.get("TCTA")
        assert tcta.homo == pytest.approx(-5.7)
        assert tcta.lumo == pytest.approx(-2.4)

    def test_mobilities_organic(self):
        tcta = self.db.get("TCTA")
        assert tcta.hole_mobility > tcta.electron_mobility

        tpbi = self.db.get("TPBi")
        assert tpbi.electron_mobility > tpbi.hole_mobility

    def test_electrode_materials(self):
        ito = self.db.get("ITO")
        assert ito.is_electrode()
        assert ito.electrode.type == "anode"
        assert ito.electrode.work_function == pytest.approx(4.7)

        al = self.db.get("Al")
        assert al.is_electrode()
        assert al.electrode.type == "cathode"
        assert al.electrode.work_function == pytest.approx(4.2)

    def test_thermal_properties(self):
        al = self.db.get("Al")
        assert al.thermal.thermal_conductivity == pytest.approx(237.0)

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.db.get("NonExistentMaterial")

    def test_duplicate_name_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1.0\nformat: material-db\nmaterials:\n"
            "  - material_name: X\n  - material_name: X\n"
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_material_db(bad)

    def test_wrong_format_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1.0\nformat: device-stack\nmaterials: []\n")
        with pytest.raises(ValueError, match="format"):
            load_material_db(bad)


# ---------------------------------------------------------------------------
# Device Stack
# ---------------------------------------------------------------------------

class TestDeviceStack:
    def setup_method(self):
        self.stack = load_device_stack(SAMPLES / "device_stack_oled.yaml")

    def test_returns_DeviceStack(self):
        assert isinstance(self.stack, DeviceStack)

    def test_layer_count(self):
        assert len(self.stack.layers) == 5

    def test_layers_sorted_by_order(self):
        orders = [l.order for l in self.stack.layers]
        assert orders == sorted(orders)

    def test_layer_types(self):
        for layer in self.stack.layers:
            assert isinstance(layer, Layer)

    def test_layer_roles(self):
        roles = [l.layer_role for l in self.stack.layers]
        assert "anode" in roles
        assert "cathode" in roles
        assert "emitter" in roles

    def test_emitter_dopant(self):
        emitters = self.stack.get_layers_by_role("emitter")
        assert len(emitters) == 1
        assert emitters[0].emitter is not None
        assert emitters[0].emitter.dopant == "Ir(ppy)3"

    def test_electrode_layers(self):
        electrodes = [l for l in self.stack.layers if l.is_electrode()]
        assert len(electrodes) == 2

    def test_total_thickness(self):
        # ITO(150) + TCTA(40) + EML(30) + TPBi(50) + Al(100) = 370 nm
        assert self.stack.total_thickness_nm() == pytest.approx(370.0)

    def test_thickness_positive(self):
        for layer in self.stack.layers:
            assert layer.thickness_nm > 0

    def test_invalid_role_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1.0\nstack:\n  name: x\n  substrate: glass\n  layers:\n"
            "    - order: 1\n      layer_name: X\n      material_name: X\n"
            "      thickness_nm: 10\n      layer_role: invalid_role\n"
        )
        with pytest.raises(ValueError, match="layer_role"):
            load_device_stack(bad)


# ---------------------------------------------------------------------------
# Solver Config
# ---------------------------------------------------------------------------

class TestSolverConfig:
    def setup_method(self):
        self.sol = load_solver_config(SAMPLES / "solver_config_oled.yaml")

    def test_returns_SolverConfig(self):
        assert isinstance(self.sol, SolverConfig)

    def test_simulation_type(self):
        assert self.sol.simulation_type == "OLED"

    def test_max_iterations(self):
        assert self.sol.max_iterations == 200

    def test_wavelength_grid(self):
        wg = self.sol.wavelength_grid
        assert wg.start_nm == pytest.approx(380.0)
        assert wg.end_nm == pytest.approx(780.0)
        assert wg.step_nm == pytest.approx(5.0)
        assert wg.num_points() == 81

    def test_bias_sweep(self):
        bs = self.sol.bias_sweep
        assert bs.start_V == pytest.approx(0.0)
        assert bs.end_V == pytest.approx(8.0)
        assert bs.step_V == pytest.approx(0.1)
        assert bs.num_points() == 81

    def test_convergence(self):
        cv = self.sol.convergence
        assert cv.tolerance == pytest.approx(1e-7)
        assert cv.criterion == "residual"
        assert cv.norm == "L2"

    def test_damping(self):
        d = self.sol.damping
        assert d.enabled is True
        assert 0 < d.factor <= 1
        assert d.adaptive is True

    def test_mesh(self):
        assert self.sol.mesh.resolution_nm == pytest.approx(1.0)

    def test_physics_flags(self):
        ph = self.sol.physics
        assert ph.include_thermal is True
        assert ph.include_excitonics is True
        assert ph.include_charge_transport is True

    def test_boundary_conditions(self):
        bc = self.sol.boundary_conditions
        assert bc.anode_potential_eV == pytest.approx(4.7)
        assert bc.cathode_potential_eV == pytest.approx(4.2)

    def test_invalid_simulation_type_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1.0\nsolver:\n  simulation_name: x\n  simulation_type: UNKNOWN\n"
            "  max_iterations: 10\n  wavelength_grid: {start_nm: 380, end_nm: 780, step_nm: 5}\n"
            "  bias_sweep: {start_V: 0, end_V: 8, step_V: 0.1}\n"
            "  convergence: {tolerance: 1e-7, criterion: residual, norm: L2}\n"
            "  damping: {enabled: true, factor: 0.5, adaptive: true}\n"
            "  mesh: {resolution_nm: 1}\n"
            "  physics: {include_thermal: true, include_excitonics: true, include_charge_transport: true}\n"
            "  boundary_conditions:\n    anode: {potential_eV: 4.7}\n    cathode: {potential_eV: 4.2}\n"
        )
        with pytest.raises(ValueError, match="simulation_type"):
            load_solver_config(bad)

    def test_invalid_damping_factor_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1.0\nsolver:\n  simulation_name: x\n  simulation_type: OLED\n"
            "  max_iterations: 10\n  wavelength_grid: {start_nm: 380, end_nm: 780, step_nm: 5}\n"
            "  bias_sweep: {start_V: 0, end_V: 8, step_V: 0.1}\n"
            "  convergence: {tolerance: 1e-7, criterion: residual, norm: L2}\n"
            "  damping: {enabled: true, factor: 1.5, adaptive: false}\n"
            "  mesh: {resolution_nm: 1}\n"
            "  physics: {include_thermal: true, include_excitonics: true, include_charge_transport: true}\n"
            "  boundary_conditions:\n    anode: {potential_eV: 4.7}\n    cathode: {potential_eV: 4.2}\n"
        )
        with pytest.raises(ValueError, match="factor"):
            load_solver_config(bad)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class TestMeasurement:
    def setup_method(self):
        self.meas = load_measurement(SAMPLES / "measurement_oled.yaml", load_data=False)

    def test_returns_Measurement(self):
        assert isinstance(self.meas, Measurement)

    def test_type(self):
        assert self.meas.type == "IVL"

    def test_units(self):
        assert self.meas.units.voltage == "V"
        assert self.meas.units.current_density == "mA/cm2"

    def test_no_data_when_skipped(self):
        assert self.meas.data is None


# ---------------------------------------------------------------------------
# Project (end-to-end)
# ---------------------------------------------------------------------------

class TestProject:
    def setup_method(self):
        self.proj = load_project(SAMPLES / "oled_input.yaml")

    def test_returns_Project(self):
        assert isinstance(self.proj, Project)

    def test_all_sub_structures_populated(self):
        assert isinstance(self.proj.material_db, MaterialDB)
        assert isinstance(self.proj.device_stack, DeviceStack)
        assert isinstance(self.proj.measurement, Measurement)
        assert isinstance(self.proj.solver_config, SolverConfig)

    def test_device_materials_in_db(self):
        db = self.proj.material_db
        stack = self.proj.device_stack
        for layer in stack.layers:
            assert layer.material_name in db.materials, (
                f"Layer '{layer.layer_name}' references material '{layer.material_name}' "
                f"which is not in the material DB"
            )

    def test_solver_wavelength_range_valid(self):
        wg = self.proj.solver_config.wavelength_grid
        assert wg.start_nm < wg.end_nm
        assert wg.step_nm > 0
