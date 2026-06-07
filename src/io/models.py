"""Typed internal data structures for SETFOS input data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Material DB
# ---------------------------------------------------------------------------

@dataclass
class ElectrodeParams:
    type: str               # anode | cathode | contact
    work_function: float    # eV


@dataclass
class ThermalProperties:
    heat_capacity: float | None         # J/(g·K)
    thermal_conductivity: float | None  # W/(m·K)
    expansion_coefficient: float | None


@dataclass
class MaterialEntry:
    material_name: str
    nk_reference: str | None           # path to n/k CSV
    homo: float | None                  # eV
    lumo: float | None                  # eV
    electron_mobility: float | None     # cm²/V/s
    hole_mobility: float | None         # cm²/V/s
    dielectric_constant: float | None
    electrode: ElectrodeParams | None
    thermal: ThermalProperties
    notes: str

    def is_electrode(self) -> bool:
        return self.electrode is not None


@dataclass
class MaterialDB:
    version: float | None
    materials: dict[str, MaterialEntry]

    def get(self, name: str) -> MaterialEntry:
        if name not in self.materials:
            raise KeyError(f"Material '{name}' not found in DB. Available: {sorted(self.materials)}")
        return self.materials[name]


# ---------------------------------------------------------------------------
# Device Stack
# ---------------------------------------------------------------------------

@dataclass
class ElectrodeLayer:
    is_electrode: bool
    type: str | None        # transparent | metal | semiconductor | unknown
    work_function: float | None  # eV


@dataclass
class EmitterParams:
    dopant: str
    dopant_fraction: str


@dataclass
class Layer:
    order: int
    layer_name: str
    material_name: str
    thickness_nm: float
    layer_role: str
    electrode: ElectrodeLayer | None
    emitter: EmitterParams | None

    def is_emitter(self) -> bool:
        return self.layer_role == "emitter"

    def is_electrode(self) -> bool:
        return self.electrode is not None and self.electrode.is_electrode


@dataclass
class DeviceStack:
    version: float | None
    name: str
    substrate: str
    layers: list[Layer]     # sorted by order ascending

    def total_thickness_nm(self) -> float:
        return sum(l.thickness_nm for l in self.layers)

    def get_layers_by_role(self, role: str) -> list[Layer]:
        return [l for l in self.layers if l.layer_role == role]


# ---------------------------------------------------------------------------
# Solver Config
# ---------------------------------------------------------------------------

@dataclass
class WavelengthGrid:
    start_nm: float
    end_nm: float
    step_nm: float

    def num_points(self) -> int:
        return round((self.end_nm - self.start_nm) / self.step_nm) + 1


@dataclass
class BiasSweep:
    start_V: float
    end_V: float
    step_V: float

    def num_points(self) -> int:
        return round(abs(self.end_V - self.start_V) / self.step_V) + 1


@dataclass
class ConvergenceCriteria:
    tolerance: float
    criterion: str   # residual | field | current
    norm: str        # L2 | Linf


@dataclass
class DampingConfig:
    enabled: bool
    factor: float    # 0 < factor <= 1
    adaptive: bool


@dataclass
class MeshConfig:
    resolution_nm: float


@dataclass
class PhysicsFlags:
    include_thermal: bool
    include_excitonics: bool
    include_charge_transport: bool


@dataclass
class BoundaryConditions:
    anode_potential_eV: float
    cathode_potential_eV: float


@dataclass
class OutputConfig:
    save_fields: bool
    output_folder: str


@dataclass
class SolverConfig:
    version: float | None
    simulation_name: str
    simulation_type: str            # OLED | PV | transfer
    max_iterations: int
    wavelength_grid: WavelengthGrid
    bias_sweep: BiasSweep
    convergence: ConvergenceCriteria
    damping: DampingConfig
    mesh: MeshConfig
    physics: PhysicsFlags
    boundary_conditions: BoundaryConditions
    output: OutputConfig
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class MeasurementUnits:
    voltage: str
    current_density: str
    luminance: str | None
    spectral_quantity: str | None


@dataclass
class Measurement:
    version: float | None
    name: str
    type: str           # IV | IVL | EL | EQE | spectral
    data_file: str | None
    units: MeasurementUnits
    temperature_K: float | None
    scan_direction: str | None
    comments: str
    data: list[dict[str, str]] | None   # rows from CSV, None if not loaded


# ---------------------------------------------------------------------------
# Project (root aggregator)
# ---------------------------------------------------------------------------

@dataclass
class Project:
    name: str
    description: str
    material_db: MaterialDB
    device_stack: DeviceStack
    measurement: Measurement
    solver_config: SolverConfig


# ---------------------------------------------------------------------------
# Emitter (optical emission specification)
# ---------------------------------------------------------------------------

@dataclass
class EmitterConfig:
    """Optical emitter specification — loaded from a standalone YAML file.

    Fields
    ------
    layer_name          : must match the layer_name in the DeviceStack
    material            : emitter material name (informational)
    pl_spectrum_file    : path to PL CSV with columns [wavelength_nm, intensity]
    horizontal_fraction : fraction of horizontally oriented dipoles [0, 1]
                          (1.0 = all horizontal, 0.0 = all vertical)
    z_distribution      : emitter position within the EML —
                          "center" | "uniform" | "front" | "back"
    emitter_type        : "phosphorescent" | "fluorescent"
    notes               : free-text annotation
    """

    layer_name: str
    material: str
    pl_spectrum_file: str
    horizontal_fraction: float
    z_distribution: str
    emitter_type: str
    notes: str = ""
