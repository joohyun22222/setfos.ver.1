"""Loaders for SETFOS input files: material DB, device stack, measurement, solver config."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BiasSweep,
    BoundaryConditions,
    ConvergenceCriteria,
    DampingConfig,
    DeviceStack,
    ElectrodeLayer,
    ElectrodeParams,
    EmitterConfig,
    EmitterParams,
    Layer,
    MaterialDB,
    MaterialEntry,
    Measurement,
    MeasurementUnits,
    MeshConfig,
    OutputConfig,
    PhysicsFlags,
    Project,
    SolverConfig,
    ThermalProperties,
    TrapState,
    WavelengthGrid,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data).__name__} in {path}")
    return data


def _resolve(path: Path, base: Path) -> Path:
    if not path.is_absolute():
        return (base / path).resolve()
    return path.resolve()


def _require(obj: dict, key: str, label: str) -> Any:
    if key not in obj or obj[key] is None:
        raise ValueError(f"[{label}] required field '{key}' is missing or null")
    return obj[key]


def _get(obj: dict, key: str, default: Any = None) -> Any:
    return obj.get(key, default)


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

_VALID_LAYER_ROLES = {
    "anode", "hole_transport", "emitter", "electron_transport",
    "cathode", "blocking", "interlayer",
}
_VALID_MEASUREMENT_TYPES = {"IV", "IVL", "EL", "EQE", "spectral"}
_VALID_SIMULATION_TYPES = {"OLED", "PV", "transfer"}
_VALID_CONVERGENCE_CRITERIA = {"residual", "field", "current"}
_VALID_NORMS = {"L2", "Linf"}


# ---------------------------------------------------------------------------
# Material DB
# ---------------------------------------------------------------------------

def _parse_material_entry(raw: dict) -> MaterialEntry:
    name = _require(raw, "material_name", "material_db.material")

    optical = _get(raw, "optical", {})
    nk_ref = _get(optical, "nk_reference")

    electrode_raw = _get(raw, "electrode")
    electrode = None
    if electrode_raw and _get(electrode_raw, "work_function") is not None:
        electrode = ElectrodeParams(
            type=_get(electrode_raw, "type", "contact"),
            work_function=float(electrode_raw["work_function"]),
        )

    therm_raw = _get(raw, "thermal_properties", {}) or {}
    thermal = ThermalProperties(
        heat_capacity=_get(therm_raw, "heat_capacity"),
        thermal_conductivity=_get(therm_raw, "thermal_conductivity"),
        expansion_coefficient=_get(therm_raw, "expansion_coefficient"),
    )

    meta = _get(raw, "metadata", {}) or {}

    trap_states = [
        _parse_trap_state(ts)
        for ts in (_get(raw, "trap_states") or [])
    ]

    return MaterialEntry(
        material_name=name,
        nk_reference=nk_ref,
        homo=_get(raw, "homo"),
        lumo=_get(raw, "lumo"),
        electron_mobility=_get(raw, "electron_mobility"),
        hole_mobility=_get(raw, "hole_mobility"),
        dielectric_constant=_get(raw, "dielectric_constant"),
        electrode=electrode,
        thermal=thermal,
        notes=_get(meta, "notes", ""),
        trap_states=trap_states,
    )


def _parse_trap_state(raw: dict) -> TrapState:
    label = "material.trap_states"
    return TrapState(
        density_m3=float(_require(raw, "density_m3", label)),
        energy_from_lumo_eV=float(_require(raw, "energy_from_lumo_eV", label)),
        sigma_n=float(_require(raw, "sigma_n", label)),
        sigma_p=float(_require(raw, "sigma_p", label)),
        vth_n=float(_get(raw, "vth_n", 1e5)),
        vth_p=float(_get(raw, "vth_p", 1e5)),
        label=str(_get(raw, "label", "trap")),
    )


def load_material_db(path: str | Path) -> MaterialDB:
    """Load and validate a material-db YAML. Returns a :class:`MaterialDB`."""
    path = Path(path)
    raw = _load_yaml(path)

    if raw.get("format") != "material-db":
        raise ValueError(
            f"[material_db] Expected format 'material-db', got '{raw.get('format')}' in {path}"
        )

    materials_list: list[dict] = raw.get("materials") or []
    if not materials_list:
        raise ValueError(f"[material_db] File '{path}' contains no materials")

    db: dict[str, MaterialEntry] = {}
    for entry in materials_list:
        mat = _parse_material_entry(entry)
        if mat.material_name in db:
            raise ValueError(f"[material_db] Duplicate material_name '{mat.material_name}' in {path}")
        db[mat.material_name] = mat

    return MaterialDB(version=raw.get("version"), materials=db)


# ---------------------------------------------------------------------------
# Device Stack
# ---------------------------------------------------------------------------

def _parse_layer(raw: dict) -> Layer:
    label = "device_stack.layer"
    order = int(_require(raw, "order", label))
    role = _require(raw, "layer_role", label)
    if role not in _VALID_LAYER_ROLES:
        raise ValueError(
            f"[{label}] Unknown layer_role '{role}' (order={order}). "
            f"Valid: {sorted(_VALID_LAYER_ROLES)}"
        )

    elec_raw = _get(raw, "electrode")
    electrode = None
    if elec_raw is not None:
        electrode = ElectrodeLayer(
            is_electrode=bool(_get(elec_raw, "is_electrode", False)),
            type=_get(elec_raw, "type"),
            work_function=_get(elec_raw, "work_function"),
        )

    emit_raw = _get(raw, "emitter")
    emitter = None
    if emit_raw is not None:
        emitter = EmitterParams(
            dopant=_require(emit_raw, "dopant", f"{label}.emitter"),
            dopant_fraction=_require(emit_raw, "dopant_fraction", f"{label}.emitter"),
        )

    return Layer(
        order=order,
        layer_name=_require(raw, "layer_name", label),
        material_name=_require(raw, "material_name", label),
        thickness_nm=float(_require(raw, "thickness_nm", label)),
        layer_role=role,
        electrode=electrode,
        emitter=emitter,
    )


def load_device_stack(path: str | Path) -> DeviceStack:
    """Load and validate a device-stack YAML. Returns a :class:`DeviceStack`."""
    path = Path(path)
    raw = _load_yaml(path)

    stack_raw = _require(raw, "stack", "device_stack")
    layers_raw: list[dict] = _require(stack_raw, "layers", "device_stack.stack")

    if not layers_raw:
        raise ValueError(f"[device_stack] No layers defined in {path}")

    layers = sorted([_parse_layer(l) for l in layers_raw], key=lambda l: l.order)

    return DeviceStack(
        version=raw.get("version"),
        name=_require(stack_raw, "name", "device_stack.stack"),
        substrate=_get(stack_raw, "substrate", "unknown"),
        layers=layers,
    )


# ---------------------------------------------------------------------------
# Solver Config
# ---------------------------------------------------------------------------

def load_solver_config(path: str | Path) -> SolverConfig:
    """Load and validate a solver-config YAML. Returns a :class:`SolverConfig`."""
    path = Path(path)
    raw = _load_yaml(path)
    sol = _require(raw, "solver", "solver_config")

    sim_type = _require(sol, "simulation_type", "solver_config")
    if sim_type not in _VALID_SIMULATION_TYPES:
        raise ValueError(
            f"[solver_config] Unknown simulation_type '{sim_type}'. "
            f"Valid: {sorted(_VALID_SIMULATION_TYPES)}"
        )

    # wavelength_grid
    wg_raw = _require(sol, "wavelength_grid", "solver_config")
    wavelength_grid = WavelengthGrid(
        start_nm=float(_require(wg_raw, "start_nm", "solver_config.wavelength_grid")),
        end_nm=float(_require(wg_raw, "end_nm", "solver_config.wavelength_grid")),
        step_nm=float(_require(wg_raw, "step_nm", "solver_config.wavelength_grid")),
    )
    if wavelength_grid.step_nm <= 0:
        raise ValueError("[solver_config.wavelength_grid] step_nm must be > 0")
    if wavelength_grid.start_nm >= wavelength_grid.end_nm:
        raise ValueError("[solver_config.wavelength_grid] start_nm must be < end_nm")

    # bias_sweep
    bs_raw = _require(sol, "bias_sweep", "solver_config")
    bias_sweep = BiasSweep(
        start_V=float(_require(bs_raw, "start_V", "solver_config.bias_sweep")),
        end_V=float(_require(bs_raw, "end_V", "solver_config.bias_sweep")),
        step_V=float(_require(bs_raw, "step_V", "solver_config.bias_sweep")),
    )
    if bias_sweep.step_V <= 0:
        raise ValueError("[solver_config.bias_sweep] step_V must be > 0")

    # convergence
    cv_raw = _require(sol, "convergence", "solver_config")
    criterion = _require(cv_raw, "criterion", "solver_config.convergence")
    if criterion not in _VALID_CONVERGENCE_CRITERIA:
        raise ValueError(
            f"[solver_config.convergence] Unknown criterion '{criterion}'. "
            f"Valid: {sorted(_VALID_CONVERGENCE_CRITERIA)}"
        )
    norm = _require(cv_raw, "norm", "solver_config.convergence")
    if norm not in _VALID_NORMS:
        raise ValueError(
            f"[solver_config.convergence] Unknown norm '{norm}'. Valid: {sorted(_VALID_NORMS)}"
        )
    convergence = ConvergenceCriteria(
        tolerance=float(_require(cv_raw, "tolerance", "solver_config.convergence")),
        criterion=criterion,
        norm=norm,
    )

    # damping
    damp_raw = _require(sol, "damping", "solver_config")
    factor = float(_require(damp_raw, "factor", "solver_config.damping"))
    if not (0 < factor <= 1):
        raise ValueError("[solver_config.damping] factor must be in (0, 1]")
    damping = DampingConfig(
        enabled=bool(_require(damp_raw, "enabled", "solver_config.damping")),
        factor=factor,
        adaptive=bool(_require(damp_raw, "adaptive", "solver_config.damping")),
    )

    # mesh
    mesh_raw = _require(sol, "mesh", "solver_config")
    mesh = MeshConfig(
        resolution_nm=float(_require(mesh_raw, "resolution_nm", "solver_config.mesh"))
    )

    # physics
    ph_raw = _require(sol, "physics", "solver_config")
    physics = PhysicsFlags(
        include_thermal=bool(_get(ph_raw, "include_thermal", False)),
        include_excitonics=bool(_get(ph_raw, "include_excitonics", False)),
        include_charge_transport=bool(_get(ph_raw, "include_charge_transport", False)),
    )

    # boundary_conditions
    bc_raw = _require(sol, "boundary_conditions", "solver_config")
    boundary = BoundaryConditions(
        anode_potential_eV=float(_require(bc_raw["anode"], "potential_eV", "solver_config.boundary_conditions.anode")),
        cathode_potential_eV=float(_require(bc_raw["cathode"], "potential_eV", "solver_config.boundary_conditions.cathode")),
    )

    # output
    out_raw = _get(sol, "output", {})
    output = OutputConfig(
        save_fields=bool(_get(out_raw, "save_fields", False)),
        output_folder=_get(out_raw, "output_folder", "./outputs"),
    )

    return SolverConfig(
        version=raw.get("version"),
        simulation_name=_require(sol, "simulation_name", "solver_config"),
        simulation_type=sim_type,
        max_iterations=int(_require(sol, "max_iterations", "solver_config")),
        wavelength_grid=wavelength_grid,
        bias_sweep=bias_sweep,
        convergence=convergence,
        damping=damping,
        mesh=mesh,
        physics=physics,
        boundary_conditions=boundary,
        output=output,
        metadata=_get(sol, "metadata", {}),
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def load_measurement(path: str | Path, load_data: bool = True) -> Measurement:
    """Load a measurement YAML and optionally its referenced CSV.

    Set *load_data=False* to skip CSV loading when the file is not yet available.
    """
    path = Path(path)
    raw = _load_yaml(path)
    meas = _require(raw, "measurement", "measurement")

    mtype = _require(meas, "type", "measurement")
    if mtype not in _VALID_MEASUREMENT_TYPES:
        raise ValueError(
            f"[measurement] Unknown type '{mtype}'. Valid: {sorted(_VALID_MEASUREMENT_TYPES)}"
        )

    units_raw = _get(meas, "units", {})
    units = MeasurementUnits(
        voltage=_get(units_raw, "voltage", "V"),
        current_density=_get(units_raw, "current_density", "mA/cm2"),
        luminance=_get(units_raw, "luminance"),
        spectral_quantity=_get(units_raw, "spectral_quantity"),
    )

    data: list[dict[str, str]] | None = None
    data_file = _get(meas, "data_file")
    if load_data and data_file:
        csv_path = _resolve(Path(data_file), path.parent)
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                data = list(csv.DictReader(f))

    return Measurement(
        version=raw.get("version"),
        name=_require(meas, "name", "measurement"),
        type=mtype,
        data_file=data_file,
        units=units,
        temperature_K=_get(meas, "temperature_K"),
        scan_direction=_get(meas, "scan_direction"),
        comments=_get(meas, "comments", ""),
        data=data,
    )


# ---------------------------------------------------------------------------
# Project (root aggregator)
# ---------------------------------------------------------------------------

def load_project(path: str | Path) -> Project:
    """Load a root project YAML and resolve all referenced sub-files.

    Returns a fully populated :class:`Project`.
    """
    path = Path(path)
    raw = _load_yaml(path)
    proj = _require(raw, "project", "oled_input")
    base = path.parent

    def rel(key: str) -> Path:
        return _resolve(Path(_require(proj, key, f"project.{key}")), base)

    return Project(
        name=_require(proj, "name", "project"),
        description=_get(proj, "description", ""),
        material_db=load_material_db(rel("material_database")),
        device_stack=load_device_stack(rel("device_stack")),
        measurement=load_measurement(rel("measurement"), load_data=False),
        solver_config=load_solver_config(rel("solver_config")),
    )


# ---------------------------------------------------------------------------
# Emitter Config
# ---------------------------------------------------------------------------

_VALID_Z_DISTRIBUTIONS = {"center", "uniform", "front", "back"}
_VALID_EMITTER_TYPES    = {"phosphorescent", "fluorescent"}


def load_emitter_config(path: str | Path) -> EmitterConfig:
    """Load and validate an emitter specification YAML.

    The ``pl_spectrum_file`` field is resolved relative to the directory
    that contains the YAML, falling back to the project root two levels up.
    """
    path = Path(path)
    raw  = _load_yaml(path)
    base = path.parent

    _require(raw, "layer_name",       "emitter_config")
    _require(raw, "pl_spectrum_file", "emitter_config")

    z_dist = str(_get(raw, "z_distribution", "center"))
    if z_dist not in _VALID_Z_DISTRIBUTIONS:
        raise ValueError(
            f"[emitter_config] z_distribution must be one of {sorted(_VALID_Z_DISTRIBUTIONS)}, "
            f"got {z_dist!r}"
        )

    h_frac = float(_get(raw, "horizontal_fraction", 1.0))
    if not 0.0 <= h_frac <= 1.0:
        raise ValueError(
            f"[emitter_config] horizontal_fraction must be in [0, 1], got {h_frac}"
        )

    emitter_type = str(_get(raw, "emitter_type", "phosphorescent"))
    if emitter_type not in _VALID_EMITTER_TYPES:
        raise ValueError(
            f"[emitter_config] emitter_type must be one of {sorted(_VALID_EMITTER_TYPES)}, "
            f"got {emitter_type!r}"
        )

    # Resolve pl_spectrum_file relative to YAML dir, then to project root
    pl_raw  = raw["pl_spectrum_file"]
    pl_path = Path(pl_raw)
    if not pl_path.is_absolute():
        candidate = (base / pl_path).resolve()
        if not candidate.exists():
            candidate = (base.parent.parent / pl_path).resolve()
        pl_path = candidate

    return EmitterConfig(
        layer_name=str(raw["layer_name"]),
        material=str(_get(raw, "material", "")),
        pl_spectrum_file=str(pl_path),
        horizontal_fraction=h_frac,
        z_distribution=z_dist,
        emitter_type=emitter_type,
        notes=str(_get(raw, "notes", "")),
    )
