"""1D mesh for drift-diffusion electrical simulation.

Builds a uniform node grid from a :class:`~src.io.models.DeviceStack`,
assigning layer-derived material properties to each node.

Physical convention
-------------------
- z = 0 at the anode (left); z = total_nm at the cathode (right)
- Nodes at z = 0, dz, 2·dz, …, total_nm
- Edges connect adjacent nodes; edge i connects node i to node i+1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.models import DeviceStack, MaterialDB, MaterialEntry, Layer

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
Q    = 1.602176634e-19    # elementary charge [C]
EPS0 = 8.854187817e-12   # vacuum permittivity [F/m]
KB   = 1.380649e-23      # Boltzmann constant [J/K]
T0   = 300.0             # reference temperature [K]
KT_EV = KB * T0 / Q     # thermal voltage at 300 K ≈ 0.02585 eV
NM_TO_M = 1e-9

# ---------------------------------------------------------------------------
# Defaults for optional material parameters
# ---------------------------------------------------------------------------
_DEFAULT_NC_NV = 1e27   # m⁻³  — representative effective DOS for organics
_DEFAULT_TAU   = 1e-6   # s    — default SRH carrier lifetime
_DEFAULT_MU    = 1e-10  # m²/Vs — near-zero fallback (undefined material)


# ---------------------------------------------------------------------------
# Per-node material properties
# ---------------------------------------------------------------------------

@dataclass
class NodeProps:
    """Electrical material properties at a single mesh node.

    Derived from :class:`~src.io.models.MaterialEntry`:

    Organics:
        chi_eV = −LUMO (electron affinity)
        Eg_eV  = LUMO − HOMO (bandgap, always positive)

    Electrodes:
        chi_eV = work_function, Eg_eV = 0 (degenerate metal)
    """
    eps_r: float            # relative permittivity
    chi_eV: float           # electron affinity [eV]
    Eg_eV: float            # transport bandgap [eV]
    Nc: float               # effective DOS — conduction band [m⁻³]
    Nv: float               # effective DOS — valence band [m⁻³]
    mu_n: float             # electron mobility [m²/V/s]
    mu_p: float             # hole mobility [m²/V/s]
    tau_n: float            # electron SRH lifetime [s]
    tau_p: float            # hole SRH lifetime [s]
    N_doping: float         # net doping N_D − N_A [m⁻³]; > 0 → n-type
    is_electrode: bool      # True for metal contacts
    work_function_eV: float # contact Fermi level [eV]; 0 for organics


# ---------------------------------------------------------------------------
# 1D mesh
# ---------------------------------------------------------------------------

@dataclass
class Mesh1D:
    """Uniform 1D node mesh for a multilayer device.

    Arrays
    ------
    z_nm               : (N,)   node positions [nm]
    z_m                : (N,)   node positions [m]
    dz_m               : (N−1,) edge spacings [m]
    node_layer         : (N,)   integer layer index per node
    layer_names        : list of layer name strings
    layer_boundaries_nm: (L+1,) interface z-coordinates [nm]
    node_props         : list of :class:`NodeProps`, length N
    """

    z_nm: np.ndarray
    z_m: np.ndarray
    dz_m: np.ndarray
    node_layer: np.ndarray       # int, (N,)
    layer_names: list[str]
    layer_boundaries_nm: np.ndarray
    node_props: list[NodeProps]

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.z_nm)

    @property
    def num_edges(self) -> int:
        return len(self.dz_m)

    # ------------------------------------------------------------------
    # Per-node property arrays (vectorised access)
    # ------------------------------------------------------------------

    def eps_array(self) -> np.ndarray:
        """ε_r at each node, shape (N,)."""
        return np.array([p.eps_r for p in self.node_props])

    def N_doping_array(self) -> np.ndarray:
        """Net doping N_D − N_A [m⁻³], shape (N,)."""
        return np.array([p.N_doping for p in self.node_props])

    def mu_n_array(self) -> np.ndarray:
        """Electron mobility [m²/Vs], shape (N,)."""
        return np.array([p.mu_n for p in self.node_props])

    def mu_p_array(self) -> np.ndarray:
        """Hole mobility [m²/Vs], shape (N,)."""
        return np.array([p.mu_p for p in self.node_props])

    def Nc_array(self) -> np.ndarray:
        """Effective conduction-band DOS [m⁻³], shape (N,)."""
        return np.array([p.Nc for p in self.node_props])

    def Nv_array(self) -> np.ndarray:
        """Effective valence-band DOS [m⁻³], shape (N,)."""
        return np.array([p.Nv for p in self.node_props])

    # ------------------------------------------------------------------
    # Layer lookup helpers
    # ------------------------------------------------------------------

    def layer_mask(self, layer_name: str) -> np.ndarray:
        """Boolean (N,) mask for nodes belonging to *layer_name*."""
        try:
            idx = self.layer_names.index(layer_name)
        except ValueError:
            raise ValueError(
                f"Layer '{layer_name}' not found. Available: {self.layer_names}"
            )
        return self.node_layer == idx

    def layer_slice(self, layer_name: str) -> slice:
        """Contiguous index slice [start:stop] for *layer_name*'s nodes."""
        mask = self.layer_mask(layer_name)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No nodes assigned to layer '{layer_name}'")
        return slice(int(indices[0]), int(indices[-1]) + 1)

    def layer_center_z_nm(self, layer_name: str) -> float:
        """Z coordinate at the geometric centre of *layer_name* [nm]."""
        idx = self.layer_names.index(layer_name)
        return float(0.5 * (self.layer_boundaries_nm[idx] + self.layer_boundaries_nm[idx + 1]))


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def build_mesh(
    stack: DeviceStack,
    material_db: MaterialDB,
    z_resolution_nm: float = 1.0,
) -> Mesh1D:
    """Build a uniform 1D mesh from a device stack.

    Parameters
    ----------
    stack            : ordered layer description
    material_db      : material property database
    z_resolution_nm  : node spacing [nm] (default 1.0)

    Returns
    -------
    :class:`Mesh1D` with per-node material properties.
    """
    layers = stack.layers  # sorted by order

    # ── node positions ──────────────────────────────────────────────────
    total_nm = stack.total_thickness_nm()
    z_nm = np.arange(0.0, total_nm + z_resolution_nm * 0.5, z_resolution_nm)
    N    = len(z_nm)
    z_m  = z_nm * NM_TO_M
    dz_m = np.diff(z_m)

    # ── layer boundaries ─────────────────────────────────────────────────
    boundaries = [0.0]
    for lyr in layers:
        boundaries.append(boundaries[-1] + lyr.thickness_nm)
    boundaries_nm = np.array(boundaries)

    # ── layer index per node ─────────────────────────────────────────────
    node_layer = np.full(N, len(layers) - 1, dtype=int)
    for i, z in enumerate(z_nm):
        for j in range(len(layers)):
            if boundaries_nm[j] <= z < boundaries_nm[j + 1]:
                node_layer[i] = j
                break

    # ── per-node properties ───────────────────────────────────────────────
    node_props: list[NodeProps] = []
    for i in range(N):
        mat = material_db.get(layers[node_layer[i]].material_name)
        node_props.append(_mat_to_node_props(mat))

    return Mesh1D(
        z_nm=z_nm,
        z_m=z_m,
        dz_m=dz_m,
        node_layer=node_layer,
        layer_names=[lyr.layer_name for lyr in layers],
        layer_boundaries_nm=boundaries_nm,
        node_props=node_props,
    )


def _mat_to_node_props(mat: MaterialEntry) -> NodeProps:
    """Derive :class:`NodeProps` from a :class:`~src.io.models.MaterialEntry`."""
    is_el = mat.is_electrode()

    if is_el:
        wf  = mat.electrode.work_function if mat.electrode else 4.5
        eps = float(mat.dielectric_constant) if mat.dielectric_constant else 10.0
        return NodeProps(
            eps_r=eps,
            chi_eV=wf,
            Eg_eV=0.0,
            Nc=_DEFAULT_NC_NV,
            Nv=_DEFAULT_NC_NV,
            mu_n=1.0,           # metal: high carrier transport
            mu_p=1.0,
            tau_n=_DEFAULT_TAU,
            tau_p=_DEFAULT_TAU,
            N_doping=0.0,
            is_electrode=True,
            work_function_eV=wf,
        )

    # Organic / semiconductor
    homo = mat.homo   # eV (negative, below vacuum)
    lumo = mat.lumo   # eV (negative, below vacuum)
    if homo is not None and lumo is not None:
        chi_eV = -lumo          # electron affinity = −LUMO
        Eg_eV  = lumo - homo    # bandgap = LUMO − HOMO > 0
    else:
        chi_eV, Eg_eV = 2.0, 3.0   # generic organic defaults

    # Mobilities: YAML values in cm²/V/s → convert to m²/V/s (×10⁻⁴)
    mu_n = float(mat.electron_mobility) * 1e-4 if mat.electron_mobility else _DEFAULT_MU
    mu_p = float(mat.hole_mobility)     * 1e-4 if mat.hole_mobility     else _DEFAULT_MU

    eps_r = float(mat.dielectric_constant) if mat.dielectric_constant else 3.0

    return NodeProps(
        eps_r=eps_r,
        chi_eV=chi_eV,
        Eg_eV=Eg_eV,
        Nc=_DEFAULT_NC_NV,
        Nv=_DEFAULT_NC_NV,
        mu_n=mu_n,
        mu_p=mu_p,
        tau_n=_DEFAULT_TAU,
        tau_p=_DEFAULT_TAU,
        N_doping=0.0,
        is_electrode=False,
        work_function_eV=0.0,
    )
