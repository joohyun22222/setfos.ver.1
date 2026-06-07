"""Optical emission solver — PL × |E|² weighting with pluggable outcoupling.

Design
------
:class:`EmissionSolver` orchestrates three steps:

  1. **Field profile** (re-uses :func:`compute_field_profile` from Step 5)
     to obtain |E(z, λ)|² across the full device at 1 nm resolution.

  2. **Emission weighting** — extracts |E|² at the emitter position and
     multiplies by the interpolated PL spectrum.

  3. **Outcoupling** — passes a :class:`StackContext` to whatever
     :class:`~src.emission.outcoupling.OutcouplingBase` model is attached
     (default: :class:`~src.emission.outcoupling.NullOutcoupling`).

To plug in a new outcoupling model (e.g. Purcell-factor calculation in Step 7)::

    from src.emission.solver import EmissionSolver
    from my_module import PurcellOutcoupling

    solver = EmissionSolver(nk_root=..., outcoupling=PurcellOutcoupling())
    result = solver.solve(stack, material_db, solver_cfg, emitter_cfg)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..io.models import DeviceStack, EmitterConfig, MaterialDB, SolverConfig
from ..optics.field_profile import FieldProfileResult, compute_field_profile
from ..optics.nk_data import NKDataProvider
from .models import EmissionResult, EmitterProfile, OutcouplingResult, StackContext
from .outcoupling import NullOutcoupling, OutcouplingBase


class EmissionSolver:
    """Optical emission solver using field-profile-weighted PL spectra.

    Parameters
    ----------
    nk_root      : directory containing material n/k CSV files
    project_root : project root directory used to resolve relative paths
                   in :class:`EmitterConfig` (defaults to current directory)
    outcoupling  : an :class:`OutcouplingBase` instance
                   (default: :class:`NullOutcoupling`)
    n_inc, n_sub : refractive indices of incident medium and substrate
    """

    def __init__(
        self,
        nk_root: str | Path,
        project_root: str | Path | None = None,
        outcoupling: OutcouplingBase | None = None,
        n_inc: float = 1.0,
        n_sub: float = 1.5,
    ):
        self._nk = NKDataProvider(Path(nk_root))
        self._root = Path(project_root) if project_root else Path.cwd()
        self._oc: OutcouplingBase = outcoupling if outcoupling is not None else NullOutcoupling()
        self.n_inc = n_inc
        self.n_sub = n_sub

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        stack: DeviceStack,
        material_db: MaterialDB,
        solver_cfg: SolverConfig,
        emitter_cfg: EmitterConfig,
        z_resolution_nm: float = 1.0,
    ) -> EmissionResult:
        """Compute the optical emission spectrum for a given OLED device.

        Parameters
        ----------
        stack, material_db, solver_cfg : standard solver inputs
        emitter_cfg   : emitter specification (layer name, PL file, orientation)
        z_resolution_nm : spatial pitch for |E|² profile sampling (nm)

        Returns
        -------
        EmissionResult containing:
        - ``weighted_spectrum`` — PL × |E|², area-normalised
        - ``emission_spectrum`` — additionally modulated by η_out(λ)
        - ``outcoupling``       — :class:`OutcouplingResult` from the attached model
        """
        wg = solver_cfg.wavelength_grid
        wavelengths = np.arange(
            wg.start_nm, wg.end_nm + wg.step_nm * 0.5, wg.step_nm
        )

        # ── 1. Build optical data ──────────────────────────────────────
        n_layers, d_layers, layer_names, material_names = (
            self._build_optical_data(stack, material_db, wavelengths)
        )

        # ── 2. Field profile ───────────────────────────────────────────
        fp = compute_field_profile(
            n_layers, d_layers, wavelengths,
            layer_names=layer_names,
            material_names=material_names,
            n_inc=self.n_inc,
            n_sub=self.n_sub,
            z_resolution_nm=z_resolution_nm,
        )

        # ── 3. Locate emitter layer ────────────────────────────────────
        eml_idx = self._find_emitter_index(stack, emitter_cfg.layer_name)
        z_start = fp.layer_boundaries_nm[eml_idx]
        z_end   = fp.layer_boundaries_nm[eml_idx + 1]

        # ── 4. |E|² at emitter position ───────────────────────────────
        E_sq, z_em = self._emitter_intensity(
            fp, emitter_cfg.z_distribution, z_start, z_end
        )

        # ── 5. Load and interpolate PL spectrum ───────────────────────
        emitter_profile = self._load_emitter(emitter_cfg, wavelengths)
        pl = emitter_profile.pl_at(wavelengths)

        # ── 6. Weighted emission (PL × |E|²) ──────────────────────────
        raw   = E_sq * pl
        total = raw.sum()
        weighted = raw / total if total > 0 else raw

        # ── 7. Outcoupling ────────────────────────────────────────────
        ctx = StackContext(
            n_layers=n_layers,
            d_layers_nm=d_layers,
            wavelengths_nm=wavelengths,
            z_emitter_nm=z_em,
            horizontal_fraction=emitter_cfg.horizontal_fraction,
            n_inc=self.n_inc,
            n_sub=self.n_sub,
            field_profile=fp,
        )
        oc_result = self._oc.compute(ctx)

        return EmissionResult(
            wavelength_nm=wavelengths,
            z_emitter_nm=z_em,
            layer_name=emitter_cfg.layer_name,
            pl_spectrum=pl,
            E_squared=E_sq,
            weighted_spectrum=weighted,
            outcoupling=oc_result,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_optical_data(
        self,
        stack: DeviceStack,
        material_db: MaterialDB,
        wavelengths: np.ndarray,
    ) -> tuple[list, list, list, list]:
        n_layers, d_layers, lnames, mnames = [], [], [], []
        for layer in stack.layers:
            mat = material_db.get(layer.material_name)
            n_layers.append(self._nk.get_nk(mat, wavelengths))
            d_layers.append(layer.thickness_nm)
            lnames.append(layer.layer_name)
            mnames.append(layer.material_name)
        return n_layers, d_layers, lnames, mnames

    def _find_emitter_index(self, stack: DeviceStack, layer_name: str) -> int:
        """Return the 0-based index of the emitter layer in the stack."""
        for i, layer in enumerate(stack.layers):
            if layer.layer_name == layer_name:
                return i
        # Fallback: first layer with role="emitter"
        for i, layer in enumerate(stack.layers):
            if layer.is_emitter():
                return i
        raise ValueError(
            f"Emitter layer '{layer_name}' not found in stack. "
            f"Available layers: {[l.layer_name for l in stack.layers]}"
        )

    def _emitter_intensity(
        self,
        fp: FieldProfileResult,
        z_distribution: str,
        z_start: float,
        z_end: float,
    ) -> tuple[np.ndarray, float]:
        """Return (|E|²(λ), representative z_nm) for the given distribution."""
        if z_distribution == "center":
            z_em = 0.5 * (z_start + z_end)
            return fp.get_intensity_at_z(z_em), z_em

        if z_distribution == "front":
            z_em = z_start + min(5.0, (z_end - z_start) * 0.1)
            return fp.get_intensity_at_z(z_em), z_em

        if z_distribution == "back":
            z_em = z_end - min(5.0, (z_end - z_start) * 0.1)
            return fp.get_intensity_at_z(z_em), z_em

        if z_distribution == "uniform":
            mask = (fp.z_nm >= z_start) & (fp.z_nm <= z_end)
            if not np.any(mask):
                z_em = 0.5 * (z_start + z_end)
                return fp.get_intensity_at_z(z_em), z_em
            return fp.E_squared[mask, :].mean(axis=0), float(0.5 * (z_start + z_end))

        raise ValueError(
            f"Unknown z_distribution '{z_distribution}'. "
            "Valid options: 'center', 'front', 'back', 'uniform'."
        )

    def _load_emitter(
        self,
        emitter_cfg: EmitterConfig,
        wavelengths: np.ndarray,
    ) -> EmitterProfile:
        """Load PL spectrum CSV and build an :class:`EmitterProfile`."""
        pl_path = Path(emitter_cfg.pl_spectrum_file)
        if not pl_path.is_absolute():
            pl_path = (self._root / pl_path).resolve()

        wl_pl, intensity = [], []
        with open(pl_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wl_pl.append(float(row["wavelength_nm"]))
                intensity.append(float(row["intensity"]))

        wl_arr  = np.array(wl_pl, dtype=float)
        int_arr = np.array(intensity, dtype=float)
        peak = int_arr.max()
        if peak > 0:
            int_arr = int_arr / peak

        return EmitterProfile(
            layer_name=emitter_cfg.layer_name,
            material=emitter_cfg.material,
            wavelength_nm=wl_arr,
            pl_spectrum=int_arr,
            horizontal_fraction=emitter_cfg.horizontal_fraction,
            z_distribution=emitter_cfg.z_distribution,
            emitter_type=emitter_cfg.emitter_type,
        )
