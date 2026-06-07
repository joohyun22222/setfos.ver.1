"""High-level RTA solver integrating DeviceStack, MaterialDB, and SolverConfig."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..io.models import DeviceStack, MaterialDB, SolverConfig
from .field_profile import FieldProfileResult, LayerAbsorption, compute_field_profile
from .nk_data import NKDataProvider
from .tmm import compute_rta


@dataclass
class RTAResult:
    """Output of the RTA solver for one device stack."""

    wavelength_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    A: np.ndarray

    # -------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------

    @property
    def rta_sum(self) -> np.ndarray:
        """R + T + A — should equal 1.0 everywhere (energy conservation)."""
        return self.R + self.T + self.A

    @property
    def max_conservation_error(self) -> float:
        """Max |R+T+A − 1|. < 1e-10 indicates good numerical precision."""
        return float(np.max(np.abs(self.rta_sum - 1.0)))

    def peak_absorption_nm(self) -> float:
        """Wavelength at which total absorptance is maximum."""
        return float(self.wavelength_nm[np.argmax(self.A)])

    # -------------------------------------------------------------------
    # I/O helpers
    # -------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save wavelength, R, T, A columns to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "wavelength_nm,R,T,A,RTA_sum\n"
        rows = "\n".join(
            f"{w:.2f},{r:.8f},{t:.8f},{a:.8f},{s:.8f}"
            for w, r, t, a, s in zip(
                self.wavelength_nm, self.R, self.T, self.A, self.rta_sum
            )
        )
        path.write_text(header + rows, encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Save result as JSON (wavelength array + R/T/A arrays)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wavelength_nm": self.wavelength_nm.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "A": self.A.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        """One-line console summary."""
        avg_R = float(np.mean(self.R))
        avg_T = float(np.mean(self.T))
        avg_A = float(np.mean(self.A))
        err   = self.max_conservation_error
        peak  = self.peak_absorption_nm()
        return (
            f"RTA summary | λ={self.wavelength_nm[0]:.0f}–{self.wavelength_nm[-1]:.0f} nm "
            f"| <R>={avg_R:.3f}  <T>={avg_T:.3f}  <A>={avg_A:.3f} "
            f"| peak_A @ {peak:.0f} nm "
            f"| max|ΔE|={err:.2e}"
        )


class RTASolver:
    """Multilayer optical RTA solver using the Transfer Matrix Method.

    Parameters
    ----------
    nk_root : path to directory containing material n/k CSV files.
    n_inc   : refractive index of the incident medium (default: 1.0 = air).
    n_sub   : refractive index of the substrate / exit medium (default: 1.5 = glass).
    """

    def __init__(
        self,
        nk_root: str | Path,
        n_inc: float = 1.0,
        n_sub: float = 1.5,
    ):
        self._nk = NKDataProvider(Path(nk_root))
        self.n_inc = n_inc
        self.n_sub = n_sub

    def solve(
        self,
        stack: DeviceStack,
        material_db: MaterialDB,
        solver_cfg: SolverConfig,
    ) -> RTAResult:
        """Run a full wavelength sweep and return :class:`RTAResult`.

        Parameters
        ----------
        stack        : device layer stack
        material_db  : material database with n/k references
        solver_cfg   : solver configuration (supplies wavelength grid)
        """
        wg = solver_cfg.wavelength_grid
        wavelengths = np.arange(wg.start_nm, wg.end_nm + wg.step_nm * 0.5, wg.step_nm)

        n_layers: list[np.ndarray] = []
        d_layers: list[float] = []

        for layer in stack.layers:
            mat = material_db.get(layer.material_name)
            n_cplx = self._nk.get_nk(mat, wavelengths)
            n_layers.append(n_cplx)
            d_layers.append(layer.thickness_nm)

        R, T, A = compute_rta(
            n_layers, d_layers, wavelengths,
            n_inc=self.n_inc,
            n_sub=self.n_sub,
        )

        return RTAResult(wavelength_nm=wavelengths, R=R, T=T, A=A)

    def solve_with_profile(
        self,
        stack: DeviceStack,
        material_db: MaterialDB,
        solver_cfg: SolverConfig,
        z_resolution_nm: float = 1.0,
    ) -> tuple[RTAResult, FieldProfileResult]:
        """Run full wavelength sweep and return both RTA and field profile.

        Parameters
        ----------
        stack, material_db, solver_cfg : same as :meth:`solve`
        z_resolution_nm : spatial sampling pitch for the field profile (nm)

        Returns
        -------
        rta_result    : RTAResult
        field_result  : FieldProfileResult with layer-resolved absorptance and |E|²(z,λ)
        """
        wg = solver_cfg.wavelength_grid
        wavelengths = np.arange(wg.start_nm, wg.end_nm + wg.step_nm * 0.5, wg.step_nm)

        n_layers: list[np.ndarray] = []
        d_layers: list[float] = []
        layer_names: list[str] = []
        material_names: list[str] = []

        for layer in stack.layers:
            mat = material_db.get(layer.material_name)
            n_layers.append(self._nk.get_nk(mat, wavelengths))
            d_layers.append(layer.thickness_nm)
            layer_names.append(layer.layer_name)
            material_names.append(layer.material_name)

        R, T, A = compute_rta(
            n_layers, d_layers, wavelengths,
            n_inc=self.n_inc,
            n_sub=self.n_sub,
        )
        rta = RTAResult(wavelength_nm=wavelengths, R=R, T=T, A=A)

        fp = compute_field_profile(
            n_layers, d_layers, wavelengths,
            layer_names=layer_names,
            material_names=material_names,
            n_inc=self.n_inc,
            n_sub=self.n_sub,
            z_resolution_nm=z_resolution_nm,
        )
        return rta, fp

    def solve_project(self, project) -> RTAResult:
        """Convenience overload that accepts a :class:`~src.io.models.Project`."""
        return self.solve(
            project.device_stack,
            project.material_db,
            project.solver_config,
        )
