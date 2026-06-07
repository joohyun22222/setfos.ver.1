"""n/k optical constant reader with wavelength interpolation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..io.models import MaterialEntry


class NKDataProvider:
    """Loads and interpolates complex refractive index data for materials.

    CSV format expected: columns [wavelength_nm, n, k] with a header row.
    """

    def __init__(self, nk_root: Path):
        self._root = Path(nk_root)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_nk(self, material: MaterialEntry, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Return complex refractive index n + ik at each wavelength.

        Falls back to n=1.5, k=0 when no n/k file is available.
        """
        key = material.material_name
        if key not in self._cache:
            self._cache[key] = self._load(material)

        wl_ref, n_ref, k_ref = self._cache[key]
        n_interp = np.interp(wavelengths_nm, wl_ref, n_ref,
                             left=n_ref[0], right=n_ref[-1])
        k_interp = np.interp(wavelengths_nm, wl_ref, k_ref,
                             left=k_ref[0], right=k_ref[-1])
        return n_interp + 1j * k_interp

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, material: MaterialEntry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Try to load from CSV; fall back to constant n=1.5, k=0."""
        ref = material.nk_reference
        if ref:
            # nk_reference may be an absolute path or relative to project root
            candidate = Path(ref)
            if not candidate.is_absolute():
                candidate = self._root / candidate.name  # strip subdirectory, look in nk_root
            if candidate.exists():
                return _read_nk_csv(candidate)

            # Try resolve from project root (path stored as "data/nk/X.csv")
            candidate2 = self._root.parent.parent / ref
            if candidate2.exists():
                return _read_nk_csv(candidate2)

        name = material.material_name
        print(f"  [NKDataProvider] WARNING: no n/k file for '{name}', using n=1.5 k=0")
        wl_dummy = np.array([300.0, 1000.0])
        return wl_dummy, np.array([1.5, 1.5]), np.array([0.0, 0.0])


def _read_nk_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a CSV with columns [wavelength_nm, n, k]."""
    wl, n, k = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl.append(float(row["wavelength_nm"]))
            n.append(float(row["n"]))
            k.append(float(row["k"]))
    wl_arr = np.array(wl)
    n_arr  = np.array(n)
    k_arr  = np.array(k)
    # Ensure wavelengths are sorted ascending
    order = np.argsort(wl_arr)
    return wl_arr[order], n_arr[order], np.maximum(0.0, k_arr[order])
