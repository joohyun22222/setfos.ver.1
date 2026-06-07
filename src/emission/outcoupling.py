"""Outcoupling efficiency calculators.

Interface contract
------------------
All models inherit from :class:`OutcouplingBase` and implement ``compute()``.
The :class:`StackContext` argument carries the full optical description of the
device so that any model — from geometric optics to a full Purcell-factor TMM
integration — can access what it needs without additional coupling to
the calling solver.

Provided models
---------------
NullOutcoupling      — η_out = 1 (identity; used as default placeholder)
FarFieldOutcoupling  — η_out = 1/(2n²)  (geometric-optics lower bound)

Step 7 will add ``PurcellOutcoupling`` which integrates over in-plane
k-vectors using the TMM dyadic Green's function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import OutcouplingResult, StackContext


class OutcouplingBase(ABC):
    """Abstract base for wavelength-resolved outcoupling calculators.

    Subclass this to plug a new optical model into :class:`EmissionSolver`
    without modifying any calling code::

        solver = EmissionSolver(nk_root=..., outcoupling=MyModel())
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier returned in :class:`OutcouplingResult`."""
        ...

    @abstractmethod
    def compute(self, context: StackContext) -> OutcouplingResult:
        """Return wavelength-resolved outcoupling efficiency.

        Parameters
        ----------
        context : :class:`StackContext`
            Full optical description of the device at the emitter position.

        Returns
        -------
        OutcouplingResult with ``eta_spectrum`` in [0, 1].
        """
        ...


# ---------------------------------------------------------------------------
# Null model — η_out = 1 (no loss)
# ---------------------------------------------------------------------------

class NullOutcoupling(OutcouplingBase):
    """Identity outcoupling: η_out(λ) = 1 at every wavelength.

    Serves as the default placeholder.  Use this when you want the field-
    and PL-weighted emission shape without any outcoupling correction.
    """

    @property
    def name(self) -> str:
        return "null"

    def compute(self, context: StackContext) -> OutcouplingResult:
        eta = np.ones_like(context.wavelengths_nm)
        return OutcouplingResult(
            wavelength_nm=context.wavelengths_nm,
            eta_spectrum=eta,
            model_name=self.name,
        )


# ---------------------------------------------------------------------------
# Geometric far-field model — η_out = 1 / (2 n_sub²)
# ---------------------------------------------------------------------------

class FarFieldOutcoupling(OutcouplingBase):
    """Geometric-optics estimate of forward outcoupling efficiency.

    For an isotropic point dipole embedded in a medium with refractive index
    n (≪ substrate), the fraction of emitted power that escapes through the
    substrate (Snell's window) is:

        η_out = 1 / (2 n_sub²)

    For glass (n_sub = 1.5): η_out ≈ 0.222.

    Limitations:
    - Ignores the microcavity enhancement / inhibition of the emission rate
      (Purcell factor — addressed in Step 7).
    - Ignores dipole orientation dependence.
    - Treats the device as a half-space (no interference effects).

    This model is wavelength-independent; the spectrum is flat.
    """

    @property
    def name(self) -> str:
        return "far_field_geometric"

    def compute(self, context: StackContext) -> OutcouplingResult:
        eta_val = 1.0 / (2.0 * context.n_sub ** 2)
        eta = np.full_like(context.wavelengths_nm, eta_val)
        return OutcouplingResult(
            wavelength_nm=context.wavelengths_nm,
            eta_spectrum=eta,
            model_name=self.name,
        )
