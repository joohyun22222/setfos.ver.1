"""Optical emission module — PL-weighted field emission and outcoupling interface."""

from .models import EmissionResult, EmitterProfile, OutcouplingResult, StackContext
from .outcoupling import FarFieldOutcoupling, NullOutcoupling, OutcouplingBase
from .solver import EmissionSolver

__all__ = [
    "EmissionSolver",
    "EmitterProfile",
    "EmissionResult",
    "OutcouplingResult",
    "StackContext",
    "OutcouplingBase",
    "NullOutcoupling",
    "FarFieldOutcoupling",
]
