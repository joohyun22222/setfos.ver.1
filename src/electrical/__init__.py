"""Electrical module — 1D drift-diffusion solver skeleton.

Public API
----------
build_mesh       : construct :class:`Mesh1D` from a device stack
build_solver     : convenience factory for a ready-to-use :class:`GummelSolver`

Mesh1D           : 1D node mesh with per-node material properties
NodeProps        : per-node electrical material parameters

GummelConfig     : convergence and damping configuration
DeviceState      : φ, n, p, Jn, Jp at one bias point
BiasPoint        : DeviceState + convergence metadata
SweepResult      : ordered J-V scan result

PoissonSolver    : finite-difference Poisson solver
ContinuitySolver : electron/hole continuity skeleton (Step 8)
GummelSolver     : self-consistent Poisson + drift-diffusion loop
BiasSweepRunner  : bias voltage sweep orchestrator
"""

from .bias_sweep import BiasSweepRunner
from .continuity import ContinuitySolver
from .drift_diffusion import GummelSolver, build_solver
from .mesh import Mesh1D, NodeProps, build_mesh
from .models import BiasPoint, DeviceState, GummelConfig, SweepResult
from .poisson import PoissonSolver

__all__ = [
    # mesh
    "Mesh1D",
    "NodeProps",
    "build_mesh",
    # models
    "GummelConfig",
    "DeviceState",
    "BiasPoint",
    "SweepResult",
    # solvers
    "PoissonSolver",
    "ContinuitySolver",
    "GummelSolver",
    "build_solver",
    # orchestration
    "BiasSweepRunner",
]
