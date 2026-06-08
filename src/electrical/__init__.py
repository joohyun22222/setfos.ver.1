"""Electrical module — 1D drift-diffusion solver.

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
RecombinationProfile : 위치별 재결합 프로파일 (Step 9)

PoissonSolver    : finite-difference Poisson solver
ContinuitySolver : electron/hole continuity (Scharfetter-Gummel)
GummelSolver     : self-consistent Poisson + drift-diffusion loop
BiasSweepRunner  : bias voltage sweep orchestrator
RecombinationSolver : SRH + Langevin 재결합 프로파일 계산 (Step 9)
RecombinationSweep  : 전압 스윕 전체 프로파일 일괄 계산 (Step 9)
"""

from .bias_sweep import BiasSweepRunner
from .continuity import ContinuitySolver
from .drift_diffusion import GummelSolver, build_solver
from .mesh import Mesh1D, NodeProps, build_mesh
from .models import BiasPoint, DeviceState, GummelConfig, RecombinationProfile, SweepResult
from .poisson import PoissonSolver
from .recombination import RecombinationSolver, RecombinationSweep
from .traps import aggregate_trap_srh, srh_lifetimes, srh_stat_densities

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
    "RecombinationProfile",
    # solvers
    "PoissonSolver",
    "ContinuitySolver",
    "GummelSolver",
    "build_solver",
    # orchestration
    "BiasSweepRunner",
    # recombination (Step 9)
    "RecombinationSolver",
    "RecombinationSweep",
    # trap physics (Step 13)
    "srh_lifetimes",
    "srh_stat_densities",
    "aggregate_trap_srh",
]
