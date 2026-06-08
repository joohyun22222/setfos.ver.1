"""Trap-state physics for SRH recombination in organic semiconductors (Step 13).

Physics
-------
Single-level Shockley-Read-Hall with explicit trap energy E_t:

    R_SRH = (np − ni²) / [τ_n·(p + p₁) + τ_p·(n + n₁)]

    τ_n = 1 / (N_t · σ_n · v_th_n)
    τ_p = 1 / (N_t · σ_p · v_th_p)

    n₁  = nᵢ · exp((E_t − E_i) / kT)
    p₁  = nᵢ · exp((E_i − E_t) / kT)

where E_i is the intrinsic Fermi level:

    E_i = E_c − Eg/2 − (kT/2)·ln(Nv/Nc)    (positive E_i is above midgap)

Midgap trap convention
----------------------
ΔE_t = energy_from_lumo_eV is the depth below LUMO:
    ΔE_t = 0     → trap at LUMO edge        → n₁ ≫ nᵢ  (electron-like)
    ΔE_t = Eg/2  → midgap trap              → n₁ = p₁ = nᵢ  (max R_SRH)
    ΔE_t = Eg    → trap at HOMO edge        → p₁ ≫ nᵢ  (hole-like)

n₁ · p₁ = nᵢ² holds exactly.

Multiple traps
--------------
`aggregate_trap_srh()` combines an arbitrary number of TrapState objects
into a single effective (tau_n_eff, tau_p_eff, n1_eff, p1_eff) suitable
for storage in :class:`~src.electrical.mesh.NodeProps`:

    1/τ_n_eff = Σ 1/τ_nᵢ   (harmonic sum — parallel recombination channels)
    1/τ_p_eff = Σ 1/τ_pᵢ

    n₁_eff = Σ(n₁ᵢ/τ_pᵢ) / Σ(1/τ_pᵢ)   (activity-weighted average)
    p₁_eff = Σ(p₁ᵢ/τ_nᵢ) / Σ(1/τ_nᵢ)
"""

from __future__ import annotations

import math

_KB_eV = 8.617333262e-5   # Boltzmann constant [eV/K]
_T0    = 300.0            # reference temperature [K]
_NI    = 1e10             # default intrinsic carrier density [m⁻³]


def srh_lifetimes(
    density_m3: float,
    sigma_n: float,
    sigma_p: float,
    vth_n: float = 1e5,
    vth_p: float = 1e5,
) -> tuple[float, float]:
    """Compute SRH carrier lifetimes from fundamental trap parameters.

    Parameters
    ----------
    density_m3  : trap density N_t [m⁻³]
    sigma_n     : electron capture cross-section [m²]
    sigma_p     : hole capture cross-section [m²]
    vth_n       : electron thermal velocity [m/s]  (default 1e5 for organics)
    vth_p       : hole thermal velocity [m/s]      (default 1e5)

    Returns
    -------
    (tau_n [s], tau_p [s])
    """
    tau_n = 1.0 / (density_m3 * sigma_n * vth_n)
    tau_p = 1.0 / (density_m3 * sigma_p * vth_p)
    return tau_n, tau_p


def srh_stat_densities(
    energy_from_lumo_eV: float,
    Eg_eV: float,
    Nc: float,
    Nv: float,
    ni: float = _NI,
    T: float = _T0,
) -> tuple[float, float]:
    """Compute SRH statistical densities n₁ and p₁ for a trap at energy E_t.

    Parameters
    ----------
    energy_from_lumo_eV : ΔE_t — trap depth below LUMO [eV].
                          0 = LUMO edge; Eg/2 = midgap; Eg = HOMO edge.
    Eg_eV               : transport bandgap [eV]
    Nc, Nv              : effective conduction/valence-band DOS [m⁻³]
    ni                  : intrinsic carrier density [m⁻³]
    T                   : temperature [K]

    Returns
    -------
    (n₁ [m⁻³], p₁ [m⁻³])   — n₁ · p₁ = nᵢ² exactly
    """
    kT = _KB_eV * T
    # E_t − E_i = (Eg/2 − ΔE_t) − (kT/2)·ln(Nv/Nc)
    delta = 0.5 * Eg_eV - energy_from_lumo_eV
    if Nc > 0.0 and Nv > 0.0:
        delta -= 0.5 * kT * math.log(Nv / Nc)
    arg = max(min(delta / kT, 40.0), -40.0)   # clamp to avoid overflow
    n1 = ni * math.exp(arg)
    p1 = ni * math.exp(-arg)
    return n1, p1


def aggregate_trap_srh(
    trap_states: list,
    Eg_eV: float,
    Nc: float,
    Nv: float,
    ni: float = _NI,
    T: float = _T0,
) -> tuple[float, float, float, float]:
    """Reduce a list of TrapState objects to a single effective (τ_n, τ_p, n₁, p₁).

    Uses harmonic-sum lifetimes and activity-weighted statistical densities.
    For a single trap level the result is exact.

    Parameters
    ----------
    trap_states : list of :class:`~src.io.models.TrapState` objects
    Eg_eV       : bandgap [eV]
    Nc, Nv      : effective DOS [m⁻³]
    ni          : intrinsic carrier density [m⁻³]
    T           : temperature [K]

    Returns
    -------
    (tau_n_eff [s], tau_p_eff [s], n1_eff [m⁻³], p1_eff [m⁻³])
    """
    if not trap_states:
        return 1e-6, 1e-6, ni, ni

    inv_tau_n = 0.0
    inv_tau_p = 0.0
    n1_weighted = 0.0
    p1_weighted = 0.0

    for ts in trap_states:
        tn, tp = srh_lifetimes(
            ts.density_m3, ts.sigma_n, ts.sigma_p, ts.vth_n, ts.vth_p
        )
        n1, p1 = srh_stat_densities(ts.energy_from_lumo_eV, Eg_eV, Nc, Nv, ni, T)
        inv_tn = 1.0 / tn
        inv_tp = 1.0 / tp
        inv_tau_n += inv_tn
        inv_tau_p += inv_tp
        n1_weighted += n1 * inv_tp   # weight by 1/τ_p (activity in electron denominator)
        p1_weighted += p1 * inv_tn   # weight by 1/τ_n (activity in hole denominator)

    tau_n_eff = 1.0 / inv_tau_n
    tau_p_eff = 1.0 / inv_tau_p
    n1_eff = n1_weighted / inv_tau_p
    p1_eff = p1_weighted / inv_tau_n
    return tau_n_eff, tau_p_eff, n1_eff, p1_eff
