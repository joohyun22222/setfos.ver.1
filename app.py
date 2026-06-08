"""SETFOS OLED Simulation Dashboard — Optical + Electrical + J-V-L.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.emission import EmissionSolver, FarFieldOutcoupling, NullOutcoupling
from src.io import load_project, load_emitter_config
from src.io.models import EmitterConfig
from src.optics.field_profile import compute_field_profile
from src.optics.nk_data import NKDataProvider
from src.optics.rta import RTASolver

NK_ROOT  = ROOT / "data" / "nk"
SAMPLES  = ROOT / "configs" / "samples"
PL_ROOT  = ROOT / "data" / "pl"

# J-V-L imports
from src.electrical import build_mesh, build_solver, GummelConfig, BiasSweepRunner
from src.electrical.recombination import RecombinationSolver
from src.electrical.continuity import ContinuitySolver
from src.jvl import JVLCalculator, JVLResult

# Layer color map
LAYER_COLORS = {
    "ITO":  "rgba(100,200,255,0.25)",
    "TCTA": "rgba(180,130,255,0.25)",
    "EML":  "rgba(80,255,120,0.35)",
    "TPBi": "rgba(255,200,80,0.25)",
    "Al":   "rgba(200,200,200,0.35)",
}
LAYER_LINE_COLORS = {
    "ITO":  "#64C8FF",
    "TCTA": "#B482FF",
    "EML":  "#50FF78",
    "TPBi": "#FFC850",
    "Al":   "#C8C8C8",
}


# ---------------------------------------------------------------------------
# Cached simulation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="시뮬레이션 실행 중…")
def run_simulation(z_distribution: str, oc_model: str, wl_start: float, wl_end: float, wl_step: float):
    project   = load_project(SAMPLES / "oled_input.yaml")
    stack     = project.device_stack
    mat_db    = project.material_db
    solver_cfg = project.solver_config

    # Override wavelength grid from sidebar
    from src.io.models import WavelengthGrid
    solver_cfg.wavelength_grid = WavelengthGrid(start_nm=wl_start, end_nm=wl_end, step_nm=wl_step)

    wavelengths = np.arange(wl_start, wl_end + wl_step * 0.5, wl_step)

    # Build optical data
    nk_provider = NKDataProvider(NK_ROOT)
    n_layers, d_layers, layer_names, mat_names = [], [], [], []
    for layer in stack.layers:
        mat = mat_db.get(layer.material_name)
        n_layers.append(nk_provider.get_nk(mat, wavelengths))
        d_layers.append(layer.thickness_nm)
        layer_names.append(layer.layer_name)
        mat_names.append(layer.material_name)

    # Field profile
    fp = compute_field_profile(
        n_layers, d_layers, wavelengths,
        layer_names=layer_names,
        material_names=mat_names,
        n_inc=1.0, n_sub=1.5,
        z_resolution_nm=1.0,
    )

    # Emission
    emitter_cfg = EmitterConfig(
        layer_name="EML",
        material="Ir(ppy)3",
        pl_spectrum_file=str(PL_ROOT / "Irppy3_pl.csv"),
        horizontal_fraction=0.78,
        z_distribution=z_distribution,
        emitter_type="phosphorescent",
    )

    oc = FarFieldOutcoupling() if oc_model == "FarField (1/2n²)" else NullOutcoupling()
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT, outcoupling=oc)
    result = solver.solve(stack, mat_db, solver_cfg, emitter_cfg)

    return fp, result, layer_names, d_layers, wavelengths


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_layer_bands(fig, boundaries, layer_names, row=1, col=1, xaxis="x"):
    """Overlay semi-transparent layer bands on a plot."""
    for i, name in enumerate(layer_names):
        z0, z1 = boundaries[i], boundaries[i + 1]
        color = LAYER_COLORS.get(name, "rgba(180,180,180,0.15)")
        fig.add_vrect(
            x0=z0, x1=z1, fillcolor=color,
            line_width=1, line_color=LAYER_LINE_COLORS.get(name, "#aaa"),
            opacity=1.0, row=row, col=col,
            annotation_text=name,
            annotation_position="top left",
            annotation_font_size=10,
        )


def field_heatmap(fp, wavelengths, layer_names):
    E2 = fp.E_squared                        # (N_z, N_wl)
    fig = go.Figure(go.Heatmap(
        x=wavelengths,
        y=fp.z_nm,
        z=E2,
        colorscale="Viridis",
        colorbar=dict(title="|E|²", thickness=14),
        hovertemplate="λ=%{x:.0f} nm<br>z=%{y:.1f} nm<br>|E|²=%{z:.3f}<extra></extra>",
    ))
    # Layer boundaries
    for z_b in fp.layer_boundaries_nm[1:-1]:
        fig.add_hline(y=z_b, line_dash="dash", line_color="white", line_width=1)
    # Layer labels
    for i, name in enumerate(layer_names):
        z_mid = fp.layer_center_z(i)
        fig.add_annotation(x=wavelengths[0] + 5, y=z_mid, text=name,
                           showarrow=False, font=dict(color="white", size=10),
                           xanchor="left")

    fig.update_layout(
        title="|E|²(z, λ) Field Profile",
        xaxis_title="Wavelength (nm)",
        yaxis_title="z (nm)",
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def field_slice_fig(fp, wl_nm: float, layer_names):
    E = fp.get_field_at_wavelength(wl_nm)
    E2 = np.abs(E) ** 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fp.z_nm, y=E2,
        mode="lines", line=dict(color="#00b4d8", width=2),
        name=f"|E|² at {wl_nm:.0f} nm",
    ))
    # Layer shading
    for i, name in enumerate(layer_names):
        z0, z1 = fp.layer_boundaries_nm[i], fp.layer_boundaries_nm[i + 1]
        color = LAYER_COLORS.get(name, "rgba(180,180,180,0.15)")
        fig.add_vrect(x0=z0, x1=z1, fillcolor=color, line_width=0, opacity=1.0)
        fig.add_vline(x=z0, line_color=LAYER_LINE_COLORS.get(name, "#aaa"), line_width=1, line_dash="dot")
    fig.add_vline(x=fp.layer_boundaries_nm[-1], line_color="#888", line_width=1, line_dash="dot")

    fig.update_layout(
        title=f"|E|²(z) at λ = {wl_nm:.0f} nm",
        xaxis_title="z (nm)",
        yaxis_title="|E|²",
        height=320,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def absorption_fig(fp, wavelengths, layer_names):
    fig = go.Figure()
    palette = ["#00b4d8", "#b482ff", "#50ff78", "#ffc850", "#c8c8c8",
               "#ff6b6b", "#ffd166", "#06d6a0"]
    for i, la in enumerate(fp.layer_absorption):
        fig.add_trace(go.Scatter(
            x=wavelengths, y=la.A_spectrum,
            mode="lines", name=la.layer_name,
            line=dict(color=palette[i % len(palette)], width=2),
            fill="tozeroy", fillcolor=palette[i % len(palette)].replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in palette[i % len(palette)] else None,
        ))
    fig.add_trace(go.Scatter(
        x=wavelengths, y=fp.total_layer_absorption,
        mode="lines", name="Total A",
        line=dict(color="white", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Layer-Resolved Absorptance A(λ)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorptance",
        yaxis=dict(range=[0, 1]),
        height=380,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def absorption_bar_fig(fp):
    names = [la.layer_name for la in fp.layer_absorption]
    means = [la.A_mean for la in fp.layer_absorption]
    peaks = [la.A_peak for la in fp.layer_absorption]
    palette = ["#00b4d8", "#b482ff", "#50ff78", "#ffc850", "#c8c8c8"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="A_mean", x=names, y=means,
                         marker_color=palette[:len(names)], opacity=0.9))
    fig.add_trace(go.Bar(name="A_peak", x=names, y=peaks,
                         marker_color=palette[:len(names)], opacity=0.5,
                         marker_pattern_shape="x"))
    fig.update_layout(
        title="Layer Mean / Peak Absorptance",
        yaxis_title="Absorptance",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=320,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def emission_fig(result, wavelengths):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths, y=result.pl_spectrum / result.pl_spectrum.max(),
        mode="lines", name="PL spectrum (norm.)",
        line=dict(color="#b482ff", width=2, dash="dot"),
    ))
    E2_norm = result.E_squared / (result.E_squared.max() + 1e-30)
    fig.add_trace(go.Scatter(
        x=wavelengths, y=E2_norm,
        mode="lines", name="|E|² at emitter (norm.)",
        line=dict(color="#00b4d8", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths, y=result.weighted_spectrum / result.weighted_spectrum.max(),
        mode="lines", name="PL × |E|² (norm.)",
        line=dict(color="#ffc850", width=2),
    ))
    em = result.emission_spectrum
    fig.add_trace(go.Scatter(
        x=wavelengths, y=em / em.max(),
        mode="lines", name="Emission (final, norm.)",
        line=dict(color="#50ff78", width=3),
        fill="tozeroy",
        fillcolor="rgba(80,255,120,0.12)",
    ))
    # Peak marker
    peak_wl = result.peak_emission_nm
    peak_idx = int(np.argmin(np.abs(wavelengths - peak_wl)))
    fig.add_vline(x=peak_wl, line_color="#50ff78", line_dash="dash", line_width=1)
    fig.add_annotation(x=peak_wl, y=1.02, text=f"{peak_wl:.0f} nm",
                       showarrow=False, font=dict(color="#50ff78", size=12))

    fig.update_layout(
        title="Optical Emission Spectrum",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (normalised)",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SETFOS OLED Simulator",
    page_icon="💡",
    layout="wide",
)

st.title("💡 SETFOS — OLED Optical Simulation Dashboard")
st.caption("Transfer Matrix Method | Layer-resolved Field Profile & Emission")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("시뮬레이션 설정")

    st.subheader("발광체 설정")
    z_dist = st.selectbox(
        "Z-distribution",
        ["center", "front", "back", "uniform"],
        index=0,
        help="EML 내 발광체 위치",
    )

    st.subheader("Outcoupling 모델")
    oc_model = st.selectbox(
        "모델",
        ["Null (η=1)", "FarField (1/2n²)"],
        index=1,
        help="NullOutcoupling: η_out=1, FarField: η_out=1/(2n²)≈0.222",
    )

    st.subheader("파장 범위")
    wl_start = st.number_input("Start (nm)", value=380.0, step=10.0)
    wl_end   = st.number_input("End (nm)",   value=780.0, step=10.0)
    wl_step  = st.number_input("Step (nm)",  value=5.0,   step=1.0)

    run_btn = st.button("▶ 시뮬레이션 실행", use_container_width=True, type="primary")

    st.divider()
    st.subheader("J-V-L 효율 파라미터")
    eta_rad = st.slider(
        "η_rad (방사 효율)",
        min_value=0.0, max_value=1.0, value=0.80, step=0.05,
        help="PLQY × η_spin. 인광(Ir(ppy)₃): PLQY×1.0, 형광: PLQY×0.25",
        key="eta_rad",
    )
    eta_out = st.slider(
        "η_out (광추출 효율)",
        min_value=0.0, max_value=1.0, value=0.20, step=0.01,
        help="유리 기판 OLED ≈ 0.20 (≈1/2n²). 이상적 = 1.0",
        key="eta_out",
    )
    st.caption(f"η_rad × η_out = {eta_rad * eta_out:.3f}  (유효 광자 추출 효율)")

    st.divider()
    st.subheader("OLED Stack")
    st.markdown("""
| Layer | Material | d (nm) |
|-------|----------|--------|
| ITO   | ITO      | 150    |
| TCTA  | TCTA     | 40     |
| EML   | TCTA:Ir(ppy)₃ | 30 |
| TPBi  | TPBi     | 50     |
| Al    | Al       | 100    |
""")

# ── Run simulation ─────────────────────────────────────────────────────────
fp, result, layer_names, d_layers, wavelengths = run_simulation(
    z_dist, oc_model,
    float(wl_start), float(wl_end), float(wl_step),
)

# ── Key metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Peak Emission", f"{result.peak_emission_nm:.0f} nm")
col2.metric("EML z-position", f"{result.z_emitter_nm:.1f} nm")
col3.metric(
    "η_out",
    f"{result.eta_out:.4f}" if result.eta_out is not None else "N/A",
    help="PL-weighted mean outcoupling efficiency",
)
col4.metric("Total A (mean)", f"{float(np.mean(fp.total_layer_absorption)):.4f}")
col5.metric(
    "Al absorptance",
    f"{float(np.mean(fp.layer_absorption[-1].A_spectrum)):.4f}",
    help="Mean absorptance in metal cathode (loss)",
)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Field Profile", "📊 Layer Absorption", "🌿 Emission", "📈 J-V-L / EQE"])

# ── Tab 1: Field Profile ───────────────────────────────────────────────────
with tab1:
    st.plotly_chart(field_heatmap(fp, wavelengths, layer_names), use_container_width=True)

    st.subheader("|E|²(z) — 단파장 슬라이스")
    wl_slider = st.slider(
        "Wavelength (nm)",
        min_value=int(wavelengths[0]),
        max_value=int(wavelengths[-1]),
        value=int(result.peak_emission_nm),
        step=int(wl_step),
    )
    st.plotly_chart(field_slice_fig(fp, float(wl_slider), layer_names), use_container_width=True)

# ── Tab 2: Absorption ──────────────────────────────────────────────────────
with tab2:
    st.plotly_chart(absorption_fig(fp, wavelengths, layer_names), use_container_width=True)
    st.plotly_chart(absorption_bar_fig(fp), use_container_width=True)

    st.subheader("Layer Absorptance 수치")
    import pandas as pd
    abs_data = {
        "Layer": [la.layer_name for la in fp.layer_absorption],
        "Material": [la.material_name for la in fp.layer_absorption],
        "Thickness (nm)": [la.thickness_nm for la in fp.layer_absorption],
        "A_mean": [f"{la.A_mean:.4f}" for la in fp.layer_absorption],
        "A_peak": [f"{la.A_peak:.4f}" for la in fp.layer_absorption],
    }
    st.dataframe(pd.DataFrame(abs_data), use_container_width=True, hide_index=True)

# ── Tab 3: Emission ────────────────────────────────────────────────────────
with tab3:
    st.plotly_chart(emission_fig(result, wavelengths), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Emission 요약")
        st.markdown(f"""
- **Peak wavelength:** {result.peak_emission_nm:.0f} nm
- **Emitter z:** {result.z_emitter_nm:.1f} nm  (EML: 190–220 nm)
- **Z-distribution:** `{z_dist}`
- **Outcoupling model:** {result.outcoupling.model_name if result.outcoupling else "none"}
- **η_out:** {f"{result.eta_out:.4f}" if result.eta_out is not None else "N/A"}
- **mean |E|²:** {float(np.mean(result.E_squared)):.4f}
""")
    with col_b:
        st.subheader("Emission Spectrum 데이터")
        em_df = pd.DataFrame({
            "λ (nm)": wavelengths,
            "PL": result.pl_spectrum.round(6),
            "|E|²": result.E_squared.round(6),
            "PL×|E|²": result.weighted_spectrum.round(8),
            "Emission": result.emission_spectrum.round(8),
        })
        st.dataframe(em_df, use_container_width=True, height=300)


# ── Tab 4: J-V-L / EQE ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="J-V-L 시뮬레이션 실행 중… (최초 1회)")
def _build_jvl_calculator(_fp, eta_rad: float, eta_out: float):
    project = load_project(SAMPLES / "oled_input.yaml")
    emitter_cfg = load_emitter_config(SAMPLES / "emitter_irppy3.yaml")
    mesh = build_mesh(project.device_stack, project.material_db, z_resolution_nm=1.0)
    # 수렴 안정성 개선: 반복 횟수 증가, damping 강화
    cfg  = GummelConfig(max_iterations=300, tolerance=1e-5, damping=0.3)
    gummel = build_solver(mesh, cfg)
    cont = ContinuitySolver(mesh)
    rec_solver = RecombinationSolver(mesh, cont)
    solver = EmissionSolver(nk_root=NK_ROOT, project_root=ROOT)
    wl = _fp.wavelength_nm
    emitter_profile = solver._load_emitter(emitter_cfg, wl)
    return JVLCalculator(
        gummel=gummel, rec_solver=rec_solver, field_profile=_fp,
        emitter_profile=emitter_profile, emitter_layer="EML",
        eta_rad=eta_rad, eta_out=eta_out,
    )


@st.cache_data(show_spinner="J-V-L sweep 계산 중…")
def run_jvl(_calc, v_start, v_end, n_pts):
    return _calc.run(v_start=v_start, v_end=v_end, n_points=n_pts)


with tab4:
    st.subheader("J-V-L Sweep 설정")
    jc1, jc2, jc3 = st.columns(3)
    jvl_v_start = jc1.number_input("V_start (V)", value=0.0, step=0.5, key="jvl_vs")
    jvl_v_end   = jc2.number_input("V_end (V)",   value=4.0, step=0.5, key="jvl_ve")
    jvl_n_pts   = jc3.number_input("N_points",    value=5,   step=1,   min_value=2, key="jvl_np")

    run_jvl_btn = st.button("▶ J-V-L 실행", type="primary", key="jvl_run")

    if run_jvl_btn or "jvl_result" in st.session_state:
        if run_jvl_btn:
            calc = _build_jvl_calculator(fp, eta_rad, eta_out)
            st.session_state["jvl_result"] = run_jvl(calc, jvl_v_start, jvl_v_end, int(jvl_n_pts))

        res: JVLResult = st.session_state["jvl_result"]

        # ── Key metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("L_max", f"{float(res.luminance.max()):.1f} cd/m²")
        m2.metric("EQE_max", f"{float(res.eqe_pct.max()):.2f} %")
        m3.metric("CE_max", f"{float(res.cd_per_A.max()):.2f} cd/A")
        m4.metric("PE_max", f"{float(res.lm_per_W.max()):.2f} lm/W")
        m5.metric("Peak λ", f"{float(res.peak_wavelength_nm[res.luminance.argmax()]):.0f} nm")

        st.divider()

        # ── J-V curve ─────────────────────────────────────────────────────
        col_jv, col_lv = st.columns(2)
        with col_jv:
            J_ma = np.abs(res.J_mAcm2)
            J_safe = np.where(J_ma > 0, J_ma, np.nan)
            fig_jv = go.Figure()
            fig_jv.add_trace(go.Scatter(
                x=res.voltages, y=J_safe,
                mode="lines+markers",
                line=dict(color="#00b4d8", width=2),
                marker=dict(size=6),
                name="|J| (mA/cm²)",
            ))
            fig_jv.update_layout(
                title="J-V Characteristic",
                xaxis_title="Voltage (V)",
                yaxis_title="|J| (mA/cm²)",
                yaxis_type="log",
                height=340,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig_jv, use_container_width=True)

        with col_lv:
            L_safe = np.where(res.luminance > 0, res.luminance, np.nan)
            fig_lv = go.Figure()
            fig_lv.add_trace(go.Scatter(
                x=res.voltages, y=L_safe,
                mode="lines+markers",
                line=dict(color="#50ff78", width=2),
                marker=dict(size=6),
                name="L (cd/m²)",
            ))
            fig_lv.update_layout(
                title="Luminance-Voltage (L-V)",
                xaxis_title="Voltage (V)",
                yaxis_title="Luminance (cd/m²)",
                yaxis_type="log",
                height=340,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig_lv, use_container_width=True)

        # ── EQE / CE / PE ──────────────────────────────────────────────────
        col_eq, col_ce = st.columns(2)
        with col_eq:
            fig_eqe = go.Figure()
            fig_eqe.add_trace(go.Scatter(
                x=res.voltages, y=res.eqe_pct,
                mode="lines+markers",
                line=dict(color="#b482ff", width=2),
                marker=dict(size=6),
                name="EQE (%)",
            ))
            fig_eqe.update_layout(
                title="External Quantum Efficiency (EQE)",
                xaxis_title="Voltage (V)",
                yaxis_title="EQE (%)",
                height=320,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig_eqe, use_container_width=True)

        with col_ce:
            fig_ce = go.Figure()
            fig_ce.add_trace(go.Scatter(
                x=res.voltages, y=res.cd_per_A,
                mode="lines+markers",
                line=dict(color="#ffc850", width=2),
                marker=dict(size=6),
                name="CE (cd/A)",
            ))
            fig_ce.add_trace(go.Scatter(
                x=res.voltages, y=res.lm_per_W,
                mode="lines+markers",
                line=dict(color="#ff6b6b", width=2, dash="dash"),
                marker=dict(size=6),
                name="PE (lm/W)",
            ))
            fig_ce.update_layout(
                title="Current & Power Efficiency",
                xaxis_title="Voltage (V)",
                yaxis_title="Efficiency",
                height=320,
                margin=dict(l=60, r=20, t=50, b=50),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_ce, use_container_width=True)

        # ── Emission spectrum at selected bias ─────────────────────────────
        st.subheader("바이어스별 EL 스펙트럼")
        v_sel = st.select_slider(
            "Voltage (V)",
            options=[f"{v:.3f}" for v in res.voltages],
            value=f"{res.voltages[-1]:.3f}",
            key="jvl_v_sel",
        )
        wl_sp, em_sp = res.spectrum_at(float(v_sel))
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=wl_sp, y=em_sp / (em_sp.max() + 1e-30),
            mode="lines",
            line=dict(color="#50ff78", width=2),
            fill="tozeroy",
            fillcolor="rgba(80,255,120,0.15)",
            name=f"V = {v_sel} V",
        ))
        fig_sp.update_layout(
            title=f"EL Spectrum @ {v_sel} V",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity (norm.)",
            height=300,
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_sp, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────
        st.subheader("J-V-L 수치 결과")
        import pandas as _pd
        jvl_df = _pd.DataFrame({
            "V (V)":         res.voltages.round(3),
            "J (A/m²)":      res.J_Am2.round(4),
            "J (mA/cm²)":    res.J_mAcm2.round(4),
            "L (cd/m²)":     res.luminance.round(3),
            "EQE (%)":       res.eqe_pct.round(3),
            "CE (cd/A)":     res.cd_per_A.round(4),
            "PE (lm/W)":     res.lm_per_W.round(4),
            "Peak λ (nm)":   res.peak_wavelength_nm.round(1),
        })
        st.dataframe(jvl_df, use_container_width=True, hide_index=True)

        # ── CSV download ───────────────────────────────────────────────────
        csv_buf = jvl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="CSV 다운로드",
            data=csv_buf,
            file_name="jvl_result.csv",
            mime="text/csv",
        )
    else:
        st.info("'▶ J-V-L 실행' 버튼을 눌러 시뮬레이션을 시작하세요.")
