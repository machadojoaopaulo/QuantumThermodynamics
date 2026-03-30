"""
app_ciclos.py — Visualizador interativo dos ciclos de Otto quântico e clássico.

Execução:
    .venv/bin/streamlit run app_ciclos.py
    (ou use o script: ./iniciar_app.sh)
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from thermodynamics_functions import (
    Z, energia_livre, energia_media, entropia,
    magnetizacao, susceptibilidade, calor_especifico,
    ciclo_classico, ciclo_quantico, ciclo_Q,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ciclos de Otto — Variedades Termodinâmicas",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ciclos de Otto — Variedades Termodinâmicas")
st.caption("Modelo de 2 spins · Z(J,h,T) = 1 + 2·cosh(2h/T) + exp(8J/T)")

# ══════════════════════════════════════════════════════════════════════════════
#  BARRA LATERAL — PARÂMETROS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Parâmetros")

    func_name = st.selectbox(
        "Função termodinâmica",
        ["M — Magnetização", "Z — Partição", "F — Energia Livre",
         "U — Energia Interna", "S — Entropia",
         "χ — Susceptibilidade", "C — Calor Específico"],
    )

    st.divider()
    st.subheader("Ciclo de Otto")

    col1, col2 = st.columns(2)
    with col1:
        hi = st.slider("hᵢ  (campo inicial)", 0.1, 4.0, 1.0, 0.05)
        Tc = st.slider("Tc  (banho frio)",    0.1, 8.0, 2.0, 0.1)
    with col2:
        hf = st.slider("hf  (campo final)",  0.1, 4.0, 2.0, 0.05)
        Th = st.slider("Th  (banho quente)", 0.1, 8.0, 1.0, 0.1)

    J      = st.slider("J  (acoplamento)",   0.0, 1.5, 0.51, 0.01)
    PASSOS = st.select_slider("Passos", [50, 100, 150, 200], value=150)

    st.divider()
    paleta = st.selectbox("Paleta 3D", ["plasma","viridis","magma","cividis","turbo","RdBu"], index=0)

    # Aviso se ciclo não faz sentido
    if hi == hf:
        st.warning("hᵢ = hf: ciclo degenerado.")
    if Tc == Th:
        st.warning("Tc = Th: sem gradiente térmico.")

# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES DISPONÍVEIS
# ══════════════════════════════════════════════════════════════════════════════
FUNCOES = {
    "M — Magnetização":    (magnetizacao,     "M(J,h,T)",  "M"),
    "Z — Partição":        (Z,                "Z(J,h,T)",  "Z"),
    "F — Energia Livre":   (energia_livre,    "F(J,h,T)",  "F"),
    "U — Energia Interna": (energia_media,    "U(J,h,T)",  "U"),
    "S — Entropia":        (entropia,         "S(J,h,T)",  "S"),
    "χ — Susceptibilidade":(susceptibilidade, "χ(J,h,T)",  "χ"),
    "C — Calor Específico":(calor_especifico, "C(J,h,T)",  "C"),
}
func, flabel, fname = FUNCOES[func_name]

# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DOS CICLOS
# ══════════════════════════════════════════════════════════════════════════════
try:
    est_C, cam_C, Tc2, Th4 = ciclo_classico(J, hi, hf, Tc, Th, PASSOS)
    est_Q, cam_Q, _,   _   = ciclo_quantico(J, hi, hf, Tc, Th, PASSOS)
    QC = ciclo_Q(func, J, cam_C)
    Q1 = float(func(J, hi, Tc))
    Q3 = float(func(J, hf, Th))
    cycle_ok = True
except Exception as e:
    st.error(f"Erro ao calcular o ciclo: {e}")
    cycle_ok = False

# ── Informações resumidas ─────────────────────────────────────────────────
if cycle_ok:
    info_cols = st.columns(4)
    info_cols[0].metric("Q₁ = Q(J,hᵢ,Tc)", f"{Q1:.4f}")
    info_cols[1].metric("Q₃ = Q(J,hf,Th)", f"{Q3:.4f}")
    info_cols[2].metric("Tc₂ clássico",     f"{Tc2:.4f}")
    info_cols[3].metric("Th₄ clássico",     f"{Th4:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  CORES DOS CICLOS
# ══════════════════════════════════════════════════════════════════════════════
LEG_C = {'12':'lime',   '23':'tomato', '34':'orange', '41':'deepskyblue'}
LEG_Q = {'12':'#1a9850','23':'#d73027','34':'#f46d43','41':'#4575b4'}

# ══════════════════════════════════════════════════════════════════════════════
#  ABAS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈  Q(T) — Curvas 1D", "🔲  Espaço (h, Q)", "🌐  Superfície 3D"])

# ─────────────────────────────────────────────────────────────────────────────
#  ABA 1 — Q(T) para h = hi e h = hf
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not cycle_ok:
        st.stop()

    T_plot = np.linspace(0.1, 20, 1000)
    fig1 = make_subplots(rows=1, cols=2,
                         subplot_titles=[f"h = {hi}", f"h = {hf}"],
                         horizontal_spacing=0.1)

    for col_idx, h_val in enumerate([hi, hf], start=1):
        Qvals = func(J, h_val, T_plot)
        fig1.add_trace(go.Scatter(
            x=T_plot, y=Qvals, mode='lines',
            line=dict(color='white', width=2.5),
            name=f"{fname}(T), h={h_val}",
            legendgroup=f'curva{col_idx}', showlegend=True,
        ), row=1, col=col_idx)

        # Estados clássicos
        for k, v in est_C.items():
            if abs(v['h'] - h_val) < 1e-9:
                Qv = float(func(J, v['h'], v['T']))
                fig1.add_trace(go.Scatter(
                    x=[v['T']], y=[Qv],
                    mode='markers+text', text=[v['label']],
                    textposition='top right',
                    marker=dict(size=10, color='lime',
                                line=dict(color='white', width=1)),
                    name='Estados clássicos',
                    legendgroup='estC', showlegend=(k == 1 and col_idx == 1),
                ), row=1, col=col_idx)

        # Estados quânticos: apenas #1 e #3 (T definido)
        for key in ['1', '3']:
            v = est_Q[key]
            if abs(v['h'] - h_val) < 1e-9:
                Qv = float(func(J, v['h'], v['T']))
                fig1.add_trace(go.Scatter(
                    x=[v['T']], y=[Qv],
                    mode='markers+text', text=[v['label']],
                    textposition='top left',
                    marker=dict(size=10, color='cyan', symbol='square',
                                line=dict(color='white', width=1)),
                    name='Q: equilíbrio (#1, #3)',
                    legendgroup='estQeq',
                    showlegend=(key == '1' and col_idx == 1),
                ), row=1, col=col_idx)

    fig1.update_xaxes(title_text="T", range=[0, 20])
    fig1.update_yaxes(title_text=flabel)
    fig1.update_layout(
        height=450,
        title=f"{flabel} em função de T  |  J = {J}",
        template='plotly_dark',
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center',
                    groupclick='togglegroup'),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Estados #2 e #4 não aparecem aqui — estão fora do equilíbrio (T indefinido).")

# ─────────────────────────────────────────────────────────────────────────────
#  ABA 2 — ESPAÇO (h, Q) COM RETÂNGULO QUÂNTICO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not cycle_ok:
        st.stop()

    h_bg  = np.linspace(max(0.05, hi - 1.2), hf + 1.2, 200)
    T_iso = np.concatenate([np.linspace(0.5, 2.0, 5),
                             np.linspace(3.0, 8.0, 4),
                             np.linspace(10,  20,  4)])

    fig2 = go.Figure()

    # Curvas iso-T de fundo (colormap coolwarm → azul=frio, vermelho=quente)
    import matplotlib
    cmap = matplotlib.colormaps['coolwarm']
    T_min, T_max = T_iso.min(), T_iso.max()
    for T_val in T_iso:
        Q_bg = func(J, h_bg, T_val)
        r, g, b, _ = cmap((T_val - T_min) / (T_max - T_min))
        col = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.6)'
        fig2.add_trace(go.Scatter(
            x=h_bg, y=Q_bg, mode='lines',
            line=dict(color=col, width=1),
            hovertemplate=f'T={T_val:.1f}<br>h=%{{x:.2f}}<br>Q=%{{y:.3f}}<extra></extra>',
            showlegend=False,
        ))

    # Ciclo clássico
    for leg in ['12','23','34','41']:
        fig2.add_trace(go.Scatter(
            x=cam_C[leg]['h'], y=QC[leg],
            mode='lines', line=dict(color=LEG_C[leg], width=2.5),
            name='Clássico (S=cte)',
            legendgroup='cicloC', showlegend=(leg == '12'),
        ))
    for k, v in est_C.items():
        Qv = float(func(J, v['h'], v['T']))
        fig2.add_trace(go.Scatter(
            x=[v['h']], y=[Qv],
            mode='markers+text', text=[v['label']], textposition='top right',
            marker=dict(size=10, color='lime', line=dict(color='white', width=1)),
            name='Estados clássicos',
            legendgroup='estC', showlegend=(k == 1),
        ))

    # Ciclo quântico — RETÂNGULO
    fig2.add_trace(go.Scatter(
        x=[hi, hf], y=[Q1, Q1],
        mode='lines', line=dict(color=LEG_Q['12'], width=2.5),
        name='Quântico: adiab. 1→2', legendgroup='cicloQA12', showlegend=True,
    ))
    fig2.add_trace(go.Scatter(
        x=[hf, hi], y=[Q3, Q3],
        mode='lines', line=dict(color=LEG_Q['34'], width=2.5),
        name='Quântico: adiab. 3→4', legendgroup='cicloQA34', showlegend=True,
    ))
    fig2.add_trace(go.Scatter(
        x=[hf, hf], y=[Q1, Q3],
        mode='lines', line=dict(color=LEG_Q['23'], width=2.5, dash='dash'),
        name='Quântico: isoc. 2→3', legendgroup='cicloQI23', showlegend=True,
    ))
    fig2.add_trace(go.Scatter(
        x=[hi, hi], y=[Q3, Q1],
        mode='lines', line=dict(color=LEG_Q['41'], width=2.5, dash='dash'),
        name='Quântico: isoc. 4→1', legendgroup='cicloQI41', showlegend=True,
    ))

    # Estados quânticos: preenchidos (#1,#3) e abertos (#2,#4)
    for (key, hv, Qv, symb, col, filled) in [
        ('1Q', hi, Q1, 'square',      'cyan',             True),
        ('3Q', hf, Q3, 'square',      'cyan',             True),
        ('2Q', hf, Q1, 'square-open', 'rgba(0,0,0,0)',    False),
        ('4Q', hi, Q3, 'square-open', 'rgba(0,0,0,0)',    False),
    ]:
        is_eq = filled
        fig2.add_trace(go.Scatter(
            x=[hv], y=[Qv],
            mode='markers+text', text=[key], textposition='top right',
            marker=dict(size=12, color=col, symbol=symb,
                        line=dict(color='cyan', width=2)),
            name='Q: equilíbrio' if is_eq else 'Q: fora-equilíbrio',
            legendgroup='estQeq' if is_eq else 'estQneq',
            showlegend=(key in ['1Q', '2Q']),
        ))

    fig2.update_layout(
        height=520,
        title=f"Ciclos no espaço (h, {fname})  |  J = {J}",
        xaxis_title='h', yaxis_title=flabel,
        template='plotly_dark',
        legend=dict(groupclick='toggleitem', x=1.01, xanchor='left',
                    y=1, yanchor='top',
                    bgcolor='rgba(20,20,20,0.8)',
                    bordercolor='rgba(200,200,200,0.3)', borderwidth=1),
        hovermode='x',
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Quântico forma um **retângulo**: adiabáticas horizontais (Q=cte), isocóricas verticais (h=cte).")

# ─────────────────────────────────────────────────────────────────────────────
#  ABA 3 — SUPERFÍCIE 3D INTERATIVA
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not cycle_ok:
        st.stop()

    h_lo = max(0.05, hi - 0.8)
    h_hi = hf + 0.8
    T_lo = 0.2
    T_hi = 18.0

    res = st.select_slider("Resolução da superfície", [40, 60, 80, 100, 120], value=80)
    h_3d = np.linspace(h_lo, h_hi, res)
    T_3d = np.linspace(T_lo, T_hi, res)
    H3d, T3d = np.meshgrid(h_3d, T_3d)
    Q3d = func(J, H3d, T3d)

    fig3 = go.Figure()

    # Superfície
    fig3.add_trace(go.Surface(
        x=H3d, y=T3d, z=Q3d,
        colorscale=paleta, opacity=0.50,
        colorbar=dict(title=flabel, thickness=14, len=0.55, x=1.02),
        showlegend=False,
        hovertemplate='h=%{x:.2f}<br>T=%{y:.2f}<br>Q=%{z:.3f}<extra></extra>',
    ))

    # ── Ciclo clássico ────────────────────────────────────────────────────
    for leg in ['12','23','34','41']:
        p = cam_C[leg]
        fig3.add_trace(go.Scatter3d(
            x=p['h'], y=p['T'], z=QC[leg],
            mode='lines', line=dict(width=5, color=LEG_C[leg]),
            name='Clássico (S=cte)',
            legendgroup='cicloC',
            legendgrouptitle_text='<b>Ciclos</b>',
            showlegend=(leg == '12'),
        ))
    for k, v in est_C.items():
        Qv = float(func(J, v['h'], v['T']))
        fig3.add_trace(go.Scatter3d(
            x=[v['h']], y=[v['T']], z=[Qv],
            mode='markers+text', text=[v['label']], textposition='top center',
            marker=dict(size=6, color='white', symbol='circle',
                        line=dict(color='lime', width=2)),
            name='Estados clássicos',
            legendgroup='estC',
            legendgrouptitle_text='<b>Estados</b>',
            showlegend=(k == 1),
        ))

    # ── Ciclo quântico: adiabáticas ───────────────────────────────────────
    h_12_arr = np.linspace(hi, hf, PASSOS)
    h_34_arr = np.linspace(hf, hi, PASSOS)
    for i, (xx, Tref, Qref) in enumerate([(h_12_arr, Tc, Q1), (h_34_arr, Th, Q3)]):
        fig3.add_trace(go.Scatter3d(
            x=xx, y=np.full(PASSOS, Tref), z=np.full(PASSOS, Qref),
            mode='lines', line=dict(width=4, color='cyan'),
            name='Quântico: adiabáticas',
            legendgroup='cicloQA',
            showlegend=(i == 0),
        ))

    # ── Ciclo quântico: isocóricas ────────────────────────────────────────
    t_arr    = np.linspace(0, 1, PASSOS)
    T_23_arr = np.linspace(Tc, Th, PASSOS)
    T_41_arr = np.linspace(Th, Tc, PASSOS)
    for i, (hv, Tv, Qline) in enumerate([
        (hf, T_23_arr, Q1 + t_arr * (Q3 - Q1)),
        (hi, T_41_arr, Q3 + t_arr * (Q1 - Q3)),
    ]):
        fig3.add_trace(go.Scatter3d(
            x=np.full(PASSOS, hv), y=Tv, z=Qline,
            mode='lines', line=dict(width=4, color='cyan', dash='dash'),
            name='Quântico: isocóricas',
            legendgroup='cicloQI',
            showlegend=(i == 0),
        ))

    # ── Estados quânticos ─────────────────────────────────────────────────
    for i, (key, hv, Tv, Qv) in enumerate([('1', hi, Tc, Q1), ('3', hf, Th, Q3)]):
        fig3.add_trace(go.Scatter3d(
            x=[hv], y=[Tv], z=[Qv],
            mode='markers+text', text=[f'{key}Q'], textposition='top center',
            marker=dict(size=7, color='cyan', symbol='square',
                        line=dict(color='white', width=1)),
            name='Q: equilíbrio (#1, #3)',
            legendgroup='estQeq',
            showlegend=(i == 0),
        ))
    for i, (key, hv, Tv, Qv) in enumerate([('2', hf, Tc, Q1), ('4', hi, Th, Q3)]):
        fig3.add_trace(go.Scatter3d(
            x=[hv], y=[Tv], z=[Qv],
            mode='markers+text', text=[f'{key}Q'], textposition='top center',
            marker=dict(size=8, color='rgba(0,0,0,0)', symbol='square-open',
                        line=dict(color='cyan', width=2)),
            name='Q: fora-equilíbrio (#2, #4)',
            legendgroup='estQneq',
            showlegend=(i == 0),
        ))

    fig3.update_layout(
        height=680,
        title=dict(text=f"Superfície {flabel} com ciclos de Otto  |  J = {J}", x=0.5),
        scene=dict(
            xaxis_title='h',
            yaxis_title='T',
            zaxis_title=fname,
            xaxis=dict(backgroundcolor='rgb(10,10,10)'),
            yaxis=dict(backgroundcolor='rgb(10,10,10)'),
            zaxis=dict(backgroundcolor='rgb(10,10,10)'),
            bgcolor='rgb(15,15,15)',
        ),
        template='plotly_dark',
        legend=dict(
            groupclick='togglegroup',
            tracegroupgap=10,
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            bgcolor='rgba(20,20,20,0.80)',
            bordercolor='rgba(200,200,200,0.3)',
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=0, r=120, t=60, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "**Dica:** arraste para girar · scroll para zoom · duplo clique para resetar a câmera. "
        "Clique nas entradas da legenda para ativar/desativar grupos."
    )
