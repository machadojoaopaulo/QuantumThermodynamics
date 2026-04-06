#!/usr/bin/env python3
"""
quantum_dashboard.py — Dashboard interativo QuantumThermo
=========================================================
Pré-requisito (executar uma vez):
    .venv/bin/python precompute_regions.py

Iniciar o dashboard:
    .venv/bin/python quantum_dashboard.py
    → Abrir http://127.0.0.1:8050

Fluxo:
  1. Selecione J, Modelo (Clássico/Quântico) e tipo de Mapa
  2. O mapa carrega instantaneamente (do cache pré-computado)
  3. Clique num ponto do mapa → define Tc e Th
  4. Explore o Manifold nas abas abaixo (M, S, U, F, Z, χ, C, Eff, Ciclos)
     com as visualizações Q×T, Q×h e Q×h×T (3D)
"""

import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import plotly.colors as pc
from joblib import Parallel, delayed

from thermodynamics_functions import (
    Z, energia_livre, energia_media, entropia,
    magnetizacao, susceptibilidade, calor_especifico,
    ciclo_classico, ciclo_quantico,
    ciclo_Q, ciclo_Q_quantum, ciclo_Q_quantum_termico,
    eficiencia_classica, eficiencia_quantica,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

J_VALS     = [0.00, 0.24, 0.51, 0.70, 1.00]
HI_DEF     = 1.0
HF_DEF     = 2.0
# Fallback (usado apenas se o cache não existir)
GRID_RES   = 150
GRID_STEPS = 30
TOL_S      = 0.01

MODO_NOMES = [
    "Motor", "Refrig (Tc<Th)", "Refrig (Tc>Th)",
    "Acel (Tc<Th)", "Acel (Tc>Th)", "Aquecedor",
]
MODO_IDX = {m: i for i, m in enumerate(MODO_NOMES)}

OBS_LIST = ["M", "S", "U", "F", "Z", "X", "C"]
FUNCOES = {
    "M": (magnetizacao,      "M"),
    "S": (entropia,          "S"),
    "U": (energia_media,     "U"),
    "F": (energia_livre,     "F"),
    "Z": (Z,                 "Z"),
    "X": (susceptibilidade,  "χ"),
    "C": (calor_especifico,  "C"),
}
QUANTUM_OBS = {"M", "S", "U"}

LEG_C = {"12": "#27ae60", "23": "#e74c3c", "34": "#e67e22", "41": "#2980b9"}
LEG_L = {"12": "1→2", "23": "2→3", "34": "3→4", "41": "4→1"}

SC_LABELS = {0: "S₁<S₃", 1: "S₁≈S₃", 2: "S₁>S₃"}

# Paleta qualitativa grande (fallback para forte/fmag)
_PAL = (
    pc.qualitative.Plotly + pc.qualitative.D3 +
    pc.qualitative.G10   + pc.qualitative.T10 +
    pc.qualitative.Alphabet
)

# Cores fixas por modo
_MODO_COLOR_MAP = {
    "Motor":          "#e74c3c",   # Vermelho
    "Refrig (Tc<Th)": "#2471a3",   # Azul escuro
    "Refrig (Tc>Th)": "#5dade2",   # Azul claro
    "Acel (Tc<Th)":   "#1e8449",   # Verde escuro
    "Acel (Tc>Th)":   "#58d68d",   # Verde claro
    "Aquecedor":      "#e67e22",   # Laranja
    "Não Motor":      "#7f8c8d",   # Cinza
}
# Cores para mag (Motor em tons de vermelho, Não Motor cinza)
_MAG_COLOR_MAP = {1: "#c0392b", 2: "#f1948a", 3: "#7f8c8d"}
# Cores para ent (Motor em tons de vermelho, Não Motor cinza)
_ENT_COLOR_MAP = {1: "#922b21", 2: "#e74c3c", 3: "#f5b7b1", 4: "#7f8c8d"}

# Rótulos e cores do mapa MAG CROSSING
_XMAG_LABELS = {
    1: "Sem cruzamento — M12 sempre acima",
    2: "Sem cruzamento — M12 sempre abaixo",
    3: "Cruzamento A cedo  (M1>M4, M2<M3, mid<)",
    4: "Cruzamento A tarde (M1>M4, M2<M3, mid>)",
    5: "Cruzamento B cedo  (M1<M4, M2>M3, mid>)",
    6: "Cruzamento B tarde (M1<M4, M2>M3, mid<)",
    7: "Duplo: mergulha (M1>M4, mid<, M2>M3)",
    8: "Duplo: sobe     (M1<M4, mid>, M2<M3)",
}
_XMAG_COLOR_MAP = {
    1: "#566573",   # cinza escuro  — sem cruzamento, acima
    2: "#aab7b8",   # cinza claro   — sem cruzamento, abaixo
    3: "#e74c3c",   # vermelho      — A cedo
    4: "#e67e22",   # laranja       — A tarde
    5: "#2980b9",   # azul          — B cedo
    6: "#1abc9c",   # teal          — B tarde
    7: "#8e44ad",   # roxo          — duplo mergulha
    8: "#d81b60",   # magenta       — duplo sobe
}

# Rótulos e cores do mapa FORTE (apenas Motor, código = meio-inteiro)
_FORTE_LABELS = {
    0.5:  "Motor | outros (S₁≈S₃)",
    3.5:  "Motor | S₁>S₃, M₁>M₄, M₂>M₃, M₁>M₃",
    4.5:  "Motor | S₁>S₃, M₁>M₄, M₂>M₃, M₁<M₃",
    5.5:  "Motor | S₁>S₃, M₁>M₄, M₂<M₃, M₁>M₃",
    6.5:  "Motor | S₁>S₃, M₁>M₄, M₂<M₃, M₁<M₃",
    7.5:  "Motor | S₁>S₃, M₁<M₄, M₂>M₃, M₁>M₃",
    8.5:  "Motor | S₁>S₃, M₁<M₄, M₂>M₃, M₁<M₃",
    9.5:  "Motor | S₁>S₃, M₁<M₄, M₂<M₃, M₁>M₃",
    10.5: "Motor | S₁>S₃, M₁<M₄, M₂<M₃, M₁<M₃",
    11.5: "Motor | S₁<S₃, M₁>M₄, M₂>M₃, M₁>M₃",
    12.5: "Motor | S₁<S₃, M₁>M₄, M₂<M₃, M₁>M₃",
    15.5: "Motor | S₁<S₃, M₁<M₄, M₂>M₃, M₁>M₃",
    16.5: "Motor | S₁<S₃, M₁<M₄, M₂>M₃, M₁<M₃",
    17.5: "Motor | S₁<S₃, M₁<M₄, M₂<M₃, M₁>M₃",
    18.5: "Motor | S₁<S₃, M₁<M₄, M₂<M₃, M₁<M₃",
}
# Cores: S1>S3 → tons quentes (laranja/vermelho), S1<S3 → tons frios (azul/roxo)
_FORTE_COLOR_MAP = {
    0.5:  "#95a5a6",  # cinza — outros
    3.5:  "#e74c3c",  4.5:  "#c0392b",  # S1>S3, M1>M4
    5.5:  "#e67e22",  6.5:  "#d35400",
    7.5:  "#f39c12",  8.5:  "#d4ac0d",  # S1>S3, M1<M4
    9.5:  "#f8c471", 10.5:  "#f0b27a",
    11.5: "#8e44ad", 12.5:  "#6c3483",  # S1<S3, M1>M4
    15.5: "#2980b9", 16.5:  "#1a5276",  # S1<S3, M1<M4
    17.5: "#1abc9c", 18.5:  "#148f77",
}


def _color_for_code(ptype, code):
    """Retorna a cor display para um dado ptype e código de região."""
    if ptype == "forte":
        key = round(float(code) * 2) / 2
        return _FORTE_COLOR_MAP.get(key, "#95a5a6")
    if ptype == "mag_cross":
        return _XMAG_COLOR_MAP.get(int(round(code)), "#95a5a6")
    code_i = int(round(code))
    if ptype == "mag":
        return _MAG_COLOR_MAP.get(code_i, "#7f8c8d")
    if ptype == "ent":
        return _ENT_COLOR_MAP.get(code_i, "#7f8c8d")
    if ptype == "modo":
        label = decode_code(ptype, code_i)
        return _MODO_COLOR_MAP.get(label, _PAL[code_i % len(_PAL)])
    # fmag: extrai o modo do início do rótulo
    label = decode_code(ptype, code_i)
    for mode_name, color in _MODO_COLOR_MAP.items():
        if label.startswith(mode_name):
            return color
    return _PAL[code_i % len(_PAL)]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions_cache")

# ══════════════════════════════════════════════════════════════════════════════
#  CACHE EM MEMÓRIA
# ══════════════════════════════════════════════════════════════════════════════

_GCACHE: dict = {}
_CCACHE: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO DO GRID (cache em disco → fallback em memória)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(J, hi, hf, tipo):
    return os.path.join(CACHE_DIR, f"{tipo}_J{J:.2f}_hi{hi:.1f}_hf{hf:.1f}.npz")


def load_grid(J, hi, hf, tipo):
    """Tenta carregar do disco; se não existir, computa em memória (fallback)."""
    mem_key = (round(J, 4), round(hi, 3), round(hf, 3), tipo)
    if mem_key in _GCACHE:
        return _GCACHE[mem_key], True  # (data, from_cache)

    path = _cache_path(J, hi, hf, tipo)
    if os.path.exists(path):
        npz = np.load(path)
        out = {k: npz[k] for k in npz.files}
        _GCACHE[mem_key] = out
        return out, True

    # Fallback: computa on-the-fly (baixa resolução)
    out = _compute_grid_live(J, hi, hf, tipo)
    _GCACHE[mem_key] = out
    return out, False


def _scond(S1, S3):
    eps = TOL_S * (abs(S1) + abs(S3)) / 2
    d = S1 - S3
    return 0 if d < -eps else (2 if d > eps else 1)


def _modo(Wi, Wo, Qi, Qo, Tc, Th):
    w = Wi + Wo
    if w < 0:                                       return "Motor"
    if w > 0 and Qo > 0 and Tc < Th:               return "Refrig (Tc<Th)"
    if w > 0 and Qi > 0 and Tc > Th:               return "Refrig (Tc>Th)"
    if w > 0 and Qo < 0 and Qi > 0 and Tc < Th:    return "Acel (Tc<Th)"
    if w > 0 and Qo > 0 and Qi < 0 and Tc > Th:    return "Acel (Tc>Th)"
    if w > 0 and Qi < 0 and Qo < 0:                return "Aquecedor"
    return "Indefinido"


def _pt_c(J, hi, hf, Tc, Th, P):
    try:
        _, _, Tc2, Th4 = ciclo_classico(J, hi, hf, Tc, Th, P)
        E1 = float(energia_media(J, hi, Tc))
        E2 = float(energia_media(J, hf, Tc2))
        E3 = float(energia_media(J, hf, Th))
        E4 = float(energia_media(J, hi, Th4))
        Wi, Wo = E2-E1, E4-E3
        Qi, Qo = E3-E2, E1-E4
        return dict(ok=True, modo=_modo(Wi, Wo, Qi, Qo, Tc, Th),
                    M1=float(magnetizacao(J, hi,  Tc )),
                    M2=float(magnetizacao(J, hf,  Tc2)),
                    M3=float(magnetizacao(J, hf,  Th )),
                    M4=float(magnetizacao(J, hi,  Th4)),
                    S1=float(entropia(J, hi, Tc)),
                    S3=float(entropia(J, hf, Th)))
    except Exception:
        return dict(ok=False)


def _pt_q(J, hi, hf, Tc, Th, P):
    try:
        eq, _, _, _ = ciclo_quantico(J, hi, hf, Tc, Th, P)
        E = [float(eq[k]["U"]) for k in ("1","2","3","4")]
        Wi, Wo = E[1]-E[0], E[3]-E[2]
        Qi, Qo = E[2]-E[1], E[0]-E[3]
        return dict(ok=True, modo=_modo(Wi, Wo, Qi, Qo, Tc, Th),
                    M1=float(magnetizacao(J, hi, Tc)),
                    M3=float(magnetizacao(J, hf, Th)),
                    S1=float(entropia(J, hi, Tc)),
                    S3=float(entropia(J, hf, Th)))
    except Exception:
        return dict(ok=False)


def _compute_grid_live(J, hi, hf, tipo):
    Tc_a = np.linspace(0.1, 20.0, GRID_RES)
    Th_a = np.linspace(0.1, 20.0, GRID_RES)
    Tcg, Thg = np.meshgrid(Tc_a, Th_a)
    fn = _pt_c if tipo == "classic" else _pt_q
    res = Parallel(n_jobs=-1)(
        delayed(fn)(J, hi, hf, float(tc), float(th), GRID_STEPS)
        for tc, th in zip(Tcg.flatten(), Thg.flatten())
    )
    n = len(res)
    modo_m      = np.full(n, np.nan, dtype=np.float32)
    mag_m       = np.full(n, np.nan, dtype=np.float32)
    ent_m       = np.full(n, np.nan, dtype=np.float32)
    forte_m     = np.full(n, np.nan, dtype=np.float32)
    fmag_m      = np.full(n, np.nan, dtype=np.float32)
    mag_cross_m = np.full(n, np.nan, dtype=np.float32)
    for i, r in enumerate(res):
        if not r["ok"]: continue
        mi = MODO_IDX.get(r["modo"], -1)
        if mi < 0: continue
        modo_m[i] = mi
        mc = 1 if r["M1"] > r["M3"] else 0
        sc = _scond(r["S1"], r["S3"])
        if mi == 0:  # Motor
            mag_m[i] = 1 if mc == 1 else 2
            ent_m[i] = sc + 1
        else:        # Não Motor
            mag_m[i] = 3
            ent_m[i] = 4
        if tipo == "classic" and "M2" in r:
            m14 = 1 if r["M1"] > r["M4"] else 0
            m23 = 1 if r["M2"] > r["M3"] else 0
            fmag_m[i] = mi*8 + m14*4 + m23*2 + mc + 1
            # mag_cross: só extremos (sem midpoint no live fallback → sem early/late/duplo)
            if mi == 0:
                if   m14 and not m23:  mag_cross_m[i] = 3.0   # cruzamento A
                elif not m14 and m23:  mag_cross_m[i] = 5.0   # cruzamento B
                elif m14 and m23:      mag_cross_m[i] = 1.0   # sem cruzamento (ou duplo)
                else:                  mag_cross_m[i] = 2.0   # sem cruzamento (ou duplo)
            if mi == 0:  # Motor apenas
                if sc == 2:   # S1 > S3
                    if   m14 and m23 and mc:             forte_m[i] = 3.5
                    elif m14 and m23 and not mc:          forte_m[i] = 4.5
                    elif m14 and not m23 and mc:          forte_m[i] = 5.5
                    elif m14 and not m23 and not mc:      forte_m[i] = 6.5
                    elif not m14 and m23 and mc:          forte_m[i] = 7.5
                    elif not m14 and m23 and not mc:      forte_m[i] = 8.5
                    elif not m14 and not m23 and mc:      forte_m[i] = 9.5
                    elif not m14 and not m23 and not mc:  forte_m[i] = 10.5
                    else:                                 forte_m[i] = 0.5
                elif sc == 0:  # S1 < S3
                    if   m14 and m23 and mc:             forte_m[i] = 11.5
                    elif m14 and not m23 and mc:          forte_m[i] = 12.5
                    elif not m14 and m23 and mc:          forte_m[i] = 15.5
                    elif not m14 and m23 and not mc:      forte_m[i] = 16.5
                    elif not m14 and not m23 and mc:      forte_m[i] = 17.5
                    elif not m14 and not m23 and not mc:  forte_m[i] = 18.5
                    else:                                 forte_m[i] = 0.5
                else:          # S1 ≈ S3
                    forte_m[i] = 0.5
    sh = Tcg.shape
    return dict(Tc=Tc_a, Th=Th_a,
                modo=modo_m.reshape(sh),      mag=mag_m.reshape(sh),
                ent=ent_m.reshape(sh),         forte=forte_m.reshape(sh),
                fmag=fmag_m.reshape(sh),       mag_cross=mag_cross_m.reshape(sh))


# ══════════════════════════════════════════════════════════════════════════════
#  DECODIFICAÇÃO DE REGIÕES → RÓTULOS LEGÍVEIS
# ══════════════════════════════════════════════════════════════════════════════

def decode_code(ptype, code):
    """Converte código de região em rótulo legível."""
    if ptype == "forte":
        key = round(float(code) * 2) / 2
        return _FORTE_LABELS.get(key, f"Forte {code}")

    code = int(round(code))

    if ptype == "modo":
        return MODO_NOMES[code] if 0 <= code < len(MODO_NOMES) else f"Modo {code}"

    if ptype == "mag_cross":
        return _XMAG_LABELS.get(code, f"XMag {code}")

    if ptype == "mag":
        # code 1 = Motor | M₁>M₃,  2 = Motor | M₁<M₃,  3 = Não Motor
        if code == 1: return "Motor | M₁>M₃"
        if code == 2: return "Motor | M₁<M₃"
        if code == 3: return "Não Motor"
        return f"Mag {code}"

    if ptype == "ent":
        # code 1 = Motor | S₁<S₃,  2 = Motor | S₁≈S₃,  3 = Motor | S₁>S₃,  4 = Não Motor
        if code == 1: return "Motor | S₁<S₃"
        if code == 2: return "Motor | S₁≈S₃"
        if code == 3: return "Motor | S₁>S₃"
        if code == 4: return "Não Motor"
        return f"Ent {code}"

    if ptype == "fmag":
        # code = mi*8 + m14*4 + m23*2 + m13 + 1
        mi   = (code - 1) // 8
        rest = (code - 1) % 8
        m14  = rest // 4
        m23  = (rest % 4) // 2
        m13  = rest % 2
        mn = MODO_NOMES[mi] if 0 <= mi < len(MODO_NOMES) else f"Modo{mi}"
        return (f"{mn} | M₁{'>'if m14 else '<'}M₄"
                f", M₂{'>'if m23 else '<'}M₃"
                f", M₁{'>'if m13 else '<'}M₃")

    return str(code)


# ══════════════════════════════════════════════════════════════════════════════
#  CICLO (manifolds)
# ══════════════════════════════════════════════════════════════════════════════

def get_cycle(J, hi, hf, Tc, Th, P=100):
    key = (round(J,4), round(hi,3), round(hf,3), round(Tc,3), round(Th,3), P)
    if key in _CCACHE:
        return _CCACHE[key]
    try:
        eC, cC, Tc2, Th4 = ciclo_classico(J, hi, hf, Tc, Th, P)
        eQ, cQ, _,   _   = ciclo_quantico( J, hi, hf, Tc, Th, P)
        out = dict(ok=True, eC=eC, cC=cC, Tc2=Tc2, Th4=Th4, eQ=eQ, cQ=cQ)
    except Exception as e:
        out = dict(ok=False, err=str(e))
    _CCACHE[key] = out
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUTORES DE FIGURAS – MANIFOLDS
# ══════════════════════════════════════════════════════════════════════════════

def _dark(height=430):
    return dict(template="plotly_dark", height=height,
                legend=dict(orientation="h", y=-0.22, font_size=10,
                            groupclick="toggleitem"),
                margin=dict(l=55, r=10, t=35, b=65))


def fig_QxT(obs, J, hi, hf, Tc, Th):
    func, ylabel = FUNCOES[obs]
    T_arr = np.linspace(0.15, 25, 800)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_arr, y=func(J, hi, T_arr),
                             name=f"h = {hi}", line=dict(color="#5dade2", width=2)))
    fig.add_trace(go.Scatter(x=T_arr, y=func(J, hf, T_arr),
                             name=f"h = {hf}", line=dict(color="#e74c3c", width=2)))
    cyc = get_cycle(J, hi, hf, Tc, Th)
    if cyc["ok"]:
        for k in (1, 2, 3, 4):
            s = cyc["eC"][k]
            fig.add_trace(go.Scatter(
                x=[s["T"]], y=[float(func(J, s["h"], s["T"]))],
                mode="markers+text",
                marker=dict(size=11, color="#ecf0f1",
                            line=dict(color="#2c3e50", width=1.5)),
                text=[s["label"]], textposition="top center", textfont_size=10,
                name=s["label"], legendgroup="classic",
                legendgrouptitle_text="Clássico" if k == 1 else None,
            ))
        # Q1, Q3 = equilíbrio (T definido) → marcados na curva
        for k in ("1", "3"):
            s = cyc["eQ"][k]
            fig.add_trace(go.Scatter(
                x=[s["T"]], y=[float(func(J, s["h"], s["T"]))],
                mode="markers+text",
                marker=dict(size=12, symbol="diamond", color="#9b59b6",
                            line=dict(color="white", width=1.5)),
                text=[s["label"]], textposition="bottom center", textfont_size=10,
                name=s["label"], legendgroup="Qstates",
                legendgrouptitle_text="Estados Q" if k == "1" else None,
            ))
        # Q2, Q4 = fora do equilíbrio (T=None) → anotação lateral
        if obs in QUANTUM_OBS:
            eQ = cyc["eQ"]
            for k, xpos in (("2", 0.98), ("4", 0.98)):
                s  = eQ[k]
                yv = float(s[obs])
                fig.add_annotation(
                    x=xpos, y=yv, xref="paper", yref="y",
                    text=f"{s['label']} (fora-equil.)",
                    showarrow=True, arrowhead=2, ax=40, ay=0,
                    font=dict(color="#9b59b6", size=10),
                    arrowcolor="#9b59b6",
                )
    fig.update_layout(xaxis_title="T", yaxis_title=ylabel,
                      title=f"{obs} × T   (J={J:.2f}, hi={hi}, hf={hf})", **_dark())
    return fig


def fig_Qxh(obs, J, hi, hf, Tc, Th):
    func, ylabel = FUNCOES[obs]
    h_arr = np.linspace(max(0.15, hi - 0.7), hf + 0.7, 500)
    T_iso = [0.5, 1, 2, 3, 5, 8, 12, 20]
    cols  = pc.sample_colorscale("RdBu", np.linspace(0, 1, len(T_iso)))
    fig = go.Figure()
    for Ti, col in zip(T_iso, cols):
        fig.add_trace(go.Scatter(x=h_arr, y=func(J, h_arr, Ti), mode="lines",
                                 line=dict(color=col, width=1, dash="dot"),
                                 name=f"T={Ti}", legendgroup="iso",
                                 legendgrouptitle_text="Iso-T" if Ti == T_iso[0] else None))
    cyc = get_cycle(J, hi, hf, Tc, Th)
    if cyc["ok"]:
        cC = cyc["cC"]
        QC = ciclo_Q(func, J, cC)
        for leg in ("12", "23", "34", "41"):
            fig.add_trace(go.Scatter(
                x=cC[leg]["h"], y=QC[leg], mode="lines",
                name=f"C {LEG_L[leg]}", line=dict(color=LEG_C[leg], width=2.5),
                legendgroup="classic",
                legendgrouptitle_text="Clássico" if leg == "12" else None))
        cQ = cyc["cQ"]
        QQ = ciclo_Q_quantum(obs, cQ) if obs in QUANTUM_OBS else ciclo_Q_quantum_termico(func, J, cQ)
        for leg in ("12", "23", "34", "41"):
            if QQ[leg] is None: continue
            fig.add_trace(go.Scatter(
                x=cQ[leg]["h"], y=QQ[leg], mode="lines",
                name=f"Q {LEG_L[leg]}", line=dict(color=LEG_C[leg], width=2, dash="dash"),
                legendgroup="quantum",
                legendgrouptitle_text="Quântico" if leg == "12" else None))
        # Estados clássicos: C1, C2, C3, C4
        for k in (1, 2, 3, 4):
            s = cyc["eC"][k]
            fig.add_trace(go.Scatter(
                x=[s["h"]], y=[float(func(J, s["h"], s["T"]))],
                mode="markers+text",
                marker=dict(size=12, color="#ecf0f1",
                            line=dict(color="#2c3e50", width=1.5)),
                text=[s["label"]], textposition="top center", textfont_size=10,
                name=s["label"], legendgroup="Cstates",
                legendgrouptitle_text="Estados C" if k == 1 else None))

        # Estados quânticos: Q1, Q2, Q3, Q4
        # Q1, Q3 = equilíbrio (T definido); Q2, Q4 = fora do equilíbrio (T=None)
        eQ = cyc["eQ"]
        for k in ("1", "2", "3", "4"):
            s = eQ[k]
            if obs in QUANTUM_OBS:
                y_val = float(s[obs])          # M/S/U direto do estado congelado
                symbol = ("diamond" if s["T"] is not None else "diamond-open")
            else:
                if s["T"] is None:
                    continue                    # F/Z/X/C indefinido fora do equilíbrio
                y_val  = float(func(J, s["h"], s["T"]))
                symbol = "diamond"
            tpos = "bottom center" if k in ("2", "4") else "top center"
            fig.add_trace(go.Scatter(
                x=[s["h"]], y=[y_val],
                mode="markers+text",
                marker=dict(size=12, symbol=symbol, color="#9b59b6",
                            line=dict(color="white", width=1.5)),
                text=[s["label"]], textposition=tpos, textfont_size=10,
                name=s["label"], legendgroup="Qstates",
                legendgrouptitle_text="Estados Q" if k == "1" else None))

    fig.update_layout(xaxis_title="h", yaxis_title=ylabel,
                      title=f"{obs} × h   (J={J:.2f}, Tc={Tc:.2f}, Th={Th:.2f})", **_dark())
    return fig




def fig_Qxhxt(obs, J, hi, hf, Tc, Th):
    func, ylabel = FUNCOES[obs]
    h_s = np.linspace(max(0.15, hi - 0.5), hf + 0.5, 60)
    T_s = np.linspace(0.2, 20, 60)
    H, TT = np.meshgrid(h_s, T_s)
    Zsurf = func(J, H, TT)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=H, y=TT, z=Zsurf, colorscale="plasma",
                              opacity=0.82, showscale=True, name="manifold",
                              lighting=dict(diffuse=0.8, specular=0.1)))
    cyc = get_cycle(J, hi, hf, Tc, Th)
    if cyc["ok"]:
        cC = cyc["cC"]
        QC = ciclo_Q(func, J, cC)
        # ── Caminhos clássicos ──────────────────────────────────────────────
        for leg in ("12", "23", "34", "41"):
            p = cC[leg]
            fig.add_trace(go.Scatter3d(x=p["h"], y=p["T"], z=QC[leg], mode="lines",
                                       name=f"C {LEG_L[leg]}",
                                       line=dict(color=LEG_C[leg], width=5),
                                       legendgroup="classic",
                                       legendgrouptitle_text="Clássico" if leg == "12" else None))
        # ── Estados clássicos C1–C4 ─────────────────────────────────────────
        for k in (1, 2, 3, 4):
            s = cyc["eC"][k]
            z_val = float(func(J, s["h"], s["T"]))
            fig.add_trace(go.Scatter3d(
                x=[s["h"]], y=[s["T"]], z=[z_val],
                mode="markers+text",
                marker=dict(size=7, color="#ecf0f1", symbol="circle",
                            line=dict(color="#2c3e50", width=1.5)),
                text=[s["label"]], textposition="top center",
                name=s["label"], legendgroup="Cstates",
                legendgrouptitle_text="Estados C" if k == 1 else None))

        cQ = cyc["cQ"]
        QQ = ciclo_Q_quantum(obs, cQ) if obs in QUANTUM_OBS else ciclo_Q_quantum_termico(func, J, cQ)
        eQ = cyc["eQ"]

        if obs in QUANTUM_OBS:
            # Posições 3D dos 4 estados quânticos:
            #   Q1 = (hi, Tc, Q1_val)  — equilíbrio
            #   Q2 = (hf, Tc, Q2_val)  — T=Tc: última temperatura definida (saiu de #1)
            #   Q3 = (hf, Th, Q3_val)  — equilíbrio
            #   Q4 = (hi, Th, Q4_val)  — T=Th: última temperatura definida (saiu de #3)
            obs1 = float(eQ["1"][obs])
            obs2 = float(eQ["2"][obs])   # = obs1 para M/S; U varia com h
            obs3 = float(eQ["3"][obs])
            obs4 = float(eQ["4"][obs])   # = obs3 para M/S; U varia com h

            pos = {
                "1": (hi, float(Tc), obs1),
                "2": (hf, float(Tc), obs2),  # T=Tc: herdado da adiabática 1→2
                "3": (hf, float(Th), obs3),
                "4": (hi, float(Th), obs4),  # T=Th: herdado da adiabática 3→4
            }

            # ── Adiabáticas: retas em planos T=Tc e T=Th ────────────────────
            # M e S: z=const (horizontal). U: z varia linearmente com h.
            for s_k, e_k, leg in (("1", "2", "12"), ("3", "4", "34")):
                x0, y0, z0 = pos[s_k]
                x1, y1, z1 = pos[e_k]
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                    name=f"Q {LEG_L[leg]}",
                    line=dict(color=LEG_C[leg], width=4),
                    legendgroup="quantum",
                    legendgrouptitle_text="Quântico" if leg == "12" else None))

            # ── Isocóricas: linhas pontilhadas retas (caminho real desconhecido)
            for s_k, e_k, leg in (("2", "3", "23"), ("4", "1", "41")):
                x0, y0, z0 = pos[s_k]
                x1, y1, z1 = pos[e_k]
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                    name=f"Q {LEG_L[leg]}",
                    line=dict(color=LEG_C[leg], width=4, dash="dash"),
                    legendgroup="quantum"))

            # ── Estados Q1–Q4 ────────────────────────────────────────────────
            for k in ("1", "2", "3", "4"):
                x, y, z = pos[k]
                s = eQ[k]
                symbol = "diamond" if s["T"] is not None else "diamond-open"
                tpos   = "bottom center" if k in ("2", "4") else "top center"
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode="markers+text",
                    marker=dict(size=7, symbol=symbol, color="#9b59b6",
                                line=dict(color="white", width=1.5)),
                    text=[s["label"]], textposition=tpos,
                    name=s["label"], legendgroup="Qstates",
                    legendgrouptitle_text="Estados Q" if k == "1" else None))
        else:
            # F, Z, X, C — adiabáticas sem T definido → só Q1 e Q3 marcados
            for k in ("1", "3"):
                s = eQ[k]
                z_val = float(func(J, s["h"], float(s["T"])))
                fig.add_trace(go.Scatter3d(
                    x=[s["h"]], y=[float(s["T"])], z=[z_val],
                    mode="markers+text",
                    marker=dict(size=7, symbol="diamond", color="#9b59b6",
                                line=dict(color="white", width=1.5)),
                    text=[s["label"]], textposition="top center",
                    name=s["label"], legendgroup="Qstates",
                    legendgrouptitle_text="Estados Q" if k == "1" else None))
    fig.update_layout(
        scene=dict(xaxis_title="h", yaxis_title="T", zaxis_title=ylabel),
        title=f"{obs} × h × T   (J={J:.2f}, Tc={Tc:.2f}, Th={Th:.2f})",
        template="plotly_dark", height=530,
        legend=dict(orientation="h", y=-0.05, font_size=10),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_eff(J, hi, hf, Tc, Th):
    fig = go.Figure()
    try:
        ec = eficiencia_classica(J, hi, hf, Tc, Th)
        eq = eficiencia_quantica(J, hi, hf, Tc, Th)
        cats = ["W", "Qin", "Qout", "η"]
        vc = [ec["W"], ec["Qin"], ec["Qout"], ec["eta"]]
        vq = [eq["W"], eq["Qin"], eq["Qout"], eq["eta"]]
        fig.add_trace(go.Bar(name="Clássico", x=cats, y=vc, marker_color="#5dade2",
                             text=[f"{v:.4f}" for v in vc], textposition="outside"))
        fig.add_trace(go.Bar(name="Quântico",  x=cats, y=vq, marker_color="#9b59b6",
                             text=[f"{v:.4f}" for v in vq], textposition="outside"))
        fig.update_layout(barmode="group",
                          title=(f"Eficiência   J={J:.2f}, Tc={Tc:.2f}, Th={Th:.2f}<br>"
                                 f"<sup>Clássico: {ec['modo']}  |  Quântico: {eq['modo']}</sup>"),
                          template="plotly_dark", height=420,
                          legend=dict(orientation="h", y=-0.15))
    except Exception as e:
        fig.add_annotation(text=f"Erro: {e}", showarrow=False,
                           font_size=14, font_color="tomato")
        fig.update_layout(template="plotly_dark", height=420)
    return fig


def fig_ciclos(J, hi, hf, Tc, Th):
    return fig_Qxh("M", J, hi, hf, Tc, Th)


# ══════════════════════════════════════════════════════════════════════════════
#  COLORSCALE DISCRETA + LEGEND PROXIES
# ══════════════════════════════════════════════════════════════════════════════

def _remap_z(z, vals):
    """Remap valores originais de z para índices consecutivos 0..n-1."""
    val_to_idx = {v: float(i) for i, v in enumerate(vals)}
    out = np.full(z.shape, np.nan, dtype=np.float32)
    for v, idx in val_to_idx.items():
        out[z == v] = idx
    return out


def _discrete_cs(z_flat, ptype="modo"):
    """Retorna (colorscale, vals, colors_per_val) para um Heatmap do Plotly.

    O colorscale é construído para valores remapeados 0..n-1 com zmin=-0.5,
    zmax=n-0.5, garantindo bandas uniformes sem interpolação espúria de cor.
    """
    vals = sorted(v for v in np.unique(z_flat) if not np.isnan(v))
    if not vals:
        return [[0, "#333"], [1, "#333"]], [], []

    colors = [_color_for_code(ptype, v) for v in vals]
    n = len(vals)

    if n == 1:
        c = colors[0]
        return [[0, c], [1, c]], vals, colors

    # Colorscale uniforme para índices 0..n-1:
    # valor remapeado i → posição (i+0.5)/n, banda [i/n, (i+1)/n)
    cs = []
    for i, c in enumerate(colors):
        cs.append([i / n, c])
        if i < n - 1:
            cs.append([(i + 1) / n - 1e-9, c])
    cs.append([1.0, colors[-1]])
    cs[0][0]  = 0.0
    cs[-1][0] = 1.0
    return cs, vals, colors


def _legend_proxy_traces(ptype, vals, colors):
    """Cria traces invisíveis (proxy) para a legenda do heatmap."""
    traces = []
    for v, col in zip(vals, colors):
        label = decode_code(ptype, v)
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=13, color=col),
            name=label,
            showlegend=True,
            legendgroup=f"region_{int(v)}",
        ))
    return traces


# ══════════════════════════════════════════════════════════════════════════════
#  DASH APP — LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

BG   = "#181825"
BG2  = "#1e1e2e"
BG3  = "#313244"
COL  = "#cdd6f4"
ACC  = "#89b4fa"
ACC2 = "#cba6f7"
GRN  = "#a6e3a1"
MUT  = "#6c7086"

def _rb():
    return dict(display="block", padding="2px 0", fontSize="13px", color=COL)


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "QuantumThermo Dashboard"

app.layout = html.Div(
    style=dict(display="flex", fontFamily="'Inter','Segoe UI',sans-serif",
               background=BG, minHeight="100vh", color=COL),
    children=[

        # ── SIDEBAR ───────────────────────────────────────────────────────
        html.Div(
            style=dict(width="230px", minWidth="230px", background=BG2,
                       padding="16px 14px", height="100vh", overflowY="auto",
                       flexShrink=0, borderRight=f"1px solid {BG3}",
                       boxSizing="border-box"),
            children=[
                html.H3("⚛ QuantumThermo",
                        style=dict(color=ACC2, margin="0 0 18px", fontSize="15px", fontWeight="700")),

                html.P("Acoplamento J",
                       style=dict(color=ACC, fontSize="11px", fontWeight="700",
                                  margin="0 0 5px", textTransform="uppercase")),
                dcc.RadioItems(id="sel-J",
                    options=[{"label": f"  J = {j:.2f}", "value": j} for j in J_VALS],
                    value=0.70, labelStyle=_rb(), inputStyle=dict(marginRight="6px")),

                html.Hr(style=dict(borderColor=BG3, margin="14px 0")),
                html.P("Modelo",
                       style=dict(color=ACC, fontSize="11px", fontWeight="700",
                                  margin="0 0 5px", textTransform="uppercase")),
                dcc.RadioItems(id="sel-tipo",
                    options=[{"label": "  Clássico", "value": "classic"},
                             {"label": "  Quântico",  "value": "quantum"}],
                    value="classic", labelStyle=_rb(), inputStyle=dict(marginRight="6px")),

                html.Hr(style=dict(borderColor=BG3, margin="14px 0")),
                html.P("Mapa de Regiões",
                       style=dict(color=ACC, fontSize="11px", fontWeight="700",
                                  margin="0 0 5px", textTransform="uppercase")),
                dcc.RadioItems(id="sel-ptype",
                    options=[
                        {"label": "  Modos",        "value": "modo"},
                        {"label": "  Mag",           "value": "mag"},
                        {"label": "  Ent",           "value": "ent"},
                        {"label": "  Mag Forte",     "value": "fmag"},
                        {"label": "  Forte",         "value": "forte"},
                        {"label": "  Mag Crossing",  "value": "mag_cross"},
                    ],
                    value="modo", labelStyle=_rb(), inputStyle=dict(marginRight="6px")),
                html.Div(id="ptype-note",
                         style=dict(fontSize="11px", color=MUT, marginTop="4px", lineHeight="1.4")),

                html.Hr(style=dict(borderColor=BG3, margin="14px 0")),
                html.P("Campo magnético",
                       style=dict(color=ACC, fontSize="11px", fontWeight="700",
                                  margin="0 0 5px", textTransform="uppercase")),
                html.Div(style=dict(display="flex", gap="8px", marginBottom="10px"), children=[
                    html.Div([
                        html.Label("hi", style=dict(fontSize="11px", color=MUT,
                                                     display="block", marginBottom="2px")),
                        dcc.Input(id="inp-hi", type="number", value=HI_DEF,
                                  step=0.1, debounce=True,
                                  style=dict(width="68px", background=BG3, color=COL,
                                             border=f"1px solid {BG3}",
                                             borderRadius="4px", padding="5px")),
                    ]),
                    html.Div([
                        html.Label("hf", style=dict(fontSize="11px", color=MUT,
                                                     display="block", marginBottom="2px")),
                        dcc.Input(id="inp-hf", type="number", value=HF_DEF,
                                  step=0.1, debounce=True,
                                  style=dict(width="68px", background=BG3, color=COL,
                                             border=f"1px solid {BG3}",
                                             borderRadius="4px", padding="5px")),
                    ]),
                ]),

                html.Hr(style=dict(borderColor=BG3, margin="14px 0")),
                html.P("Ponto selecionado",
                       style=dict(color=ACC, fontSize="11px", fontWeight="700",
                                  margin="0 0 5px", textTransform="uppercase")),
                html.Div(id="disp-tcth",
                         style=dict(background=BG3, borderRadius="6px",
                                    padding="8px 10px", fontFamily="monospace",
                                    fontSize="13px", color=ACC2, whiteSpace="pre-line",
                                    lineHeight="1.8"),
                         children="Clique no mapa →"),
            ],
        ),

        # ── ÁREA PRINCIPAL (coluna única, scroll vertical) ────────────────
        html.Div(
            style=dict(flex=1, padding="14px 16px", overflowY="auto",
                       display="flex", flexDirection="column", gap="16px"),
            children=[

                # ── Regiões (topo, largura total) ─────────────────────────
                html.Div([
                    html.H4(id="title-reg",
                            style=dict(color=COL, margin="0 0 8px", fontSize="13px", fontWeight="600"),
                            children="Regiões"),
                    html.Div(id="cache-status",
                             style=dict(fontSize="11px", color=MUT, marginBottom="6px")),
                    dcc.Loading(type="dot", color=ACC,
                                children=dcc.Graph(
                                    id="graph-reg",
                                    style=dict(height="520px"),
                                    config=dict(displayModeBar=False),
                                )),
                ]),

                # ── Manifold (abaixo, inicialmente oculto) ────────────────
                html.Div(
                    id="mf-section",
                    style=dict(display="none"),
                    children=[
                        html.H4(id="mf-title",
                                style=dict(color=COL, margin="0 0 4px",
                                           fontSize="13px", fontWeight="600")),
                        html.Div(id="cyc-summary",
                                 style=dict(fontSize="11px", color=GRN,
                                            background=BG2, padding="6px 10px",
                                            borderRadius="4px", marginBottom="10px",
                                            whiteSpace="pre-line", lineHeight="1.7",
                                            fontFamily="monospace")),
                        dcc.Tabs(
                            id="mf-tabs", value="M",
                            colors=dict(border=BG3, primary=ACC, background=BG2),
                            children=[
                                dcc.Tab(label=o, value=o,
                                        style=dict(padding="5px 10px", color=MUT,
                                                   fontSize="13px", background=BG2,
                                                   border=f"1px solid {BG3}"),
                                        selected_style=dict(padding="5px 10px",
                                                            color=BG2, fontSize="13px",
                                                            background=ACC,
                                                            border=f"1px solid {ACC}",
                                                            fontWeight="700"))
                                for o in OBS_LIST + ["Eff", "Ciclos"]
                            ],
                        ),
                        html.Div(id="view-radio-box",
                                 style=dict(margin="10px 0 6px"),
                                 children=dcc.RadioItems(
                                     id="radio-view",
                                     value="QxT",
                                     options=[],
                                     inline=True,
                                     labelStyle=dict(marginRight="18px", color=COL, fontSize="13px"),
                                     inputStyle=dict(marginRight="5px"),
                                 )),
                        dcc.Loading(type="dot", color=ACC,
                                    children=dcc.Graph(id="graph-mf",
                                                       style=dict(height="500px"))),
                    ],
                ),

            ],
        ),

        # ── STORES ────────────────────────────────────────────────────────
        dcc.Store(id="store-key"),
        dcc.Store(id="store-tcth"),
        dcc.Store(id="store-mfp"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# 1 ─ Atualizar regiões ao mudar J / tipo / hi / hf ───────────────────────────
@app.callback(
    Output("store-key",      "data"),
    Output("title-reg",      "children"),
    Output("cache-status",   "children"),
    Input("sel-J",    "value"),
    Input("sel-tipo", "value"),
    Input("inp-hi",   "value"),
    Input("inp-hf",   "value"),
)
def cb_update_key(J, tipo, hi, hf):
    hi = float(hi or HI_DEF)
    hf = float(hf or HF_DEF)
    _, from_cache = load_grid(J, hi, hf, tipo)
    key   = json.dumps(dict(J=J, hi=hi, hf=hf, tipo=tipo))
    title = f"Regiões — J = {J:.2f},  hi = {hi:.1f},  hf = {hf:.1f}  ({tipo})"
    status = ("✓ Cache carregado" if from_cache
              else "⚠ Cache não encontrado — use precompute_regions.py para alta resolução")
    return key, title, status


# 2 ─ Renderizar heatmap ───────────────────────────────────────────────────────
DISPLAY_RES = 600   # resolução máxima exibida no browser

@app.callback(
    Output("graph-reg",  "figure"),
    Input("store-key",   "data"),
    Input("sel-ptype",   "value"),
    Input("store-tcth",  "data"),
)
def cb_regions(key_json, ptype, tcth_json):
    if not key_json:
        return go.Figure(layout=dict(template="plotly_dark"))

    k = json.loads(key_json)
    J, hi, hf, tipo = k["J"], k["hi"], k["hf"], k["tipo"]
    g, _ = load_grid(J, hi, hf, tipo)

    # Fallback para quântico sem forte/fmag
    if ptype in ("forte", "fmag") and tipo == "quantum":
        ptype = "mag"

    z_full = {"modo": g["modo"], "mag": g["mag"], "ent": g["ent"],
              "forte": g["forte"], "fmag": g["fmag"],
              "mag_cross": g.get("mag_cross", np.full_like(g["modo"], np.nan))
             }.get(ptype, g["modo"])

    # Downsample para exibição (o browser não precisa de 3000×3000)
    # z_full tem shape (N_Tc, N_Th); Plotly Heatmap espera z[j][i] = valor em (x[i], y[j])
    # → transpõe para shape (N_Th, N_Tc) com x=Tc, y=Th
    N = z_full.shape[0]
    if N > DISPLAY_RES:
        step = N // DISPLAY_RES
        z    = z_full[::step, ::step].T   # (N_Th', N_Tc')
        Tc_d = g["Tc"][::step]
        Th_d = g["Th"][::step]
    else:
        z    = z_full.T
        Tc_d = g["Tc"]
        Th_d = g["Th"]

    cs, vals, colors = _discrete_cs(z.flatten(), ptype)
    if not vals:
        return go.Figure(layout=dict(template="plotly_dark"))

    # Remap z para índices consecutivos → colorscale com bandas uniformes
    z_display = _remap_z(z, vals)
    zmin, zmax = -0.5, len(vals) - 0.5

    fig = go.Figure()

    # Heatmap principal (sem colorbar — usamos a legenda)
    fig.add_trace(go.Heatmap(
        x=Tc_d, y=Th_d, z=z_display,
        colorscale=cs, zmin=zmin, zmax=zmax,
        showscale=False,
        hovertemplate="Tc = %{x:.3f}<br>Th = %{y:.3f}<br>%{customdata}<extra></extra>",
        customdata=[[decode_code(ptype, v) if not np.isnan(v) else "—"
                     for v in row] for row in z],
        zsmooth=False,
    ))

    # Diagonal Tc = Th
    td = np.array([0.1, 20.0])
    fig.add_trace(go.Scatter(x=td, y=td, mode="lines",
                             line=dict(color="white", width=1.5, dash="dash"),
                             showlegend=False, hoverinfo="skip"))

    # Marcador do ponto selecionado
    if tcth_json:
        d = json.loads(tcth_json)
        fig.add_trace(go.Scatter(
            x=[d["Tc"]], y=[d["Th"]], mode="markers",
            marker=dict(symbol="x-open", size=16, color="white",
                        line=dict(color="white", width=2.5)),
            showlegend=False,
            hovertemplate=f"Tc={d['Tc']:.3f}<br>Th={d['Th']:.3f}<extra>Selecionado</extra>",
        ))

    # Proxy traces para a legenda
    for trace in _legend_proxy_traces(ptype, vals, colors):
        fig.add_trace(trace)

    fig.update_layout(
        xaxis_title="Tc", yaxis_title="Th",
        template="plotly_dark", height=510,
        xaxis=dict(range=[0.1, 20]),
        yaxis=dict(range=[0.1, 20]),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            xanchor="left", yanchor="top",
            font_size=11,
            bgcolor="rgba(30,30,46,0.85)",
            bordercolor=BG3, borderwidth=1,
            itemclick="toggle",
        ),
        margin=dict(l=55, r=10, t=10, b=55),
    )
    return fig


# 3 ─ Clique no mapa → Tc, Th ─────────────────────────────────────────────────
@app.callback(
    Output("store-tcth", "data"),
    Output("disp-tcth",  "children"),
    Input("graph-reg",   "clickData"),
    prevent_initial_call=True,
)
def cb_click(click):
    if not click:
        return no_update, no_update
    pt = click["points"][0]
    Tc, Th = float(pt["x"]), float(pt["y"])
    return json.dumps(dict(Tc=Tc, Th=Th)), f"Tc = {Tc:.3f}\nTh = {Th:.3f}"


# 4 ─ Mostrar manifold ─────────────────────────────────────────────────────────
@app.callback(
    Output("mf-section",  "style"),
    Output("mf-title",    "children"),
    Output("cyc-summary", "children"),
    Output("store-mfp",   "data"),
    Input("store-tcth",   "data"),
    State("store-key",    "data"),
    prevent_initial_call=True,
)
def cb_show_mf(tcth_json, key_json):
    if not tcth_json or not key_json:
        return dict(display="none"), "", "", None
    d  = json.loads(tcth_json)
    k  = json.loads(key_json)
    Tc, Th    = d["Tc"], d["Th"]
    J, hi, hf = k["J"], k["hi"], k["hf"]
    try:
        ec = eficiencia_classica(J, hi, hf, Tc, Th)
        eq = eficiencia_quantica(J, hi, hf, Tc, Th)
        summary = (f"Clássico  modo={ec['modo']}   W={ec['W']:+.4f}   η={ec['eta']:.4f}\n"
                   f"Quântico  modo={eq['modo']}   W={eq['W']:+.4f}   η={eq['eta']:.4f}")
    except Exception as e:
        summary = f"Erro ao calcular ciclo: {e}"
    title = (f"Manifolds — J={J:.2f}, hi={hi}, hf={hf}  "
             f"| Tc={Tc:.3f}, Th={Th:.3f}")
    mfp = dict(J=J, hi=hi, hf=hf, Tc=Tc, Th=Th)
    return dict(display="block"), title, summary, mfp


# 5 ─ Opções do radio de visualização ─────────────────────────────────────────
@app.callback(
    Output("radio-view",     "options"),
    Output("radio-view",     "value"),
    Output("view-radio-box", "style"),
    Input("mf-tabs",         "value"),
)
def cb_radio_opts(obs):
    if obs in ("Eff", "Ciclos"):
        return [], "QxT", dict(display="none", margin="10px 0 6px")
    opts = [{"label": f"  {obs}×T",         "value": "QxT"},
            {"label": f"  {obs}×h",         "value": "Qxh"},
            {"label": f"  {obs}×h×T (3D)",  "value": "QxhxT"}]
    return opts, "QxT", dict(display="block", margin="10px 0 6px")


# 6 ─ Figura do manifold ───────────────────────────────────────────────────────
@app.callback(
    Output("graph-mf",   "figure"),
    Input("mf-tabs",     "value"),
    Input("radio-view",  "value"),
    State("store-mfp",   "data"),
)
def cb_mf_fig(obs, view, mfp):
    if not mfp:
        return go.Figure(layout=dict(template="plotly_dark"))
    J, hi, hf = mfp["J"], mfp["hi"], mfp["hf"]
    Tc, Th    = mfp["Tc"], mfp["Th"]
    if obs == "Eff":    return fig_eff(J, hi, hf, Tc, Th)
    if obs == "Ciclos": return fig_ciclos(J, hi, hf, Tc, Th)
    if   view == "QxT":   return fig_QxT(obs, J, hi, hf, Tc, Th)
    elif view == "Qxh":   return fig_Qxh(obs, J, hi, hf, Tc, Th)
    else:                  return fig_Qxhxt(obs, J, hi, hf, Tc, Th)


# 7 ─ Nota sobre Mag Forte no modo quântico ───────────────────────────────────
@app.callback(
    Output("ptype-note",  "children"),
    Input("sel-ptype",    "value"),
    Input("sel-tipo",     "value"),
)
def cb_ptype_note(ptype, tipo):
    if tipo == "quantum" and ptype in ("forte", "fmag"):
        return "⚠ Forte/Mag Forte disponível apenas para Clássico. Exibindo Mag."
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pré-carrega grids em memória (dos arquivos de cache) ao iniciar
    print("\n" + "=" * 54)
    print("  ⚛  QuantumThermo Dashboard")
    print("  → http://127.0.0.1:8050")
    print("=" * 54)
    print("\nCarregando caches disponíveis...")
    loaded = 0
    for tipo in ("classic", "quantum"):
        for J in J_VALS:
            path = _cache_path(J, HI_DEF, HF_DEF, tipo)
            if os.path.exists(path):
                load_grid(J, HI_DEF, HF_DEF, tipo)
                loaded += 1
    print(f"  {loaded} grids carregados de {CACHE_DIR}/")
    if loaded == 0:
        print("  ⚠  Nenhum cache encontrado.")
        print("     Execute: .venv/bin/python precompute_regions.py")
    print()
    app.run(debug=False, port=8050)
