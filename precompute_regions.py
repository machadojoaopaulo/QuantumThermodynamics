#!/usr/bin/env python3
"""
precompute_regions.py — Pré-computa os mapas de regiões em alta resolução
==========================================================================
Executar UMA VEZ antes de usar o dashboard:

    .venv/bin/python precompute_regions.py

Os resultados são salvos em  regions_cache/  e carregados instantaneamente
pelo quantum_dashboard.py.

Estratégia (caso clássico):
  Tc_add depende APENAS de Tc  →  N chamadas de fsolve
  Th_add depende APENAS de Th  →  N chamadas de fsolve
  Grid (N×N) montado por broadcasting numpy  →  sem fsolve adicional

  Para N=3000: 6.000 fsolve em vez de 9.000.000  (~1500× mais rápido)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from thermodynamics_functions import (
    _adiab_classica,
    ciclo_quantico,
    energia_media, magnetizacao, entropia,
)

# ── Configurações ─────────────────────────────────────────────────────────────
RES    = 15000         # pontos por eixo
PASSOS = 60           # iterações internas do fsolve (suficiente para classificação)
HI     = 1.0
HF     = 2.0
J_VALS = [0.00, 0.24, 0.51, 0.70, 1.00]
TIPOS  = ["classic", "quantum"]
TOL_S  = 0.01

MODO_NOMES = [
    "Motor", "Refrig (Tc<Th)", "Refrig (Tc>Th)",
    "Acel (Tc<Th)", "Acel (Tc>Th)", "Aquecedor",
]
MODO_IDX = {m: i for i, m in enumerate(MODO_NOMES)}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers: uma adiabática por vez ───────────────────────────────────────────

def _T_at_h(J, h_start, h_end, T_start, P):
    """Temperatura ao final de uma adiabática de h_start até h_end."""
    try:
        _, T = _adiab_classica(J, h_start, h_end, T_start,
                               entropia(J, h_start, T_start), P)
        return float(T[-1])
    except Exception:
        return np.nan

# Mantidos como wrappers para compatibilidade com código paralelo existente
def _Tc_add(J, hi, hf, Tc, P):
    return _T_at_h(J, hi, hf, Tc, P)

def _Th_add(J, hf, hi, Th, P):
    return _T_at_h(J, hf, hi, Th, P)

def _Tc_mid(J, hi, h_mid, Tc, P):
    return _T_at_h(J, hi, h_mid, Tc, P)

def _Th_mid(J, hf, h_mid, Th, P):
    return _T_at_h(J, hf, h_mid, Th, P)


# ── Grid clássico (vetorizado) ────────────────────────────────────────────────

def _classic_grid(J, hi, hf, Tc_a, Th_a, P):
    """
    Monta os mapas de regiões para o caso clássico usando broadcasting.

    Passo 1: pré-computa Tc_add[i] e Th_add[j] em paralelo (2×N fsolve)
    Passo 2: todas as quantidades do grid são combinações 1D → broadcasting (sem fsolve)
    """
    N = len(Tc_a)

    print(f"    Pré-computando {N} Tc_add (paralelo)...")
    t0 = time.time()
    Tc_add = np.array(
        Parallel(n_jobs=-1)(
            delayed(_Tc_add)(J, hi, hf, float(tc), P)
            for tc in tqdm(Tc_a, desc="    Tc_add", leave=False)
        ), dtype=np.float64
    )

    print(f"    Pré-computando {N} Th_add (paralelo)...")
    Th_add = np.array(
        Parallel(n_jobs=-1)(
            delayed(_Th_add)(J, hf, hi, float(th), P)
            for th in tqdm(Th_a, desc="    Th_add", leave=False)
        ), dtype=np.float64
    )
    print(f"    fsolve: {time.time()-t0:.1f}s  →  montando grid por broadcasting...")

    # Máscaras NaN (pontos onde fsolve não convergiu)
    valid_c = ~np.isnan(Tc_add)
    valid_h = ~np.isnan(Th_add)

    # ── Energias (1D) ─────────────────────────────────────────────────────────
    # Inicializa com NaN; preenche apenas onde fsolve convergiu
    E1 = np.full(N, np.nan);  E1[valid_c] = energia_media(J, hi,  Tc_a[valid_c])
    E2 = np.full(N, np.nan);  E2[valid_c] = energia_media(J, hf,  Tc_add[valid_c])
    E3 = np.full(N, np.nan);  E3[valid_h] = energia_media(J, hf,  Th_a[valid_h])
    E4 = np.full(N, np.nan);  E4[valid_h] = energia_media(J, hi,  Th_add[valid_h])

    # ── Broadcasting para o grid (N×N) ────────────────────────────────────────
    # Eixo 0 = Tc (linhas), eixo 1 = Th (colunas)
    Wi = (E2 - E1)[:, np.newaxis]   # (N, 1)
    Wo = (E4 - E3)[np.newaxis, :]   # (1, N)
    Qi = (E3 - E2)[np.newaxis, :]   # (1, N)  — E3-E2 = E3(Th)-E2(Tc)... wait
    # Correção: Qin = E3 - E2 onde E2 depende de Tc e E3 depende de Th
    # Wi = E2(Tc) - E1(Tc)  → varia com Tc
    # Wo = E4(Th) - E3(Th)  → varia com Th
    # Qin = E3(Th) - E2(Tc) → depende de ambos
    # Qout = E1(Tc) - E4(Th) → depende de ambos
    W   = Wi + Wo                                       # (N, N)
    Qin = E3[np.newaxis, :] - E2[:, np.newaxis]        # (N, N)
    Qout= E1[:, np.newaxis] - E4[np.newaxis, :]        # (N, N)

    Tc_g = Tc_a[:, np.newaxis]   # (N, 1)
    Th_g = Th_a[np.newaxis, :]   # (1, N)

    # ── Classificação dos modos ────────────────────────────────────────────────
    modo_m = np.full((N, N), np.nan, dtype=np.float32)
    m = MODO_IDX
    modo_m[W < 0]                                              = m["Motor"]
    modo_m[(W > 0) & (Qout > 0) & (Tc_g <  Th_g)]            = m["Refrig (Tc<Th)"]
    modo_m[(W > 0) & (Qin  > 0) & (Tc_g >  Th_g)]            = m["Refrig (Tc>Th)"]
    modo_m[(W > 0) & (Qout < 0) & (Qin  > 0) & (Tc_g < Th_g)]= m["Acel (Tc<Th)"]
    modo_m[(W > 0) & (Qout > 0) & (Qin  < 0) & (Tc_g > Th_g)]= m["Acel (Tc>Th)"]
    modo_m[(W > 0) & (Qin  < 0) & (Qout < 0)]                 = m["Aquecedor"]
    # Invalida onde fsolve falhou
    nan_mask = np.isnan(W)
    modo_m[nan_mask] = np.nan

    # ── Magnetizações e entropias (1D) ────────────────────────────────────────
    M1 = np.full(N, np.nan);  M1[valid_c] = magnetizacao(J, hi,  Tc_a[valid_c])
    M2 = np.full(N, np.nan);  M2[valid_c] = magnetizacao(J, hf,  Tc_add[valid_c])
    M3 = np.full(N, np.nan);  M3[valid_h] = magnetizacao(J, hf,  Th_a[valid_h])
    M4 = np.full(N, np.nan);  M4[valid_h] = magnetizacao(J, hi,  Th_add[valid_h])
    S1 = np.full(N, np.nan);  S1[valid_c] = entropia(J, hi, Tc_a[valid_c])
    S3 = np.full(N, np.nan);  S3[valid_h] = entropia(J, hf, Th_a[valid_h])

    # ── Condições Mag e Ent (broadcasting) ────────────────────────────────────
    mc  = (M1[:, np.newaxis] > M3[np.newaxis, :]).astype(np.float32)   # M1 > M3

    dS  = S1[:, np.newaxis] - S3[np.newaxis, :]                        # S1 - S3
    eps = TOL_S * (np.abs(S1[:, np.newaxis]) + np.abs(S3[np.newaxis, :])) / 2
    sc  = np.where(dS < -eps, 0, np.where(dS > eps, 2, 1)).astype(np.float32)

    m14 = (M1[:, np.newaxis] > M4[np.newaxis, :]).astype(np.float32)
    m23 = (M2[:, np.newaxis] > M3[np.newaxis, :]).astype(np.float32)
    m13 = mc  # mesmo que mc

    mi  = modo_m  # alias

    # mag: Motor (M1>M3=1, M1<M3=2) ou Não Motor=3
    _valid  = ~np.isnan(modo_m)
    _motor  = _valid & (modo_m == MODO_IDX["Motor"])
    mag_m   = np.full((N, N), np.nan, dtype=np.float32)
    mag_m[_motor & (mc == 1)] = 1.0
    mag_m[_motor & (mc == 0)] = 2.0
    mag_m[_valid & ~_motor]   = 3.0
    # ent: Motor (S1<S3=1, S1≈S3=2, S1>S3=3) ou Não Motor=4
    ent_m   = np.full((N, N), np.nan, dtype=np.float32)
    ent_m[_motor & (sc == 0)] = 1.0
    ent_m[_motor & (sc == 1)] = 2.0
    ent_m[_motor & (sc == 2)] = 3.0
    ent_m[_valid & ~_motor]   = 4.0
    # forte: apenas Motor, subdividido por S1 vs S3 e comparações de M
    forte_m = np.full((N, N), np.nan, dtype=np.float32)
    # S1 > S3  (sc == 2)
    _ms = _motor & (sc == 2)
    forte_m[_ms & (m14==1) & (m23==1) & (m13==1)] = 3.5
    forte_m[_ms & (m14==1) & (m23==1) & (m13==0)] = 4.5
    forte_m[_ms & (m14==1) & (m23==0) & (m13==1)] = 5.5
    forte_m[_ms & (m14==1) & (m23==0) & (m13==0)] = 6.5
    forte_m[_ms & (m14==0) & (m23==1) & (m13==1)] = 7.5
    forte_m[_ms & (m14==0) & (m23==1) & (m13==0)] = 8.5
    forte_m[_ms & (m14==0) & (m23==0) & (m13==1)] = 9.5
    forte_m[_ms & (m14==0) & (m23==0) & (m13==0)] = 10.5
    # S1 < S3  (sc == 0)
    _ms = _motor & (sc == 0)
    forte_m[_ms & (m14==1) & (m23==1) & (m13==1)] = 11.5
    forte_m[_ms & (m14==1) & (m23==0) & (m13==1)] = 12.5
    # 13.5 / 14.5: condições idênticas a 11.5 / 12.5 no código original → nunca atingidas
    forte_m[_ms & (m14==0) & (m23==1) & (m13==1)] = 15.5
    forte_m[_ms & (m14==0) & (m23==1) & (m13==0)] = 16.5
    forte_m[_ms & (m14==0) & (m23==0) & (m13==1)] = 17.5
    forte_m[_ms & (m14==0) & (m23==0) & (m13==0)] = 18.5
    # Motor com S1≈S3 ou igualdade numérica → 0.5
    forte_m[_motor & np.isnan(forte_m)] = 0.5
    fmag_m  = np.where(~np.isnan(mi), mi * 8  + m14*4 + m23*2 + m13 + 1,        np.nan).astype(np.float32)

    # ── Mag Crossing: adiabáticas cruzadas no diagrama M×h (Motor apenas) ────
    h_mid = (hi + hf) / 2
    print(f"    Pré-computando {N} T_mid (adiabática 1→2, paralelo)...")
    Tc_mid_arr = np.array(
        Parallel(n_jobs=-1)(
            delayed(_Tc_mid)(J, hi, h_mid, float(tc), P)
            for tc in tqdm(Tc_a, desc="    Tc_mid", leave=False)
        ), dtype=np.float64
    )
    print(f"    Pré-computando {N} T_mid (adiabática 3→4, paralelo)...")
    Th_mid_arr = np.array(
        Parallel(n_jobs=-1)(
            delayed(_Th_mid)(J, hf, h_mid, float(th), P)
            for th in tqdm(Th_a, desc="    Th_mid", leave=False)
        ), dtype=np.float64
    )
    valid_c_mid = ~np.isnan(Tc_mid_arr)
    valid_h_mid = ~np.isnan(Th_mid_arr)
    M12_mid = np.full(N, np.nan)
    M34_mid = np.full(N, np.nan)
    M12_mid[valid_c_mid] = magnetizacao(J, h_mid, Tc_mid_arr[valid_c_mid])
    M34_mid[valid_h_mid] = magnetizacao(J, h_mid, Th_mid_arr[valid_h_mid])

    # m_mid[i,j] = M12_mid[i] > M34_mid[j]  (broadcasting, NaN onde fsolve falhou)
    with np.errstate(invalid="ignore"):
        m_mid_raw = (M12_mid[:, np.newaxis] > M34_mid[np.newaxis, :]).astype(np.float32)
    nan_mid = np.isnan(M12_mid[:, np.newaxis]) | np.isnan(M34_mid[np.newaxis, :])
    m_mid = np.where(nan_mid, np.nan, m_mid_raw).astype(np.float32)

    # Classificação:  (m14, m_mid, m23) → código
    #  (1,1,1)=1  (0,0,0)=2  (1,0,0)=3  (1,1,0)=4
    #  (0,1,1)=5  (0,0,1)=6  (1,0,1)=7  (0,1,0)=8
    mag_cross_m = np.full((N, N), np.nan, dtype=np.float32)
    _vc = _motor & ~nan_mid            # Motor com midpoint válido
    mag_cross_m[_vc & (m14==1) & (m_mid==1) & (m23==1)] = 1.0
    mag_cross_m[_vc & (m14==0) & (m_mid==0) & (m23==0)] = 2.0
    mag_cross_m[_vc & (m14==1) & (m_mid==0) & (m23==0)] = 3.0
    mag_cross_m[_vc & (m14==1) & (m_mid==1) & (m23==0)] = 4.0
    mag_cross_m[_vc & (m14==0) & (m_mid==1) & (m23==1)] = 5.0
    mag_cross_m[_vc & (m14==0) & (m_mid==0) & (m23==1)] = 6.0
    mag_cross_m[_vc & (m14==1) & (m_mid==0) & (m23==1)] = 7.0
    mag_cross_m[_vc & (m14==0) & (m_mid==1) & (m23==0)] = 8.0
    # Fallback: Motor sem midpoint → classificação apenas pelos extremos
    _vf = _motor & nan_mid
    mag_cross_m[_vf & (m14==1) & (m23==1)] = 1.0
    mag_cross_m[_vf & (m14==0) & (m23==0)] = 2.0
    mag_cross_m[_vf & (m14==1) & (m23==0)] = 3.0
    mag_cross_m[_vf & (m14==0) & (m23==1)] = 5.0

    return dict(Tc=Tc_a.astype(np.float32), Th=Th_a.astype(np.float32),
                modo=modo_m, mag=mag_m, ent=ent_m, forte=forte_m, fmag=fmag_m,
                mag_cross=mag_cross_m)


# ── Grid quântico (vetorizado) ────────────────────────────────────────────────

def _quantum_grid(J, hi, hf, Tc_a, Th_a, P):
    """
    Caso quântico: populações congeladas → sem fsolve.
    Tudo vetorizado com numpy.
    """
    from thermodynamics_functions import _pops_termicas, _obs_de_pops

    N = len(Tc_a)

    # Energias nos estados de equilíbrio
    E1 = energia_media(J, hi, Tc_a)   # (N,)
    E3 = energia_media(J, hf, Th_a)   # (N,)

    # Populações congeladas → U nos estados 2 e 4
    # E2 = U das pops de #1 avaliadas em hf
    # E4 = U das pops de #3 avaliadas em hi
    def _U_frozen(J, h_start, T_arr, h_end):
        """U com populações congeladas de (J, h_start, T) avaliadas em h_end."""
        out = np.empty(len(T_arr))
        for k, T in enumerate(T_arr):
            pops = _pops_termicas(J, h_start, T)
            obs  = _obs_de_pops(pops, np.array([h_end]), J)
            out[k] = float(obs["U"][0])
        return out

    E2 = _U_frozen(J, hi, Tc_a, hf)   # (N,)
    E4 = _U_frozen(J, hf, Th_a, hi)   # (N,)

    # Broadcasting para o grid
    Wi  = (E2 - E1)[:, np.newaxis]                          # (N, 1)
    Wo  = (E4 - E3)[np.newaxis, :]                          # (1, N)
    W   = Wi + Wo                                            # (N, N)
    Qin = E3[np.newaxis, :] - E2[:, np.newaxis]             # (N, N)
    Qout= E1[:, np.newaxis] - E4[np.newaxis, :]             # (N, N)

    Tc_g = Tc_a[:, np.newaxis]
    Th_g = Th_a[np.newaxis, :]

    modo_m = np.full((N, N), np.nan, dtype=np.float32)
    md = MODO_IDX
    modo_m[W < 0]                                               = md["Motor"]
    modo_m[(W > 0) & (Qout > 0) & (Tc_g <  Th_g)]             = md["Refrig (Tc<Th)"]
    modo_m[(W > 0) & (Qin  > 0) & (Tc_g >  Th_g)]             = md["Refrig (Tc>Th)"]
    modo_m[(W > 0) & (Qout < 0) & (Qin  > 0) & (Tc_g < Th_g)] = md["Acel (Tc<Th)"]
    modo_m[(W > 0) & (Qout > 0) & (Qin  < 0) & (Tc_g > Th_g)] = md["Acel (Tc>Th)"]
    modo_m[(W > 0) & (Qin  < 0) & (Qout < 0)]                  = md["Aquecedor"]

    M1 = magnetizacao(J, hi, Tc_a)[:, np.newaxis]
    M3 = magnetizacao(J, hf, Th_a)[np.newaxis, :]
    S1 = entropia(J, hi, Tc_a)[:, np.newaxis]
    S3 = entropia(J, hf, Th_a)[np.newaxis, :]

    mc  = (M1 > M3).astype(np.float32)
    dS  = S1 - S3
    eps = TOL_S * (np.abs(S1) + np.abs(S3)) / 2
    sc  = np.where(dS < -eps, 0, np.where(dS > eps, 2, 1)).astype(np.float32)

    mi = modo_m
    _valid  = ~np.isnan(mi)
    _motor  = _valid & (mi == MODO_IDX["Motor"])
    mag_m   = np.full((N, N), np.nan, dtype=np.float32)
    mag_m[_motor & (mc == 1)] = 1.0
    mag_m[_motor & (mc == 0)] = 2.0
    mag_m[_valid & ~_motor]   = 3.0
    ent_m   = np.full((N, N), np.nan, dtype=np.float32)
    ent_m[_motor & (sc == 0)] = 1.0
    ent_m[_motor & (sc == 1)] = 2.0
    ent_m[_motor & (sc == 2)] = 3.0
    ent_m[_valid & ~_motor]   = 4.0

    # ── Mag Crossing quântico: midpoint com populações congeladas ─────────────
    h_mid = (hi + hf) / 2

    def _M_frozen_arr(h_start, T_arr, h_eval):
        """M com populações congeladas de (h_start, T) avaliadas em h_eval."""
        out = np.empty(len(T_arr))
        for k, T in enumerate(T_arr):
            pops = _pops_termicas(J, h_start, T)
            obs  = _obs_de_pops(pops, np.array([h_eval]), J)
            out[k] = float(obs["M"][0])
        return out

    # M ao longo das adiabáticas em h_mid (1D, varia só com Tc ou Th)
    M12_mid_1d = _M_frozen_arr(hi, Tc_a, h_mid)   # (N,)
    M34_mid_1d = _M_frozen_arr(hf, Th_a, h_mid)   # (N,)

    # Magnetizações nos extremos (M1, M2, M3, M4) para m14 e m23
    M1_1d = magnetizacao(J, hi, Tc_a)   # (N,)
    M3_1d = magnetizacao(J, hf, Th_a)   # (N,)
    M2_1d = _M_frozen_arr(hi, Tc_a, hf) # M2 = M frozen de 1 em hf
    M4_1d = _M_frozen_arr(hf, Th_a, hi) # M4 = M frozen de 3 em hi

    m14_q = (M1_1d[:, np.newaxis] > M4_1d[np.newaxis, :]).astype(np.float32)
    m23_q = (M2_1d[:, np.newaxis] > M3_1d[np.newaxis, :]).astype(np.float32)
    m_mid_q = (M12_mid_1d[:, np.newaxis] > M34_mid_1d[np.newaxis, :]).astype(np.float32)

    mag_cross_m = np.full((N, N), np.nan, dtype=np.float32)
    mag_cross_m[_motor & (m14_q==1) & (m_mid_q==1) & (m23_q==1)] = 1.0
    mag_cross_m[_motor & (m14_q==0) & (m_mid_q==0) & (m23_q==0)] = 2.0
    mag_cross_m[_motor & (m14_q==1) & (m_mid_q==0) & (m23_q==0)] = 3.0
    mag_cross_m[_motor & (m14_q==1) & (m_mid_q==1) & (m23_q==0)] = 4.0
    mag_cross_m[_motor & (m14_q==0) & (m_mid_q==1) & (m23_q==1)] = 5.0
    mag_cross_m[_motor & (m14_q==0) & (m_mid_q==0) & (m23_q==1)] = 6.0
    mag_cross_m[_motor & (m14_q==1) & (m_mid_q==0) & (m23_q==1)] = 7.0
    mag_cross_m[_motor & (m14_q==0) & (m_mid_q==1) & (m23_q==0)] = 8.0

    return dict(Tc=Tc_a.astype(np.float32), Th=Th_a.astype(np.float32),
                modo=modo_m, mag=mag_m, ent=ent_m,
                forte=np.full((N, N), np.nan, dtype=np.float32),
                fmag=np.full((N, N), np.nan, dtype=np.float32),
                mag_cross=mag_cross_m)


# ── Salvar / carregar ─────────────────────────────────────────────────────────

def cache_path(J, hi, hf, tipo):
    return os.path.join(CACHE_DIR, f"{tipo}_J{J:.2f}_hi{hi:.1f}_hf{hf:.1f}.npz")


def compute_and_save(J, hi, hf, tipo):
    path = cache_path(J, hi, hf, tipo)
    if os.path.exists(path):
        print(f"  [cache] {os.path.basename(path)} já existe — pulando.")
        return

    print(f"\n  ── {tipo.upper()}  J={J:.2f}  ({RES}×{RES}, PASSOS={PASSOS}) ──")
    t0 = time.time()

    Tc_a = np.linspace(0.1, 20.0, RES)
    Th_a = np.linspace(0.1, 20.0, RES)

    if tipo == "classic":
        data = _classic_grid(J, hi, hf, Tc_a, Th_a, PASSOS)
    else:
        data = _quantum_grid(J, hi, hf, Tc_a, Th_a, PASSOS)

    np.savez_compressed(path, **data)

    dt = time.time() - t0
    n_ok = int(np.sum(~np.isnan(data["modo"])))
    total = RES * RES
    print(f"  ✓ Salvo: {os.path.basename(path)}  "
          f"({n_ok/total*100:.1f}% válidos, {dt:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print(f"  Pré-computando mapas de regiões")
    print(f"  Resolução : {RES}×{RES}  ({RES**2:,} pontos)")
    print(f"  PASSOS    : {PASSOS}")
    print(f"  J values  : {J_VALS}")
    print(f"  Saída     : {CACHE_DIR}/")
    print("=" * 60)

    t_total = time.time()
    for tipo in TIPOS:
        for J in J_VALS:
            compute_and_save(J, HI, HF, tipo)

    print()
    print(f"✓ Concluído em {(time.time()-t_total)/60:.1f} min")
    print(f"  Arquivos em: {CACHE_DIR}/")
    print()
