"""
thermodynamics_functions.py
============================
Funções termodinâmicas do modelo de 2 spins clássico (4 estados).

Parâmetros:
    J  — acoplamento spin-spin
    h  — campo magnético externo
    T  — temperatura

Equação de estado base:
    Z(J,h,T) = 1 + 2·cosh(2h/T) + exp(8J/T)
"""

import numpy as np
from scipy.optimize import fsolve

# ══════════════════════════════════════════════════════════════════════════════
#  EQUAÇÕES DE ESTADO
# ══════════════════════════════════════════════════════════════════════════════

def Z(J, h, T):
    """Função de Partição  Z = 1 + 2·cosh(2h/T) + exp(8J/T)"""
    return 1.0 + 2.0 * np.cosh(2*h/T) + np.exp(8*J/T)

def energia_livre(J, h, T):
    """Energia Livre de Helmholtz  F = −T·ln(Z)"""
    return -T * np.log(Z(J, h, T))

def energia_media(J, h, T):
    """Energia Interna  U = −(8J·e^{8J/T} + 4h·sinh(2h/T)) / Z"""
    num = -(8*J * np.exp(8*J/T) + 4*h * np.sinh(2*h/T))
    return num / Z(J, h, T)

def entropia(J, h, T):
    """Entropia  S = ln(Z) − U/T  (equivalente a −∂F/∂T)"""
    num = (-(8*J * np.exp(8*J/T)) - (4*h * np.sinh(2*h/T))) / T
    return np.log(Z(J, h, T)) + num / Z(J, h, T)

def magnetizacao(J, h, T):
    """Magnetização  M = 4·sinh(2h/T) / Z"""
    return 4.0 * np.sinh(2*h/T) / Z(J, h, T)

def susceptibilidade(J, h, T):
    """Susceptibilidade magnética  χ = ∂M/∂h"""
    Zv = Z(J, h, T)
    t1 = 8.0 * np.cosh(2*h/T) / (T * Zv)
    t2 = 16.0 * np.sinh(2*h/T)**2 / (T * Zv**2)
    return t1 - t2

def calor_especifico(J, h, T):
    """Calor Específico  C = dU/dT  (fórmula analítica completa)"""
    e   = np.exp(8.0 * J / T)
    csh = np.cosh(2.0 * h / T)
    sh  = np.sinh(2.0 * h / T)
    Zv  = Z(J, h, T)

    t1 = T * (
        (64.0 * e * J**2) / T**4
        + (16.0 * e * J)  / T**3
        + (8.0  * h**2 * csh) / T**4
        + (8.0  * h    * sh)  / T**3
    ) / Zv

    inner = -(8.0 * e * J) / T**2 - (4.0 * h * sh) / T**2

    t2 = 2.0 * inner / Zv
    t3 = -T * inner**2 / Zv**2

    return T * (t1 + t2 + t3)


# Dicionário para iterar sobre todas as funções de forma genérica
FUNCOES = {
    'Z' : (Z,              r'$Z(J,h,T)$',   'Z'),
    'F' : (energia_livre,  r'$F(J,h,T)$',   'F'),
    'U' : (energia_media,  r'$U(J,h,T)$',   'U'),
    'S' : (entropia,       r'$S(J,h,T)$',   'S'),
    'M' : (magnetizacao,   r'$M(J,h,T)$',   'M'),
    'X' : (susceptibilidade, r'$\chi(J,h,T)$', r'$\chi$'),
    'C' : (calor_especifico, r'$C(J,h,T)$', 'C'),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PICOS
# ══════════════════════════════════════════════════════════════════════════════

def T_peak_generic(func, J, h, T_range=None):
    """Temperatura do máximo de func(J,h,T) para J e h fixos."""
    if T_range is None:
        T_range = np.linspace(0.05, 30, 5000)
    return T_range[np.argmax(func(J, h, T_range))]

def T_peak_M(J, h, T_range=None):
    """Temperatura do pico da magnetização M(J,h,T)."""
    return T_peak_generic(magnetizacao, J, h, T_range)

def T_peak_C(J, h, T_range=None):
    """Temperatura do pico de Schottky do calor específico C(J,h,T)."""
    return T_peak_generic(calor_especifico, J, h, T_range)


# ══════════════════════════════════════════════════════════════════════════════
#  CICLO CLÁSSICO  (adiabáticas com S = cte via fsolve)
# ══════════════════════════════════════════════════════════════════════════════

def _adiab_classica(J, h_start, h_end, T_start, S_fixed, PASSOS):
    """Percorre a adiabática clássica mantendo S = S_fixed."""
    h_vals = np.linspace(h_start, h_end, PASSOS)
    T_vals = np.zeros(PASSOS)
    T_curr = T_start
    S_curr = round(S_fixed, 10)
    for i, h in enumerate(h_vals):
        sol = fsolve(lambda T: S_curr - entropia(J, h, abs(T[0])), T_curr)[0]
        T_curr = abs(sol)
        S_curr = round(entropia(J, h, T_curr), 10)
        T_vals[i] = T_curr
    return h_vals, T_vals

def ciclo_classico(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Calcula o ciclo de Otto clássico.

    Retorna
    -------
    estados : dict  {1,2,3,4} → {h, T, label}
    caminhos: dict  {'12','23','34','41'} → {h, T}
    Tc_add  : temperatura em 2 (após adiabática 1→2)
    Th_add  : temperatura em 4 (após adiabática 3→4)
    """
    h_12, T_12 = _adiab_classica(J, hi, hf, Tc, entropia(J, hi, Tc), PASSOS)
    Tc_add = T_12[-1]

    h_34, T_34 = _adiab_classica(J, hf, hi, Th, entropia(J, hf, Th), PASSOS)
    Th_add = T_34[-1]

    estados = {
        1: dict(h=hi, T=Tc,     label='1C'),
        2: dict(h=hf, T=Tc_add, label='2C'),
        3: dict(h=hf, T=Th,     label='3C'),
        4: dict(h=hi, T=Th_add, label='4C'),
    }
    caminhos = {
        '12': dict(h=h_12,               T=T_12),
        '23': dict(h=np.full(PASSOS, hf), T=np.linspace(Tc_add, Th, PASSOS)),
        '34': dict(h=h_34,               T=T_34),
        '41': dict(h=np.full(PASSOS, hi), T=np.linspace(Th_add, Tc, PASSOS)),
    }
    return estados, caminhos, Tc_add, Th_add


# ══════════════════════════════════════════════════════════════════════════════
#  CICLO QUÂNTICO  (física correta: populações congeladas nas adiabáticas)
# ══════════════════════════════════════════════════════════════════════════════

def _pops_termicas(J, h, T):
    """Populações da distribuição de Boltzmann para o sistema de 4 estados.

    Níveis de energia: E = {0,  -2h,  +2h,  -8J}
    Índices:           n = { 0,    1,    2,    3 }
    """
    Zv = Z(J, h, T)
    return np.array([
        1.0 / Zv,
        np.exp( 2*h / T) / Zv,
        np.exp(-2*h / T) / Zv,
        np.exp( 8*J / T) / Zv,
    ])


def _obs_de_pops(pops, h_arr, J):
    """
    Calcula M, S e U a partir das populações CONGELADAS para um array de h.

    Níveis:  E_n(h) = {0, -2h, +2h, -8J}
    m_n      (magnetização do nível n) = {0, +2, -2, 0}  ← independem de h

    M = 2·p1 − 2·p2        → constante (não depende de h)
    S = −Σ p_n ln(p_n)     → constante
    U = −2h·(p1−p2) − 8J·p3 → linear em h
    """
    p0, p1, p2, p3 = pops
    M_val = float(2*p1 - 2*p2)
    S_val = float(-np.sum(pops * np.log(pops)))
    h_arr = np.asarray(h_arr)
    U_arr = -2 * h_arr * (p1 - p2) - 8*J * p3
    return dict(
        M=np.full_like(h_arr, M_val, dtype=float),
        S=np.full_like(h_arr, S_val, dtype=float),
        U=U_arr,
    )


def ciclo_quantico(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Ciclo de Otto quântico.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADIABÁTICAS  (1→2  e  3→4)                                        │
    │  • h muda → níveis de energia mudam                                 │
    │  • populações CONGELADAS → M = cte  e  S = cte (invariantes)        │
    │  • T NÃO é definido: o sistema sai do equilíbrio                    │
    │  • Em qualquer diagrama (h, Q): Q é CONSTANTE durante a adiabática  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  ISOCÓRICAS  (2→3  e  4→1)                                          │
    │  • h fixo, sistema termaliza com o banho                            │
    │  • Conecta o estado congelado (2 ou 4) ao próximo equilíbrio        │
    │  • M, S, U interpolados linearmente entre os estados                │
    └─────────────────────────────────────────────────────────────────────┘

    Estados:
      #1 = (J, hi, Tc)  equilíbrio térmico real
      #2 = (J, hf, None) fora do equilíbrio — populações congeladas de #1
      #3 = (J, hf, Th)  equilíbrio térmico real
      #4 = (J, hi, None) fora do equilíbrio — populações congeladas de #3

    Nos diagramas (h, Q): o ciclo forma um RETÂNGULO:
      adiabáticas → linhas horizontais (Q = cte)
      isocóricas  → linhas verticais (h = cte, Q varia de Q1 a Q3)
    """
    h_12 = np.linspace(hi, hf, PASSOS)
    h_34 = np.linspace(hf, hi, PASSOS)
    t    = np.linspace(0, 1, PASSOS)

    # ── Populações congeladas dos estados de equilíbrio #1 e #3 ──────────
    p_1 = _pops_termicas(J, hi, Tc)
    p_3 = _pops_termicas(J, hf, Th)

    # ── Adiabáticas: M, S constantes; U linear em h ───────────────────────
    obs_12 = _obs_de_pops(p_1, h_12, J)   # M = M1, S = S1 para todo h ∈ [hi,hf]
    obs_34 = _obs_de_pops(p_3, h_34, J)   # M = M3, S = S3 para todo h ∈ [hf,hi]

    # ── Observáveis nos 4 estados ─────────────────────────────────────────
    obs_1 = {k: float(v[0])  for k, v in obs_12.items()}  # início de 1→2
    obs_2 = {k: float(v[-1]) for k, v in obs_12.items()}  # fim de 1→2 (congelado)
    obs_3 = {k: float(v[0])  for k, v in obs_34.items()}  # início de 3→4
    obs_4 = {k: float(v[-1]) for k, v in obs_34.items()}  # fim de 3→4 (congelado)

    estados = {
        '1': dict(h=hi, T=Tc,   label='1Q', **obs_1),  # equilíbrio
        '2': dict(h=hf, T=None, label='2Q', **obs_2),  # fora do equilíbrio
        '3': dict(h=hf, T=Th,   label='3Q', **obs_3),  # equilíbrio
        '4': dict(h=hi, T=None, label='4Q', **obs_4),  # fora do equilíbrio
    }

    # ── Isocóricas: interpolação linear entre estados congelado→equilíbrio ─
    caminhos = {
        '12': dict(h=h_12, T=None, **obs_12),
        '23': dict(
            h=np.full(PASSOS, hf),
            T=np.linspace(Tc, Th, PASSOS),
            M=obs_2['M'] + t * (obs_3['M'] - obs_2['M']),
            S=obs_2['S'] + t * (obs_3['S'] - obs_2['S']),
            U=obs_2['U'] + t * (obs_3['U'] - obs_2['U']),
        ),
        '34': dict(h=h_34, T=None, **obs_34),
        '41': dict(
            h=np.full(PASSOS, hi),
            T=np.linspace(Th, Tc, PASSOS),
            M=obs_4['M'] + t * (obs_1['M'] - obs_4['M']),
            S=obs_4['S'] + t * (obs_1['S'] - obs_4['S']),
            U=obs_4['U'] + t * (obs_1['U'] - obs_4['U']),
        ),
    }

    return estados, caminhos, Tc, Th


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS: avalia funções ao longo dos caminhos do ciclo
# ══════════════════════════════════════════════════════════════════════════════

def ciclo_Q(func, J, caminhos):
    """
    Avalia func(J, h, T) ao longo dos caminhos de um ciclo CLÁSSICO.
    (Requer T definido em todo o caminho — use apenas para ciclo_classico.)

    Parâmetros
    ----------
    func     : callable  (J, h, T) → float | array
    J        : float
    caminhos : dict retornado por ciclo_classico ou ciclo_quantico

    Retorna
    -------
    dict com as mesmas chaves, valores = array de func ao longo do caminho
    """
    return {leg: func(J, p['h'], p['T']) for leg, p in caminhos.items()}


def ciclo_Q_quantum(obs_key, cam_Q):
    """
    Retorna os valores de um observável ao longo do ciclo QUÂNTICO.

    Para adiabáticas: usa os valores pré-calculados de populações congeladas.
    Para isocóricas:  usa a interpolação linear entre estados reais.

    obs_key : 'M', 'S' ou 'U'
              (F, Z, C, chi não são definidos nas adiabáticas quânticas)
    """
    return {leg: p[obs_key] for leg, p in cam_Q.items()}


def ciclo_Q_quantum_termico(func, J, cam_Q):
    """
    Avalia func(J, h, T) ao longo das ISOCÓRICAS do ciclo quântico,
    e retorna None para as adiabáticas (T indefinido).

    Útil para F, Z, C, chi onde os valores fora-de-equilíbrio não existem.
    """
    result = {}
    for leg, p in cam_Q.items():
        if p['T'] is None:
            result[leg] = None   # adiabática: T indefinido
        else:
            result[leg] = func(J, p['h'], p['T'])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  EFICIÊNCIA DO CICLO DE OTTO
# ══════════════════════════════════════════════════════════════════════════════

def _modo_operacao(Win, Wout, Qin, Qout):
    """
    Classifica o modo de operação do ciclo com base nos sinais dos fluxos.

    Convenção (do ponto de vista do sistema):
      Win  = E2 − E1  (trabalho na adiabática 1→2)
      Wout = E4 − E3  (trabalho na adiabática 3→4)
      Qin  = E3 − E2  (calor trocado na isocórica quente)
      Qout = E1 − E4  (calor trocado na isocórica fria)
      W    = Win + Wout  (W < 0 → sistema produz trabalho)

    Modos:
      Motor       : W < 0, Qin > 0, Qout < 0  (converte calor em trabalho)
      Refrigerador: W > 0, Qin < 0, Qout > 0  (bombeia calor frio → quente)
      Acelerador  : W < 0, Qin > 0, Qout > 0  (absorve calor dos 2 banhos)
      Aquecedor   : W > 0, Qin < 0, Qout < 0  (deposita calor nos 2 banhos)
    """
    W = Win + Wout
    if   W < 0 and Qin > 0 and Qout < 0:
        return 'Motor'
    elif W > 0 and Qin < 0 and Qout > 0:
        return 'Refrigerador'
    elif W < 0 and Qin > 0 and Qout > 0:
        return 'Acelerador'
    elif W > 0 and Qin < 0 and Qout < 0:
        return 'Aquecedor'
    else:
        return 'Indefinido'


def eficiencia_classica(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Eficiência do ciclo de Otto CLÁSSICO para um ponto (Tc, Th).

    Retorna dict: Win, Wout, Qin, Qout, W, eta, modo, Tc_add, Th_add
    """
    _, _, Tc_add, Th_add = ciclo_classico(J, hi, hf, Tc, Th, PASSOS)
    E1 = float(energia_media(J, hi, Tc))
    E2 = float(energia_media(J, hf, Tc_add))
    E3 = float(energia_media(J, hf, Th))
    E4 = float(energia_media(J, hi, Th_add))
    Win  = E2 - E1
    Wout = E4 - E3
    Qin  = E3 - E2
    Qout = E1 - E4
    W    = Win + Wout
    eta  = abs(W) / abs(Qin) if Th >= Tc else abs(W) / abs(Qout)
    return dict(Win=Win, Wout=Wout, Qin=Qin, Qout=Qout, W=W, eta=eta,
                modo=_modo_operacao(Win, Wout, Qin, Qout),
                Tc_add=Tc_add, Th_add=Th_add)


def eficiencia_quantica(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Eficiência do ciclo de Otto QUÂNTICO para um ponto (Tc, Th).

    Usa populações congeladas nas adiabáticas (ver ciclo_quantico).
    Retorna dict: Win, Wout, Qin, Qout, W, eta, modo
    """
    est_Q, _, _, _ = ciclo_quantico(J, hi, hf, Tc, Th, PASSOS)
    E1 = est_Q['1']['U']
    E2 = est_Q['2']['U']
    E3 = est_Q['3']['U']
    E4 = est_Q['4']['U']
    Win  = float(E2 - E1)
    Wout = float(E4 - E3)
    Qin  = float(E3 - E2)
    Qout = float(E1 - E4)
    W    = Win + Wout
    eta  = abs(W) / abs(Qin) if Th >= Tc else abs(W) / abs(Qout)
    return dict(Win=Win, Wout=Wout, Qin=Qin, Qout=Qout, W=W, eta=eta,
                modo=_modo_operacao(Win, Wout, Qin, Qout))


def resumo_ciclo(J, hi, hf, Tc, Th, PASSOS=100):
    """Imprime resumo analítico do ciclo clássico e quântico."""
    est_C, cam_C, Tc2, Th4   = ciclo_classico(J, hi, hf, Tc, Th, PASSOS)
    est_Q, cam_Q, _, _        = ciclo_quantico(J, hi, hf, Tc, Th, PASSOS)

    print("═" * 60)
    print(f"  J={J}  hi={hi}  hf={hf}  Tc={Tc}  Th={Th}")
    print("═" * 60)
    print("\n── Clássico (S = cte nas adiabáticas) ──")
    for k, v in est_C.items():
        print(f"  Estado {v['label']}: h={v['h']:.2f}  T={v['T']:.4f}  "
              f"S={entropia(J,v['h'],v['T']):.4f}")
    print("\n── Quântico (M = cte e S = cte nas adiabáticas) ──")
    for k, v in est_Q.items():
        T_str = f"{v['T']:.4f}" if v['T'] is not None else "undef (fora-equil.)"
        print(f"  Estado {v['label']}: h={v['h']:.2f}  T={T_str}  "
              f"M={v['M']:.4f}  S={v['S']:.4f}  U={v['U']:.4f}")
    print()
