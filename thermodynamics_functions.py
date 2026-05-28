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
      Qin  = E3 − E2  (calor na isocórica em hf — banho em Th)
      Qout = E1 − E4  (calor na isocórica em hi — banho em Tc)
      W    = Win + Wout  (W < 0 → sistema produz trabalho)

    A conservação de energia exige W + Qin + Qout = 0 (ciclo fechado).
    Das 8 combinações de sinais possíveis, apenas 6 são fisicamente realizáveis.

    Modos com Th > Tc (regime padrão):
      Motor       : W < 0, Qin > 0, Qout < 0  (absorve em Th, libera em Tc)
      Refrigerador: W > 0, Qin < 0, Qout > 0  (bombeia calor Tc → Th)
      Acelerador  : W < 0, Qin > 0, Qout > 0  (absorve calor dos 2 banhos)
      Aquecedor   : W > 0, Qin < 0, Qout < 0  (deposita calor nos 2 banhos)

    Modos com Tc > Th (regime invertido — papéis dos banhos trocados):
      Motor       : W < 0, Qin < 0, Qout > 0  (absorve em Tc, libera em Th)
      Refrigerador: W > 0, Qin > 0, Qout < 0  (bombeia calor Th → Tc)

    Nota: as 6 combinações acima cobrem todos os casos realizáveis.
    As 2 combinações restantes (W<0,Qin<0,Qout<0) e (W>0,Qin>0,Qout>0)
    violam a conservação de energia e não ocorrem numericamente.
    """
    W = Win + Wout
    if   W < 0 and Qin > 0 and Qout < 0:
        return 'Motor'
    elif W < 0 and Qin < 0 and Qout > 0:
        return 'Motor'          # regime Tc > Th (invertido)
    elif W > 0 and Qin < 0 and Qout > 0:
        return 'Refrigerador'
    elif W > 0 and Qin > 0 and Qout < 0:
        return 'Refrigerador'   # regime Tc > Th (invertido)
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


# ══════════════════════════════════════════════════════════════════════════════
#  ENERGIA POR NÍVEL  (contribuição de cada autovalor em cada processo)
# ══════════════════════════════════════════════════════════════════════════════

def _energia_nivel(pops, h, J):
    """
    Energia individual de cada nível: U_n = E_n · p_n

    Níveis (ordem de _pops_termicas):
        pops[0] → E =  0      (estado singleto)
        pops[1] → E = −2h
        pops[2] → E = +2h
        pops[3] → E = −8J
    """
    return np.array([
        0.0         * pops[0],
        (-2*h)      * pops[1],
        (+2*h)      * pops[2],
        (-8*J)      * pops[3],
    ])


def _empacota_niveis(dU):
    """Empacota array de 4 contribuições num dict com chaves descritivas."""
    return {
        '+2h': float(dU[2]),
        '0'  : float(dU[0]),
        '-2h': float(dU[1]),
        '-8J': float(dU[3]),
    }


def energias_por_nivel_classico(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Energia trocada por cada nível de autovalor no ciclo de Otto CLÁSSICO.

    Decomposição:
      • Adiabáticas (1→2, 3→4) — trabalho por nível:
          ΔW_n = E_n_final · p_n_final − E_n_inicial · p_n_inicial
      • Isocóricas (2→3, 4→1) — calor por nível:
          ΔQ_n = E_n · (p_n_final − p_n_inicial)

    Retorna
    -------
    dict com chaves 'W12', 'Q23', 'W34', 'Q41' e 'totais'.
    Cada entrada é um dict com chaves '+2h', '0', '-2h', '-8J' e 'total'.
    """
    _, _, Tc_add, Th_add = ciclo_classico(J, hi, hf, Tc, Th, PASSOS)

    p1  = _pops_termicas(J, hi,  Tc)
    p2C = _pops_termicas(J, hf,  Tc_add)
    p3  = _pops_termicas(J, hf,  Th)
    p4C = _pops_termicas(J, hi,  Th_add)

    # ── Trabalho nas adiabáticas: ΔW_n = E_n_f·p_n_f − E_n_i·p_n_i ─────────
    W12 = _energia_nivel(p2C, hf, J) - _energia_nivel(p1,  hi, J)
    W34 = _energia_nivel(p4C, hi, J) - _energia_nivel(p3,  hf, J)

    # ── Calor nas isocóricas: ΔQ_n = E_n · Δp_n ──────────────────────────────
    # isocórica 2→3: h = hf fixo
    E_at_hf = np.array([0.0, -2*hf, +2*hf, -8*J])
    Q23 = E_at_hf * (p3 - p2C)

    # isocórica 4→1: h = hi fixo
    E_at_hi = np.array([0.0, -2*hi, +2*hi, -8*J])
    Q41 = E_at_hi * (p1 - p4C)

    def _dict(dU):
        d = _empacota_niveis(dU)
        d['total'] = sum(d.values())
        return d

    return dict(
        W12=_dict(W12), Q23=_dict(Q23),
        W34=_dict(W34), Q41=_dict(Q41),
        Tc_add=Tc_add,  Th_add=Th_add,
    )


def energias_por_nivel_quantico(J, hi, hf, Tc, Th):
    """
    Energia trocada por cada nível de autovalor no ciclo de Otto QUÂNTICO.

    Nas adiabáticas quânticas as populações ficam CONGELADAS (p = p_i para todo h),
    por isso o trabalho por nível se reduz a:
        ΔW_n = p_n_frozen · (E_n_final − E_n_inicial)

    Níveis com E independente de h (E=0 e E=−8J) não fazem trabalho adiabático.

    Retorna
    -------
    dict com chaves 'W12', 'Q23', 'W34', 'Q41'.
    Cada entrada é um dict com chaves '+2h', '0', '-2h', '-8J' e 'total'.
    """
    p1 = _pops_termicas(J, hi, Tc)   # estado #1 — equilíbrio real
    p3 = _pops_termicas(J, hf, Th)   # estado #3 — equilíbrio real
    # estado #2: pops congeladas de #1 em h=hf
    # estado #4: pops congeladas de #3 em h=hi

    # ── Trabalho adiabático 1→2: p = p1 (frozen), E_n muda de hi→hf ─────────
    # ΔW_n = p1_n · (E_n(hf) − E_n(hi))
    # E(+2h): Δ = +2(hf−hi);  E(-2h): Δ = -2(hf−hi);  E=0,E=-8J: Δ = 0
    delta_E_12 = np.array([0.0, -2*(hf-hi), +2*(hf-hi), 0.0])
    W12 = p1 * delta_E_12

    # ── Calor isocórico 2→3: h = hf fixo, pops vão de p1 (frozen) → p3 ─────
    E_at_hf = np.array([0.0, -2*hf, +2*hf, -8*J])
    Q23 = E_at_hf * (p3 - p1)

    # ── Trabalho adiabático 3→4: p = p3 (frozen), E_n muda de hf→hi ─────────
    delta_E_34 = np.array([0.0, -2*(hi-hf), +2*(hi-hf), 0.0])
    W34 = p3 * delta_E_34

    # ── Calor isocórico 4→1: h = hi fixo, pops vão de p3 (frozen) → p1 ─────
    E_at_hi = np.array([0.0, -2*hi, +2*hi, -8*J])
    Q41 = E_at_hi * (p1 - p3)

    def _dict(dU):
        d = _empacota_niveis(dU)
        d['total'] = sum(d.values())
        return d

    return dict(W12=_dict(W12), Q23=_dict(Q23), W34=_dict(W34), Q41=_dict(Q41))


# ══════════════════════════════════════════════════════════════════════════════
#  EFICIÊNCIA POR NÍVEL  (framework do artigo — eq. 4)
# ══════════════════════════════════════════════════════════════════════════════

def eficiencia_por_nivel(J, hi, hf, Tc, Th, PASSOS=100):
    """
    Decompõe a eficiência do ciclo de Otto segundo o framework de
    níveis 'working' vs 'idle' (ver eq. 4 do artigo de referência).

    Classificação dos níveis
    ────────────────────────
    • Working  (n ∈ W)  :  E_n varia com h  →  E = ±2h
    • Idle     (n ∉ W)  :  E_n independe de h  →  E = −8J  e  E = 0

    Ciclo QUÂNTICO — resultado exato (eq. 4)
    ──────────────────────────────────────────
    Para o ciclo quântico, os níveis idle NÃO fazem trabalho (W_{idle}^Q = 0),
    portanto toda a sua contribuição se manifesta apenas como calor trocado
    nos processos isocóricos. Definindo:

        η₀ = 1 − hᵢ/h_f      ← eficiência dos níveis working sozinhos
                                  (análogo à eficiência de Carnot para o Otto quântico)

    A eq. (4) do artigo afirma que a eficiência total é:

        η^Q / η₀  =  1 − q^{hot}_{idle} / Q^{hot}_{total}        (eq. 4)

    onde  q^{hot}_{idle} = Q_in_{−8J}  é o calor que o nível idle troca
    com o banho quente. O resultado tem uma interpretação física direta:

        • q^{hot}_{idle} > 0  →  idle absorve calor do banho quente
                                   →  menos calor chega aos níveis working
                                   →  η^Q < η₀  (degradação)

        • q^{hot}_{idle} < 0  →  idle despeja calor no banho quente
                                   →  banho quente "recebe de volta" calor
                                   →  η^Q > η₀  (enhancement)

    Ciclo CLÁSSICO — quebra da eq. (4)
    ────────────────────────────────────
    No ciclo clássico, as adiabáticas re-equilibram as populações, de modo
    que o nível −8J TAMBÉM faz trabalho (W_{−8J}^C ≠ 0). A eq. (4) deixa
    de ser exata. Definimos o desvio:

        δ_C  =  η^C/η₀  −  (1 − Q_in_{−8J}^C / Q_in^C)

    Este desvio quantifica o quanto o ciclo clássico quebra a relação do
    artigo. Fisicamente, δ_C é não-nulo porque no clássico o nível −8J
    contribui tanto via calor (como no quântico) quanto via trabalho
    (exclusivo do ciclo clássico).

    Convenção de Q_in / Q_out
    ──────────────────────────
        Th ≥ Tc  →  Q_in = Q23  (banho em Th é o quente)
        Tc  > Th →  Q_in = Q41  (banho em Tc é o quente — motor invertido)

    Parâmetros
    ──────────
    J, hi, hf, Tc, Th : parâmetros do ciclo (ver ciclo_classico / ciclo_quantico)
    PASSOS            : pontos nas adiabáticas clássicas (padrão 100)

    Retorna
    ───────
    dict com:
        eta0        : eficiência de referência  η₀ = 1 − hi/hf
        eta_Carnot  : η_Carnot = 1 − min(Tc,Th)/max(Tc,Th)

        # Ciclo quântico
        eta_Q       : eficiência total quântica
        ratio_Q     : η^Q / η₀
        x_Q         : 1 − Q_in_{−8J}^Q / Q_in^Q  (deve ser = ratio_Q, verifica eq. 4)
        desvio_Q    : ratio_Q − x_Q               (deve ser ≈ 0 numericamente)
        Q_idle_Q    : Q_in_{−8J}^Q  (calor do nível idle no banho quente)
        Q_in_Q      : Q_in total quântico
        modo_Q      : modo de operação (Motor, Refrigerador, ...)

        # Ciclo clássico
        eta_C       : eficiência total clássica
        ratio_C     : η^C / η₀
        x_C         : 1 − Q_in_{−8J}^C / Q_in^C  (análogo à eq. 4, mas ≠ ratio_C)
        desvio_C    : ratio_C − x_C               (quebra da eq. 4 no clássico)
        Q_idle_C    : Q_in_{−8J}^C
        Q_in_C      : Q_in total clássico
        W_idle_C    : trabalho total do nível −8J no clássico (W12_{-8J} + W34_{-8J})
        modo_C      : modo de operação
    """
    eta0       = 1.0 - hi / hf
    eta_Carnot = 1.0 - min(Tc, Th) / max(Tc, Th)

    # ── Ciclo quântico ────────────────────────────────────────────────────────
    # A eq. (4) do artigo é derivada com Q_in = Q23 e η₀ = 1 − hi/hf.
    # Isso é válido SEMPRE (não apenas para Th ≥ Tc), pois a identidade é puramente
    # algébrica: −W_{±2h} = η₀ · Q23_{±2h}  (pois E_{±2h} ∝ h, populações congeladas).
    #
    # Portanto usamos SEMPRE proc_in = 'Q23' para a verificação da eq. (4), mesmo
    # quando Tc > Th (onde Q23 < 0 fora do motor). O sinal de η_signed e x_Q
    # fica consistente, e desvio_Q ≈ 0 para qualquer ponto do espaço de parâmetros.
    #
    # Prova: W_{idle}^Q = 0 → W_total^Q = W_{±2h}^Q
    #        Q23_{±2h} = hf·(M1 − M3)   e   −W_{±2h} = (hf−hi)·(M1−M3)
    #        ∴ −W_{±2h} / Q23_{±2h} = (hf−hi)/hf = η₀   (independente do sinal de M1−M3)
    #        ∴ η^Q_signed = −W/Q23 = η₀·(1 − Q23_{-8J}/Q23_total) = η₀·x_Q   (exato)
    nQ = energias_por_nivel_quantico(J, hi, hf, Tc, Th)
    efQ = eficiencia_quantica(J, hi, hf, Tc, Th)

    Q_in_Q    = nQ['Q23']['total']
    Q_idle_Q  = nQ['Q23']['-8J']
    W_total_Q = nQ['W12']['total'] + nQ['W34']['total']

    # Eficiência assinada: η_signed = −W/Q23  (pode ser < 0 fora do Motor)
    eta_Q_signed = (-W_total_Q / Q_in_Q) if Q_in_Q != 0 else np.nan
    x_Q          = (1.0 - Q_idle_Q / Q_in_Q) if Q_in_Q != 0 else np.nan
    ratio_Q      = (eta_Q_signed / eta0)       if eta0   != 0 else np.nan
    desvio_Q     = ratio_Q - x_Q   # deve ser ~0 para qualquer (Tc, Th)

    # ── Ciclo clássico ────────────────────────────────────────────────────────
    # Idem: usamos sempre Q23 e η₀ = 1 − hi/hf como referência.
    # A eq. (4) NÃO é exata no clássico porque W_{-8J}^C ≠ 0.
    # O desvio δ_C = ratio_C − x_C captura essa quebra.
    nC  = energias_por_nivel_classico(J, hi, hf, Tc, Th, PASSOS)
    efC = eficiencia_classica(J, hi, hf, Tc, Th, PASSOS)

    Q_in_C    = nC['Q23']['total']
    Q_idle_C  = nC['Q23']['-8J']
    W_total_C = nC['W12']['total'] + nC['W34']['total']
    W_idle_C  = nC['W12']['-8J'] + nC['W34']['-8J']

    eta_C_signed = (-W_total_C / Q_in_C) if Q_in_C != 0 else np.nan
    x_C          = (1.0 - Q_idle_C / Q_in_C) if Q_in_C != 0 else np.nan
    ratio_C      = (eta_C_signed / eta0)       if eta0   != 0 else np.nan
    desvio_C     = ratio_C - x_C

    return dict(
        eta0=eta0, eta_Carnot=eta_Carnot,
        # Quântico
        eta_Q=efQ['eta'], eta_Q_signed=eta_Q_signed,
        ratio_Q=ratio_Q, x_Q=x_Q, desvio_Q=desvio_Q,
        Q_idle_Q=Q_idle_Q, Q_in_Q=Q_in_Q, modo_Q=efQ['modo'],
        # Clássico
        eta_C=efC['eta'], eta_C_signed=eta_C_signed,
        ratio_C=ratio_C, x_C=x_C, desvio_C=desvio_C,
        Q_idle_C=Q_idle_C, Q_in_C=Q_in_C, W_idle_C=W_idle_C, modo_C=efC['modo'],
    )


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
