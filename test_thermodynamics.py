"""
test_thermodynamics.py
======================
Testes unitários para funções termodinâmicas e ciclos de Otto.

Execução:
    python3 -m pytest test_thermodynamics.py -v          # todos os testes
    python3 -m pytest test_thermodynamics.py -v -k ciclo # só os ciclos
    python3 test_thermodynamics.py                       # unittest direto
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '.')

from thermodynamics_functions import (
    Z, energia_livre, energia_media, entropia,
    magnetizacao, susceptibilidade, calor_especifico,
    ciclo_classico, ciclo_quantico,
    _pops_termicas, _obs_de_pops,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def deriv_num(f, x, dx=1e-6):
    """Derivada central de ordem 2."""
    return (f(x + dx) - f(x - dx)) / (2 * dx)


CASOS = [
    (0.51, 1.0,  2.0),   # caso padrão
    (0.35, 0.5,  0.8),   # J menor, T baixa
    (1.0,  2.0,  5.0),   # J grande, T alta
    (0.0,  1.0,  3.0),   # J = 0 (sem acoplamento)
    (0.51, 0.01, 1.0),   # h muito pequeno
]

CICLOS = [
    (0.51, 1.0, 2.0, 2.0, 1.0),   # caso padrão
    (0.35, 0.5, 1.5, 1.5, 0.8),   # J menor
    (1.0,  1.0, 3.0, 3.0, 0.5),   # J grande
    (0.0,  1.0, 2.0, 2.0, 1.0),   # J = 0
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIMITES ASSINTÓTICOS
# ─────────────────────────────────────────────────────────────────────────────

class TestLimites(unittest.TestCase):

    def test_Z_alta_T_converge_para_4(self):
        """Z(J, h, T→∞) → 4  (todos os 4 estados equiprováveis).

        Nota: para J grande, requer T >> 8J para convergir.
        Usamos T=1e6 para garantir convergência a places=3.
        """
        for J in [0, 0.5, 1.0]:
            for h in [0.0, 0.5, 2.0]:
                self.assertAlmostEqual(Z(J, h, 1e6), 4.0, places=3,
                    msg=f"J={J}, h={h}")

    def test_S_alta_T_converge_para_ln4(self):
        """S(J, h, T→∞) → ln(4) ≈ 1.3863  (máxima entropia = 4 estados)."""
        for J, h, _ in CASOS:
            self.assertAlmostEqual(entropia(J, h, 1e4), np.log(4), places=3,
                msg=f"J={J}, h={h}")

    def test_S_baixa_T_tende_a_zero_sem_degeneracao(self):
        """S(J, h, T→0) → 0 quando ground state é não-degenerado.

        Degenerescência ocorre quando 2h = 8J (h = 4J).
        Nesse caso S(T→0) → ln(2), não 0.
        Testamos apenas casos sem degenerescência: h ≠ 4J.
        """
        # T_safe: suficientemente baixo para S≈0, mas evitando overflow numérico
        # overflow ocorre quando exp(8J/T) > float_max, i.e., T < 8J/709
        # Para J=1.0: T_min ≈ 0.011 → usamos T=0.05
        casos_nao_degenerados = [
            (0.5, 0.5, 0.01),   # -8J=-4, -2h=-1 → gap=3 → T=0.01 ok
            (1.0, 0.5, 0.05),   # -8J=-8, -2h=-1 → gap=7 → T=0.05 (T=0.01 overflow)
            (0.5, 3.0, 0.01),   # -2h=-6, -8J=-4 → gap=2 → T=0.01 ok
        ]
        for J, h, T in casos_nao_degenerados:
            self.assertAlmostEqual(entropia(J, h, T), 0.0, places=2,
                msg=f"J={J}, h={h}, T={T}")

    def test_S_baixa_T_degeneracao_ground_state(self):
        """S(J, h, T→0) → ln(2) quando 2h = 8J (ground state duplamente degenerado).

        Para J=0.5, h=2.0: E(-2h) = E(-8J) = -4  →  dois estados de menor energia
        →  S(T→0) = ln(2) ≈ 0.6931  (degenerescência do estado fundamental).
        """
        J, h = 0.5, 2.0   # 2h = 8J = 4  ← degenerado
        self.assertAlmostEqual(entropia(J, h, 0.01), np.log(2), places=4)

    def test_M_zero_sem_campo(self):
        """M(J, h=0, T) = 0  (sem campo → sem magnetização)."""
        for J in [0, 0.5, 1.0]:
            for T in [0.1, 1.0, 10.0]:
                self.assertAlmostEqual(magnetizacao(J, 0.0, T), 0.0, places=12,
                    msg=f"J={J}, T={T}")

    def test_C_baixa_T_tende_a_zero(self):
        """C(J, h, T→0) → 0  (sistema congela abaixo do pico de Schottky).

        Atenção: para h próximo de 4J (quase-degenerescência), o pico de Schottky
        ocorre em T muito baixo. Testamos apenas casos onde o gap ΔE é grande
        o suficiente para C ≈ 0 já em T=0.05.

        Para J=0.51, h=2.0: gap = |−8J − (−2h)| = |−4.08 − (−4)| = 0.08 → pico em T≈0.04
        → T=0.05 ainda está perto do pico  (NÃO testar este caso aqui).
        """
        casos_gap_grande = [
            (0.51, 0.5),   # -8J=-4.08, -2h=-1  → gap ≈ 3 → pico em T>>0.05
            (0.51, 1.0),   # -8J=-4.08, -2h=-2  → gap ≈ 2 → pico em T~1
        ]
        for J, h in casos_gap_grande:
            self.assertAlmostEqual(calor_especifico(J, h, 0.05), 0.0, places=2,
                msg=f"J={J}, h={h}")

    def test_C_alta_T_tende_a_zero(self):
        """C(J, h, T→∞) → 0  (todos estados populados — U constante)."""
        for h in [0.5, 1.0]:
            self.assertAlmostEqual(calor_especifico(0.51, h, 1e4), 0.0, places=3,
                msg=f"h={h}")

    def test_U_baixa_T_converge_estado_fundamental(self):
        """U(J, h, T→0) → energia do estado fundamental = min(−2h, −8J).

        Os 4 níveis são: {0, −2h, +2h, −8J}.
        O estado fundamental é min(−2h, −8J), NÃO −8J − 4h.
        (Cada nível é independente; não há soma de J e h no mesmo nível.)

        Casos testados:
          J=0.5, h=1.0 → min(−2, −4) = −4 = −8J  (J domina: h < 4J)
          J=0.5, h=3.0 → min(−6, −4) = −6 = −2h  (h domina: h > 4J)
        """
        # Caso 1: J domina (h < 4J)
        J, h = 0.5, 1.0   # -8J=-4, -2h=-2 → gs = -4
        U_gs = min(-2*h, -8*J)   # = -4
        self.assertAlmostEqual(energia_media(J, h, 0.01), U_gs, places=2,
            msg=f"J={J}, h={h}: gs deveria ser -8J={-8*J}")

        # Caso 2: h domina (h > 4J)
        J, h = 0.5, 3.0   # -8J=-4, -2h=-6 → gs = -6
        U_gs = min(-2*h, -8*J)   # = -6
        self.assertAlmostEqual(energia_media(J, h, 0.01), U_gs, places=2,
            msg=f"J={J}, h={h}: gs deveria ser -2h={-2*h}")

    def test_U_alta_T_tende_a_zero(self):
        """U(J, h, T→∞) → 0  (média sobre 4 estados iguais, E médio = 0)."""
        # média de {0, -2h, +2h, -8J} com pesos iguais = -2J
        for J, h, _ in CASOS:
            U_esperado = (-8*J) / 4.0  # média dos 4 níveis com peso igual
            self.assertAlmostEqual(energia_media(J, h, 1e4), U_esperado, places=2,
                msg=f"J={J}, h={h}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RELAÇÕES TERMODINÂMICAS EXATAS
# ─────────────────────────────────────────────────────────────────────────────

class TestRelacoes(unittest.TestCase):

    def test_F_igual_U_menos_TS(self):
        """F = U − T·S  (definição de energia livre)."""
        for J, h, T in CASOS:
            F = energia_livre(J, h, T)
            U = energia_media(J, h, T)
            S = entropia(J, h, T)
            self.assertAlmostEqual(F, U - T*S, places=8,
                msg=f"J={J}, h={h}, T={T}")

    def test_S_igual_menos_dF_dT(self):
        """S = −∂F/∂T  (relação de Maxwell)."""
        for J, h, T in CASOS:
            S_analitico = entropia(J, h, T)
            S_numerico  = deriv_num(lambda t: -energia_livre(J, h, t), T, dx=1e-6)
            self.assertAlmostEqual(S_analitico, S_numerico, places=5,
                msg=f"J={J}, h={h}, T={T}")

    def test_M_igual_menos_dF_dh(self):
        """M = −∂F/∂h  (relação de Maxwell)."""
        for J, h, T in CASOS:
            if h < 0.05:
                continue   # derivada numérica instável perto de h=0
            M_analitico = magnetizacao(J, h, T)
            M_numerico  = deriv_num(lambda hv: -energia_livre(J, hv, T), h, dx=1e-6)
            self.assertAlmostEqual(M_analitico, M_numerico, places=5,
                msg=f"J={J}, h={h}, T={T}")

    def test_C_igual_dU_dT(self):
        """C = ∂U/∂T  (definição de calor específico)."""
        for J, h, T in CASOS:
            C_analitico = calor_especifico(J, h, T)
            C_numerico  = deriv_num(lambda t: energia_media(J, h, t), T, dx=1e-5)
            self.assertAlmostEqual(C_analitico, C_numerico, places=4,
                msg=f"J={J}, h={h}, T={T}")

    def test_chi_igual_dM_dh(self):
        """χ = ∂M/∂h  (definição de susceptibilidade)."""
        for J, h, T in CASOS:
            if h < 0.05:
                continue
            chi_analitico = susceptibilidade(J, h, T)
            chi_numerico  = deriv_num(lambda hv: magnetizacao(J, hv, T), h, dx=1e-6)
            self.assertAlmostEqual(chi_analitico, chi_numerico, places=5,
                msg=f"J={J}, h={h}, T={T}")

    def test_chi_igual_menos_d2F_dh2(self):
        """χ = −∂²F/∂h²."""
        for J, h, T in CASOS:
            if h < 0.1:
                continue
            chi_analitico = susceptibilidade(J, h, T)
            dh = 1e-4
            chi_numerico  = -(energia_livre(J, h+dh, T)
                               - 2*energia_livre(J, h, T)
                               + energia_livre(J, h-dh, T)) / dh**2
            self.assertAlmostEqual(chi_analitico, chi_numerico, places=4,
                msg=f"J={J}, h={h}, T={T}")

    def test_populacoes_somam_um(self):
        """Σ p_n = 1  (normalização)."""
        for J, h, T in CASOS:
            pops = _pops_termicas(J, h, T)
            self.assertAlmostEqual(pops.sum(), 1.0, places=12)

    def test_pops_reproduzem_M_e_S(self):
        """M e S calculados de populações coincidem com funções diretas."""
        for J, h, T in CASOS:
            pops = _pops_termicas(J, h, T)
            obs  = _obs_de_pops(pops, np.array([h]), J)

            self.assertAlmostEqual(float(obs['M'][0]), magnetizacao(J, h, T), places=10,
                msg=f"M de pops ≠ magnetizacao() para J={J},h={h},T={T}")
            self.assertAlmostEqual(float(obs['S'][0]), entropia(J, h, T), places=10,
                msg=f"S de pops ≠ entropia() para J={J},h={h},T={T}")
            self.assertAlmostEqual(float(obs['U'][0]), energia_media(J, h, T), places=10,
                msg=f"U de pops ≠ energia_media() para J={J},h={h},T={T}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CICLO QUÂNTICO
# ─────────────────────────────────────────────────────────────────────────────

class TestCicloQuantico(unittest.TestCase):

    def _ciclo(self, params, PASSOS=200):
        J, hi, hf, Tc, Th = params
        return ciclo_quantico(J, hi, hf, Tc, Th, PASSOS=PASSOS)

    # ── Estrutura das adiabáticas ──────────────────────────────────────────

    def test_T_none_nas_adiabaticas(self):
        """T deve ser None nas adiabáticas (T não definido fora do equilíbrio)."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            self.assertIsNone(cam['12']['T'],
                msg=f"adiab 1→2: T deveria ser None  params={params}")
            self.assertIsNone(cam['34']['T'],
                msg=f"adiab 3→4: T deveria ser None  params={params}")

    def test_T_definido_nas_isogoricas(self):
        """T deve ser um array nas isocóricas (termalização)."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            for leg in ['23', '41']:
                self.assertIsNotNone(cam[leg]['T'],
                    msg=f"isocórica {leg}: T não deveria ser None")
                self.assertEqual(len(cam[leg]['T']), 200)

    # ── Invariantes da adiabática ──────────────────────────────────────────

    def test_M_rigorosamente_constante_12(self):
        """M = constante ao longo de toda a adiabática 1→2."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            M = cam['12']['M']
            self.assertAlmostEqual(M.max(), M.min(), places=14,
                msg=f"M variou na adiabática 1→2  params={params}")

    def test_M_rigorosamente_constante_34(self):
        """M = constante ao longo de toda a adiabática 3→4."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            M = cam['34']['M']
            self.assertAlmostEqual(M.max(), M.min(), places=14,
                msg=f"M variou na adiabática 3→4  params={params}")

    def test_S_rigorosamente_constante_12(self):
        """S = constante ao longo de toda a adiabática 1→2."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            S = cam['12']['S']
            self.assertAlmostEqual(S.max(), S.min(), places=14,
                msg=f"S variou na adiabática 1→2  params={params}")

    def test_S_rigorosamente_constante_34(self):
        """S = constante ao longo de toda a adiabática 3→4."""
        for params in CICLOS:
            _, cam, _, _ = self._ciclo(params)
            S = cam['34']['S']
            self.assertAlmostEqual(S.max(), S.min(), places=14,
                msg=f"S variou na adiabática 3→4  params={params}")

    def test_U_linear_em_h_nas_adiabaticas(self):
        """U(h) = −M_frozen·h − 8J·p3  (linear em h, slope = −M_frozen)."""
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            _, cam, _, _ = self._ciclo(params)
            for leg, (h0, T0) in [('12', (hi, Tc)), ('34', (hf, Th))]:
                pops  = _pops_termicas(J, h0, T0)
                h_arr = cam[leg]['h']
                U_arr = cam[leg]['U']
                M_froz = float(2*pops[1] - 2*pops[2])
                U_exp  = -h_arr * M_froz - 8*J * float(pops[3])
                np.testing.assert_allclose(U_arr, U_exp, rtol=1e-12,
                    err_msg=f"U não linear em h na adiabática {leg}  params={params}")

    # ── Definição térmica dos 4 estados ───────────────────────────────────

    def test_estados_equilibrio_tem_T_definido(self):
        """Estados #1 e #3 (equilíbrio) devem ter T definido; #2 e #4 (fora-equil.) T=None."""
        for params in CICLOS:
            est, _, _, _ = self._ciclo(params)
            self.assertIsNotNone(est['1']['T'],
                msg=f"estado 1 tem T=None  params={params}")
            self.assertIsNotNone(est['3']['T'],
                msg=f"estado 3 tem T=None  params={params}")
            self.assertIsNone(est['2']['T'],
                msg=f"estado 2 deveria ter T=None (fora-equilíbrio)  params={params}")
            self.assertIsNone(est['4']['T'],
                msg=f"estado 4 deveria ter T=None (fora-equilíbrio)  params={params}")

    def test_estados_sao_termicos(self):
        """Estados #1 e #3 são térmicos; #2 e #4 são congelados (#2=pop. de #1, #4=pop. de #3)."""
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            est, _, _, _ = self._ciclo(params)
            # estados de equilíbrio
            for key, (h_val, T_val) in [('1', (hi, Tc)), ('3', (hf, Th))]:
                self.assertAlmostEqual(est[key]['M'],
                    float(magnetizacao(J, h_val, T_val)), places=10,
                    msg=f"M do estado {key} ≠ térmico  params={params}")
                self.assertAlmostEqual(est[key]['S'],
                    float(entropia(J, h_val, T_val)), places=10,
                    msg=f"S do estado {key} ≠ térmico  params={params}")
                self.assertAlmostEqual(est[key]['U'],
                    float(energia_media(J, h_val, T_val)), places=10,
                    msg=f"U do estado {key} ≠ térmico  params={params}")
            # estados fora-do-equilíbrio: congelados das populações do estado anterior
            # #2: M,S congelados de #1 (mesmo M,S); U recalculado em h=hf
            self.assertAlmostEqual(est['2']['M'], est['1']['M'], places=12,
                msg=f"M do estado 2 ≠ M do estado 1 (deveria ser congelado)  params={params}")
            self.assertAlmostEqual(est['2']['S'], est['1']['S'], places=12,
                msg=f"S do estado 2 ≠ S do estado 1 (deveria ser congelado)  params={params}")
            # #4: M,S congelados de #3
            self.assertAlmostEqual(est['4']['M'], est['3']['M'], places=12,
                msg=f"M do estado 4 ≠ M do estado 3 (deveria ser congelado)  params={params}")
            self.assertAlmostEqual(est['4']['S'], est['3']['S'], places=12,
                msg=f"S do estado 4 ≠ S do estado 3 (deveria ser congelado)  params={params}")

    # ── Isocóricas ────────────────────────────────────────────────────────

    def test_h_constante_nas_isogoricas(self):
        """h deve ser constante nas isocóricas (campo fixo)."""
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            _, cam, _, _ = self._ciclo(params)
            np.testing.assert_allclose(cam['23']['h'], hf, rtol=1e-12,
                err_msg=f"h variou na isocórica 2→3  params={params}")
            np.testing.assert_allclose(cam['41']['h'], hi, rtol=1e-12,
                err_msg=f"h variou na isocórica 4→1  params={params}")

    def test_isogorica_23_comeca_no_estado2(self):
        """Isocórica 2→3 deve começar com os valores térmicos do estado 2."""
        for params in CICLOS:
            est, cam, _, _ = self._ciclo(params)
            self.assertAlmostEqual(cam['23']['M'][0], est['2']['M'], places=10)
            self.assertAlmostEqual(cam['23']['S'][0], est['2']['S'], places=10)
            self.assertAlmostEqual(cam['23']['U'][0], est['2']['U'], places=10)

    def test_isogorica_23_termina_no_estado3(self):
        """Isocórica 2→3 deve terminar com os valores do estado 3."""
        for params in CICLOS:
            est, cam, _, _ = self._ciclo(params)
            self.assertAlmostEqual(cam['23']['M'][-1], est['3']['M'], places=10)
            self.assertAlmostEqual(cam['23']['S'][-1], est['3']['S'], places=10)
            self.assertAlmostEqual(cam['23']['U'][-1], est['3']['U'], places=10)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CICLO CLÁSSICO
# ─────────────────────────────────────────────────────────────────────────────

class TestCicloClassico(unittest.TestCase):

    def test_S_constante_nas_adiabaticas(self):
        """S deve ser constante nas adiabáticas clássicas (S = cte por fsolve)."""
        tol = 1e-4
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            _, cam, _, _ = ciclo_classico(J, hi, hf, Tc, Th, PASSOS=200)
            for leg in ['12', '34']:
                h_p = cam[leg]['h']
                T_p = cam[leg]['T']
                S_p = entropia(J, h_p, T_p)
                self.assertLess(S_p.std(), tol,
                    msg=f"S não constante na adiabática {leg}  params={params}")

    def test_h_constante_nas_isogoricas(self):
        """h constante nas isocóricas."""
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            _, cam, _, _ = ciclo_classico(J, hi, hf, Tc, Th)
            np.testing.assert_allclose(cam['23']['h'], hf, rtol=1e-12)
            np.testing.assert_allclose(cam['41']['h'], hi, rtol=1e-12)

    def test_T_definido_em_todos_caminhos(self):
        """T deve ser um array em todos os caminhos do ciclo clássico."""
        for params in CICLOS:
            J, hi, hf, Tc, Th = params
            _, cam, _, _ = ciclo_classico(J, hi, hf, Tc, Th)
            for leg, path in cam.items():
                self.assertIsNotNone(path['T'],
                    msg=f"T é None no ciclo clássico (leg={leg})")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    unittest.main(verbosity=2)
