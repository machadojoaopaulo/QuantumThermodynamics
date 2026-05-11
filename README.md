---
title: Quantum Otto Cycle Dashboard
emoji: ⚛
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Quantum Thermodynamics

Pesquisa de João Paulo Machado sobre termodinâmica quântica e clássica de sistemas de 2 spins (4 estados) em ciclos de Otto.

---

## Modelo físico

Sistema de 4 estados (2 spins) com função de partição:

```
Z(J, h, T) = 1 + 2·cosh(2h/T) + exp(8J/T)
```

Parâmetros: **J** (acoplamento spin-spin), **h** (campo magnético externo), **T** (temperatura).  
Níveis de energia: `{0, −2h, +2h, −8J}`.

---

## Estrutura do projeto

| Arquivo / Pasta | Descrição |
|---|---|
| `thermodynamics_functions.py` | Biblioteca central — Z, F, U, S, M, χ, C, ciclos, eficiência, energia por nível |
| `test_thermodynamics.py` | Suite de testes (32 testes, todos passando) |
| `quantum_dashboard.py` | Dashboard Dash interativo → `http://127.0.0.1:8050` |
| `Otto_*_Manifold.ipynb` | Notebooks de variedade termodinâmica (M, Z, F, U, S, χ, C, Eff) |
| `Otto_Niveis_*.ipynb` | Análise por nível de autovalor |
| `Otto_Regioes_Manifold.ipynb` | Mapa de modos de operação no plano (Tc × Th) |
| `Otto_Regioes_Fracas_Manifold.ipynb` | Mapa de regiões com condições fracas (M₁ vs M₃, S₁ vs S₃) — Q e C |
| `Otto_Regioes_Fortes_Manifold.ipynb` | Mapa de regiões com condições fortes (M₁,M₂,M₃,M₄) — somente clássico |
| `PhysRevE_Comparison.ipynb` | Comparação sistemática com PhysRevE.104.044133 |

### Pastas de saída geradas pelos notebooks

| Pasta | Notebook |
|---|---|
| `Regioes_Classical/` | `Otto_Regioes_Manifold.ipynb` |
| `Regioes_Fracas/` | `Otto_Regioes_Fracas_Manifold.ipynb` |
| `Regioes_Fortes/` | `Otto_Regioes_Fortes_Manifold.ipynb` |
| `Magnetization_Classical/` | `Otto_M_Manifold.ipynb` |
| `Niveis_Classical/` | `Otto_Niveis_Manifold.ipynb` |
| `Niveis_Ciclo_Classical/` | `Otto_Niveis_Ciclo_Manifold.ipynb` |
| `Niveis_AntiCorr_Classical/` | `Otto_Niveis_AntiCorr_Manifold.ipynb` |
| `Niveis_Eff_Classical/` | `Otto_Niveis_Eff_Manifold.ipynb` |
| `PhysRevE_Comparison/` | `PhysRevE_Comparison.ipynb` |

---

## Modos de operação do ciclo de Otto

| Modo | Condição | Cor |
|---|---|---|
| Motor | W < 0 | Vermelho `#CC0000` |
| Refrigerador | W > 0, fluxo de calor dependente do regime | Azul `#0055BB` |
| Acelerador | W > 0, condições simétricas por regime | Verde `#008833` |
| Aquecedor | W > 0, Qin < 0, Qout < 0 | Laranja `#FF8800` |

> Classificação distingue regime **Th ≥ Tc** e **Tc > Th** — ver `_classifica()` em `Otto_Regioes_Manifold.ipynb`.

---

## Como executar

```bash
# Testes
.venv/bin/python test_thermodynamics.py

# Dashboard
.venv/bin/python quantum_dashboard.py
```

> Usar sempre o Python do `.venv`, nunca o `python3` do sistema.

---

## Diário de trabalho

### 2026-04-19

**Alterações que ficaram**

1. **`Otto_Regioes_Manifold.ipynb` — correção da função `_classifica`**
   - Problema: a classificação anterior não distinguia o regime Tc > Th do regime Th > Tc para Refrigerador e Acelerador.
   - Correção: `_classifica(W, Qin, Qout, th_gt_tc)` agora recebe a máscara `TH >= TC` e aplica condições simétricas por regime.
   - Motor simplificado para `W < 0` (independe dos sinais de Qin/Qout).
   - Ambas as chamadas (`modo_Q` e `modo_C`) atualizadas com `TH >= TC`.

2. **`Otto_Regioes_Manifold.ipynb` — função `gera_regioes(N, J)`**
   - Encapsula toda a pipeline: grade NxN → ciclo quântico (vetorizado) → ciclo clássico (paralelo via joblib) → Fig 1 (lado a lado Q vs C) → Fig 2 (comparação, com % de discordância).
   - Parâmetros opcionais: `hi=1.0`, `hf=2.0`, `T_min=0.05`, `T_max=20.0`, `PASSOS=80`, `salvar=True`.
   - Retorna `(modo_Q, modo_C, TC, TH)` para uso posterior na sessão.
   - Arquivos salvos com nome incluindo `J` e `N` (ex: `fig1_modos_J=1.0_N=1500.png`).
   - Notebook reduzido para 5 células: intro → mkdir → imports → função → chamada.

---

### 2026-04-20

**Alterações que ficaram**

1. **`Otto_Regioes_Fracas_Manifold.ipynb` — novo notebook criado**
   - Analisa o ciclo de Otto com **condições fracas**: a região Motor ($W < 0$) é subdividida em até 9 sub-regiões com base nas comparações $S_1$ vs $S_3$ e $M_1$ vs $M_3$ (estados de equilíbrio 1 e 3).
   - Classificação: `_classifica_fraca(W, S1, S3, M1, M3)` — retorna valores 1.1 a 10.1.
   - Função principal `gera_regioes_fracas(N, J, ...)` — grade N×N, ciclos Q e C vetorizados, Fig 1 (lado a lado) + Fig 2 (comparação), tabela de distribuição percentual por região.
   - Paleta de 10 cores com colormap discreto (`cmap_fracas`).
   - Chamadas para J = 0.0, 0.24, 0.51, 1.0. Saídas em `Regioes_Fracas/`.

2. **`Otto_Regioes_Fracas_Manifold.ipynb` — célula de mini-diagramas M-h (`plota_mh_minimap`)**
   - Para cada sub-região Motor com área significativa, calcula o ciclo clássico no centroide e desenha um mini-diagrama $M \times h$ embutido (`inset_axes`) no mapa de regiões.
   - Cada um dos 4 processos (adiab. 1→2, isoc. 2→3, adiab. 3→4, isoc. 4→1) é plotado na cor do dashboard e recebe uma **seta de direção no ponto médio** via `_seta_meio`.
   - `_cruzamento_adiab(M1, M2, M3, M4, hi, hf)` detecta o cruzamento linear entre as adiabáticas: condição `(M1−M4)·(M2−M3) < 0`, posição exata via $x^* = (M_4-M_1)/[(M_2-M_1)-(M_3-M_4)]$.
   - Cruzamento marcado com **★ dourada** + seta curva interna apontando para o ponto de cruzamento.
   - Se o inset for deslocado por clipping (região próxima da borda), uma seta branca no mapa principal liga o centroide ao inset.

3. **`Otto_Regioes_Fracas_Manifold.ipynb` — legenda extraída como imagem separada**
   - Removidas as legendas embutidas das Fig 1 e Fig 2 para reduzir o tamanho das figuras.
   - Fig 1: `figsize` ajustado de `(15,6)` → `(12,6)`; Fig 2: `(8,6.5)` → `(7,6.5)`.
   - Nova função `plota_legenda_fracas(salvar=True)` gera `Regioes_Fracas/legenda_fracas.png` como figura autônoma (`figsize=(5.5, 4.0)`).
   - Nova célula de chamada inserida após a célula principal.

4. **`Otto_Regioes_Fracas_Manifold.ipynb` — visualização do ciclo M-h (Q vs C)**
   - Nova célula final com diagrama $M \times h$ comparando ciclo quântico e clássico para um ponto representativo.
   - Ciclo quântico: adiabáticas horizontais ($M_2 = M_1$, $M_4 = M_3$) — conservação de $M$ no processo isentrópico quântico.
   - Ciclo clássico: adiabáticas inclinadas via `_adiab_classica` — $M$ muda, permitindo cruzamento das adiabáticas.
   - `_cruzamento_adiab_local` redefinida localmente na célula para evitar dependência de ordem de execução.

5. **`Otto_Regioes_Fortes_Manifold.ipynb` — novo notebook criado (somente clássico)**
   - Analisa as **condições fortes**: usa todas as quatro magnetizações $M_1, M_2, M_3, M_4$ para subdividir a região Motor.
   - Justificativa: no quântico, $M_2 = M_1$ e $M_4 = M_3$ (adiabático conserva populações), então as condições fortes colapsam para as fracas — análise forte é exclusiva do caso clássico.
   - **18 regiões**: `0.5` (motor degenerado), `1.5` (não-motor), `3.5`–`10.5` (8 sub-regiões com $S_1 > S_3$), `11.5`–`18.5` (8 sub-regiões com $S_1 < S_3$).
   - Paleta bipartida: tons quentes (vermelho/laranja/amarelo) para $S_1 > S_3$; tons frios (azul/verde) para $S_1 < S_3$.
   - Classificação via `_classifica_forte(W, S1, S3, M1, M2, M3, M4)` — 4 comparações binárias: $S_1$ vs $S_3$, $M_1$ vs $M_4$, $M_2$ vs $M_3$, $M_1$ vs $M_3$.
   - Função principal `gera_regioes_fortes(N, J, ...)` — calcula $T_{a2}$ e $T_{b4}$ via `_adiab_classica` com joblib; $M_2$ shape `(1,N)`, $M_4$ shape `(N,1)` — broadcasting correto com a grade `(N,N)`.
   - Eixos rotulados $T_a$ (x) e $T_b$ (y); variáveis internas: `Ta_arr`, `Tb_arr`, `Ta2_arr`, `Tb4_arr`.
   - Saídas em `Regioes_Fortes/`. Chamadas para J = 0.0, 0.24, 0.51, 0.7, 1.0.

6. **`Otto_Regioes_Fortes_Manifold.ipynb` — correção do bug de copiar-e-colar na classificação**
   - O bloco $S_1 < S_3$ da `_classifica_forte` tinha condições duplicadas: regiões 11.5 e 13.5 com condições idênticas, assim como 12.5 e 14.5.
   - Faltavam as variantes `M1<M3` para o subgrupo `M1>M4` no bloco $S_1 < S_3$.
   - Corrigido para espelhar o bloco $S_1 > S_3$: 11.5 (`M2>M3, M1>M3`), 12.5 (`M2>M3, M1<M3`), 13.5 (`M2<M3, M1>M3`), 14.5 (`M2<M3, M1<M3`).

---

---

### 2026-04-23

**Alterações que ficaram**

1. **`Otto_Cycle_Diagrams.ipynb` — novo notebook criado**
   - Quatro funções de plot do ciclo de Otto: `plot_ciclo_Mxh`, `plot_ciclo_MxT`, `plot_ciclo_Sxh`, `plot_ciclo_SxT`.
   - `PASSOS` (precisão numérica) desacoplado de `n_arrows` (número de setas visuais por segmento).
   - Setas apenas no interior de cada segmento (`np.linspace(..., n_arrows+2)[1:-1]`), sem setas nos pontos de canto.
   - `mutation_scale=20` para setas ligeiramente maiores.
   - Cores das isocóricas condicionais ao regime: `Th ≥ Tc` → 2→3 vermelho (aquecimento) e 4→1 azul (resfriamento); `Tc > Th` → invertido.
   - Sem legenda de curvas em nenhuma das quatro funções.
   - Detecção automática do cruzamento das adiabáticas no diagrama M×h (`_find_adiab_crossing`): marca com ★ dourada e caixa de anotação no canto superior esquerdo com $h^*$, $M^*$, $S^*_{12}$, $S^*_{34}$, $T^*_{12}$, $T^*_{34}$ em notação LaTeX.
   - Texto todo em inglês (Magnetization, Entropy, External Field, Temperature, Engine, Refrigerator, Accelerator, Heater).
   - Célula 2×2 (`Painel`) e célula de exemplo com cruzamento adiabático (`J=0.51, Tc=2, Th=1`).

2. **Renomeação de arquivos de saída — compatibilidade LaTeX**
   - Todos os arquivos em `Regioes_Classical/`, `Regioes_Fracas/` e `Regioes_Fortes/` renomeados: removidos `=` e `.` dos nomes (ex: `fig1_modos_J=0.24_N=10000.png` → `fig1_modos_J024_N10000.png`).
   - Motivação: `pdftex.def` rejeita arquivos cujo nome contém `=` ou `.` antes da extensão.

3. **`Otto_Regioes_Fortes_Manifold.ipynb` — célula de painel 2×2**
   - Nova célula ao final do notebook: plota as regiões com condições fortes para J = 0, 0.24, 0.51 e 0.70 lado a lado num painel 2×2.
   - N=5000 por eixo, joblib em paralelo, figsize=(12, 10).
   - Salva `Regioes_Fortes/fig_fortes_painel_N5000.png` e `.pdf`.

---

## Pendente / Próximos passos

- [ ] Verificar visualmente se os mapas de regiões com a nova classificação estão coerentes com as Fig. 3 do PhysRevE.104.044133 (especialmente a região Tc > Th).
- [ ] Rodar `gera_regioes` para diferentes valores de J (ex: 0, 0.24, 0.51, 0.7, 1.0) e comparar os mapas resultantes.
- [ ] Avaliar se os outros notebooks que usam classificação de modos (`PhysRevE_Comparison.ipynb`, `Otto_Niveis_Eff_Manifold.ipynb`) precisam da mesma correção de Tc vs Th.
