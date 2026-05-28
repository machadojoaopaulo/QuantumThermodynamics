# Perguntas Abertas — Análise por Nível de Energia

## Pergunta 1 — Participação relativa do nível −8J  ✅ RESPONDIDA (2026-04-08)

**Definição implementada:**
```
P^X_{-8J} = |X_{-8J}| / (|X_{-2h}| + |X_{+2h}| + |X_{-8J}|)
```
onde X ∈ {W (trabalho), Q23 (isocórica quente), Q41 (isocórica fria)}.

**Resultados (J=0.51, N=250, T∈[0.1,20]):**

| Canal | Clássico: mediana | máx | fração>50% | Quântico |
|---|---|---|---|---|
| P^W   | 0.310 | 0.920 | 1.3% | **0 (identicamente)** |
| P^Q23 | 0.317 | 0.953 | 6.9% | mediana=0.624, **100%>50%** |
| P^Q41 | **0.746** | 0.782 | **100%** | mediana=0.768, **100%>50%** |

**Conclusões:**
- **Clássico P^{Q41}**: o nível −8J domina Q41 (banho frio) em **100% do Motor** — é o canal principal de resfriamento isocórico.
- **Quântico**: −8J domina **ambas** as isocóricas (Q23 e Q41) em 100% dos pontos, embora não faça trabalho (P^W=0).
- **Clássico P^W**: menor participação; fica acima de 50% só numa região restrita (J grande, Th≫Tc).
- Figura: `fig10_participacao_relativa.png`

---

## Pergunta 2 — Canal invertido do nível −8J nas isocóricas  ✅ RESPONDIDA (2026-04-08)

**Condição:**
```
Q23_{-8J} < 0  (emite para o banho QUENTE)   AND
Q41_{-8J} > 0  (absorve do banho FRIO)
```

**Resultados (J=0.51):**

| Modelo | % pontos Motor | Fração |Q23_{-8J}| inv. | Fração |Q41_{-8J}| inv. |
|---|---|---|---|
| Clássico | **1.6%** | 1.78% | 1.58% |
| Quântico | **0.0%** | 0.00% | 0.00% |

**Conclusões:**
- **O canal invertido existe no clássico** (J=0.51): há uma região de (Tc,Th) onde o nível −8J funciona como um sub-ciclo refrigerador dentro do motor — transportando calor do banho frio para o quente.
- A região é pequena (~1.6% dos pontos Motor), mas fisicamente real.
- **No quântico o canal invertido não existe**: as isocóricas quânticas nunca satisfazem ambas as condições simultaneamente (as isocóricas quânticas usam pops congeladas de #1 e #3, o que impede essa configuração).
- Figura: `fig11_canal_invertido.png`

---

## Estado das perguntas

- Participação relativa P_{−8J}: ✅ **respondida** (Fig 10)
- Canal invertido simultâneo em Q23 e Q41: ✅ **respondida** (Fig 11)
- Todas as figuras em `Niveis_AntiCorr_Classical/fig10_*.png` e `fig11_*.png`
- Células 22–25 adicionadas ao `Otto_Niveis_AntiCorr_Manifold.ipynb`
