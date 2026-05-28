"""
plot_magnetization_universal.py
================================
Figura publicável única: |M|(τ) em variáveis adimensionais τ = T/h, j = J/h.

Modelo de 4 níveis (kB = 1):
    E1 = +2h (m=+2),  E2 = 0 (m=0),  E3 = -2h (m=-2),  E4 = -8J (m=0)

Variáveis adimensionais:
    τ = T/h,  j = J/h  →  j_c = 1/4

Equações de estado adimensionais:
    Z(j,τ) = 1 + 2 cosh(2/τ) + exp(8j/τ)
    |M|(j,τ) = 4 sinh(2/τ) / Z

Sugestão de caption:
    "Universal dimensionless magnetization |M| as a function of reduced
    temperature τ = T/h, for several values of the coupling ratio j = J/h.
    The critical ratio j_c = 1/4 separates the polarized regime (j < 1/4),
    where |M| decreases monotonically from saturation, from the singlet
    regime (j > 1/4), where |M| vanishes at zero temperature, rises as the
    gap δ = 8j - 2 is overcome thermally, peaks near τ ≈ δ, and decays at
    high temperature. Open circles mark the numerically determined peak
    positions."
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Estilo — mesmas convenções de plot_magnetization.py ──────────────────────

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset'  : 'stix',
    'font.size'         : 11,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 11,
    'legend.fontsize'   : 9,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'lines.linewidth'   : 2.0,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'text.usetex'       : False,
    'xtick.direction'   : 'in',
    'ytick.direction'   : 'in',
    'xtick.top'         : True,
    'ytick.right'       : True,
    'axes.linewidth'    : 1.0,
    'axes.edgecolor'    : 'black',
})

# Paleta viridis discreta (5 cores, amigável a daltonismo)
_cmap   = plt.cm.viridis
COLORS5 = [_cmap(v) for v in np.linspace(0, 0.90, 5)]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'My_Graphics')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Modelo físico adimensional ────────────────────────────────────────────────

def absM(j, tau):
    """
    |M|(j,τ) — numericamente estável via normalização pelo expoente máximo.

    Assintóticas:
        τ → 0, j < 1/4  →  |M| = 2
        τ → 0, j = 1/4  →  |M| = 1
        τ → 0, j > 1/4  →  |M| = 0
        τ → ∞           →  |M| = 0
    """
    tau = np.asarray(tau, dtype=float)
    x   = 2.0 / tau          # = 2h/T
    y   = 8.0 * j / tau      # = 8J/T
    m   = np.maximum(x, y)   # expoente de normalização

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        num = np.exp(x - m) - np.exp(-x - m)
        den = np.exp(-m) + np.exp(x - m) + np.exp(-x - m) + np.exp(y - m)
        result = 2.0 * num / den

    return result


def peak_position(j):
    """
    Localiza o pico de |M|(j,τ) via minimize_scalar (regime singleto, j > 1/4).

    Retorna
    -------
    (tau_peak, M_peak) se j > 1/4,  (None, None) se monotônica.
    """
    if j <= 0.25:
        return None, None

    res = minimize_scalar(
        lambda tau: -absM(j, tau),
        bounds=(0.01, 50.0),
        method='bounded',
        options={'xatol': 1e-8},
    )
    tau_pk = float(res.x)
    M_pk   = float(absM(j, tau_pk))
    return tau_pk, M_pk


# ── Figura ────────────────────────────────────────────────────────────────────

def make_figure():
    j_vals = [0.00, 0.10, 0.25, 0.50, 1.00]

    labels = [
        r'$j=0$  (polarized, non-interacting)',
        r'$j=0.1$  (polarized)',
        r'$j=1/4$  (critical)',
        r'$j=1/2$  (singlet)',
        r'$j=1$  (deep singlet)',
    ]

    # Grade τ densa em baixas temperaturas (captura picos em τ < 2)
    tau_arr = np.unique(np.concatenate([
        np.geomspace(0.02, 2.0,  500),
        np.linspace(2.0,  15.0, 800),
    ]))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for j, lbl, color in zip(j_vals, labels, COLORS5):
        vals = absM(j, tau_arr)
        ax.plot(tau_arr, vals, color=color, lw=2.0, label=lbl)

        # Marcador de pico para o regime singleto (j > 1/4)
        tau_pk, M_pk = peak_position(j)
        if tau_pk is not None:
            ax.scatter(
                [tau_pk], [M_pk],
                s=55, marker='o',
                facecolors='none',
                edgecolors=color,
                linewidths=1.8,
                zorder=6,
            )

    # Linha de saturação
    ax.axhline(2.0, color='dimgray', ls=':', lw=1.1, alpha=0.65, zorder=1)
    ax.text(14.7, 2.04, 'saturation', ha='right', va='bottom',
            fontsize=8.5, color='dimgray', style='italic')

    # Lembrete de convenção (canto inferior direito, fora das curvas)
    ax.text(
        0.97, 0.04,
        r'$\tau \equiv T/h, \quad j \equiv J/h$',
        transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='lightgray', alpha=0.9),
    )

    ax.set_xlabel(r'$\tau = T/h$')
    ax.set_ylabel(r'$|M|$')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2.15)
    ax.legend(loc='upper right', framealpha=0.9,
              edgecolor='lightgray', borderpad=0.6)

    fig.tight_layout()

    base = os.path.join(OUT_DIR, 'Fig_MvsT_universal')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Salvo: {base}.pdf / .png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sep = '=' * 60
    print(sep)
    print('  Universal |M|(τ)  —  variáveis adimensionais')
    print(f'  Output → {OUT_DIR}')
    print(sep)

    make_figure()

    # ── Tabela de picos (singleto) ────────────────────────────────────────────
    j_singlet = [j for j in [0.00, 0.10, 0.25, 0.50, 1.00] if j > 0.25]

    print()
    print(f"  {'j':>6}  {'τ_peak (num)':>13}  {'δ=8j−2 (theory)':>17}  {'|M|_peak':>10}")
    print('  ' + '-' * 54)
    for j in j_singlet:
        tau_pk, M_pk = peak_position(j)
        delta = 8*j - 2
        print(f"  {j:>6.2f}  {tau_pk:>13.5f}  {delta:>17.5f}  {M_pk:>10.5f}")

    # ── Limites τ → 0 ────────────────────────────────────────────────────────
    print()
    print('  Limites τ → 0  (avaliado em τ = 0.01):')
    j_all = [0.00, 0.10, 0.25, 0.50, 1.00]
    for j in j_all:
        val = absM(j, 0.01)
        print(f"    j = {j:.2f}:  |M|(τ→0) ≈ {val:.5f}")
    print('  Esperado: 2, 2, 1, 0, 0')

    print(sep)
