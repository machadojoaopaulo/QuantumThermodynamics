"""
plot_magnetization.py
=====================
Figuras de artigo: magnetização M(J,h,T) do sistema quântico de 4 níveis.

Modelo (kB = 1):
    E1 = +2h  (m = +2)
    E2 =  0   (m =  0)
    E3 = -2h  (m = -2)
    E4 = -8J  (m =  0)

    Z(J,h,T) = 1 + 2·cosh(2h/T) + exp(8J/T)
    M(J,h,T) = −4·sinh(2h/T) / Z

Transição de estado fundamental em h = 4J:
    h < 4J → Singlet GS (E4, m = 0)
    h > 4J → Polarized GS (E3, m = −2)
    Gap singleto–tripleto: Δ = 8J − 2h
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Estilo global ─────────────────────────────────────────────────────────────

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
    'lines.linewidth'   : 1.8,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'text.usetex'       : False,
})

# Paleta amigável a daltonismo (Color Brewer Set1/2)
COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'My_Graphics')
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    base = os.path.join(OUT_DIR, name)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Salvo: {base}.pdf / .png")


# ── Modelo físico (vetorizado e numericamente estável) ────────────────────────

def Z(J, h, T):
    """Função de partição Z = 1 + 2·cosh(2h/T) + exp(8J/T)."""
    return 1.0 + 2.0 * np.cosh(2*h/T) + np.exp(8*J/T)


def M(J, h, T):
    """
    Magnetização M = −4·sinh(2h/T) / Z  (estável em baixas temperaturas).

    Utiliza normalização por exp(max(2h/T, 8J/T)) para evitar overflow.
    Limites:
        T→0, polarizado (h>4J): M → −2
        T→0, singleto (J>h/4): M → 0
        T→∞:                   M → 0
    """
    T = np.asarray(T, dtype=float)
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)

    x = 2*h / T   # 2h/T
    y = 8*J / T   # 8J/T
    m = np.maximum(x, y)   # expoente de normalização

    # M = −2·(e^{x−m} − e^{−x−m}) / (e^{−m} + e^{x−m} + e^{−x−m} + e^{y−m})
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        num = np.exp(x - m) - np.exp(-x - m)
        den = np.exp(-m) + np.exp(x - m) + np.exp(-x - m) + np.exp(y - m)
        result = -2.0 * num / den

    return result


def absM(J, h, T):
    return np.abs(M(J, h, T))


# ── Localização do pico de |M|(T) ─────────────────────────────────────────────

def find_Tmax(J, h, T_arr):
    """
    Retorna T* onde |M(J,h,T)| é máximo.
    Usa argmax na grade fina, seguido de refinamento com brentq.
    """
    vals = absM(J, h, T_arr)
    idx  = int(np.argmax(vals))

    # Tenta refinar com brentq se o máximo não está na borda
    if 1 <= idx <= len(T_arr) - 2:
        def dM_dT(T):
            dT = max(T * 1e-5, 1e-7)
            return absM(J, h, T + dT) - absM(J, h, T - dT)

        # Garante que há troca de sinal ao redor do máximo
        lo = T_arr[max(idx - 5, 0)]
        hi = T_arr[min(idx + 5, len(T_arr) - 1)]
        try:
            if dM_dT(lo) > 0 and dM_dT(hi) < 0:
                return brentq(dM_dT, lo, hi, xtol=1e-6)
        except (ValueError, RuntimeError):
            pass

    return float(T_arr[idx])


# ── Figura 1 — Diagrama de níveis E_n(h) ─────────────────────────────────────
# Valida (F): cruzamento de estados fundamentais em h = 4J.

def fig1_energy_levels():
    x = np.linspace(0, 8, 600)   # h/J

    E1 =  2 * x
    E2 =  np.zeros_like(x)
    E3 = -2 * x
    E4 = -8 * np.ones_like(x)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))

    lw_thin = 1.0
    lw_gs   = 2.8

    # Todos os níveis (finos)
    ax.plot(x, E1, color=COLORS[0], lw=lw_thin, label=r'$E_1 = +2h$  ($m=+2$)')
    ax.plot(x, E2, color=COLORS[1], lw=lw_thin, label=r'$E_2 = 0$  ($m=0$)')
    ax.plot(x, E3, color=COLORS[2], lw=lw_thin, label=r'$E_3 = -2h$  ($m=-2$)')
    ax.plot(x, E4, color=COLORS[3], lw=lw_thin, label=r'$E_4 = -8J$  ($m=0$)')

    # Estado fundamental (destacado com linha mais grossa)
    # h/J < 4: GS = E4 (singleto)
    mask_s = x <= 4
    ax.plot(x[mask_s], E4[mask_s], color=COLORS[3], lw=lw_gs, zorder=5)
    # h/J > 4: GS = E3 (polarizado)
    mask_p = x >= 4
    ax.plot(x[mask_p], E3[mask_p], color=COLORS[2], lw=lw_gs, zorder=5)

    # Linha de cruzamento
    ax.axvline(4, color='dimgray', ls='--', lw=1.4, zorder=4,
               label=r'$h/J = 4$ (crossing)')

    # Sombreamento de regiões
    ax.axvspan(0, 4, alpha=0.08, color=COLORS[3])
    ax.axvspan(4, 8, alpha=0.08, color=COLORS[2])

    # Anotações de região
    ax.text(2.0, -6.5, 'Singlet GS\n$(m=0)$', ha='center', va='center',
            fontsize=9, color=COLORS[3],
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS[3], alpha=0.85))
    ax.text(6.0, -4.5, 'Polarized GS\n$(m=-2)$', ha='center', va='center',
            fontsize=9, color=COLORS[2],
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS[2], alpha=0.85))

    # Seta de gap em x = 2 (singleto)
    x0, gap_lo, gap_hi = 2.3, -4.0, -8.0
    ax.annotate('', xy=(x0, gap_hi), xytext=(x0, gap_lo),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax.text(x0 + 0.2, (gap_lo + gap_hi)/2, r'$\Delta = 8J - 2h$',
            fontsize=8, va='center', color='black')

    ax.set_xlabel(r'$h/J$')
    ax.set_ylabel(r'$E_n / J$')
    ax.set_xlim(0, 8)
    ax.set_ylim(-12, 10)
    ax.set_title('Energy levels and ground-state crossing (validates F)')
    ax.legend(fontsize=8.5, loc='upper left')

    fig.tight_layout()
    _save(fig, 'Fig1_energy_levels')


# ── Figura 2 — |M| vs T para vários J/h (h = 1) ──────────────────────────────
# Valida (B), (C), (D), (E): monotone decay (polarized), non-monotone (singlet),
# T→0 limits, T→∞ limit.

def fig2_M_vs_T():
    h     = 1.0
    T_arr = np.linspace(0.02, 15, 3000)

    J_vals   = [0.1,  0.25,  0.5,   1.0]
    regimes  = ['polarized', 'critical', 'singlet', 'singlet']
    ls_styles = ['-', '--', '-', '-']
    labels = [
        r'$J/h = 0.10$  (polarized)',
        r'$J/h = 0.25$  (critical $h=4J$)',
        r'$J/h = 0.50$  (singlet)',
        r'$J/h = 1.00$  (singlet)',
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    tmax_rows = []

    for i, (J, regime, ls, lbl) in enumerate(zip(J_vals, regimes, ls_styles, labels)):
        vals = absM(J, h, T_arr)
        ax.plot(T_arr, vals, color=COLORS[i], lw=1.8, ls=ls, label=lbl)

        if regime == 'singlet':
            Tmax  = find_Tmax(J, h, T_arr)
            Mmax  = float(absM(J, h, Tmax))
            Delta = 8*J - 2*h
            ax.axvline(Tmax, color=COLORS[i], ls=':', lw=1.1, alpha=0.85)
            ax.annotate(
                f'$T_{{\\rm max}}={Tmax:.2f}$',
                xy=(Tmax, Mmax),
                xytext=(Tmax + 0.5, Mmax + 0.05),
                fontsize=8, color=COLORS[i],
                arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=0.8),
            )
            tmax_rows.append((J, h, Tmax, Delta))

    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel(r'$k_B T$')
    ax.set_ylabel(r'$|M|$')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2.1)
    ax.set_title(r'$|M|$ vs $T$, fixed $h=1$ (validates B, C, D, E)')
    ax.legend(loc='upper right')

    fig.tight_layout()
    _save(fig, 'Fig2_absM_vs_T')

    # Relatório numérico
    print()
    print(f"  {'J':>6}  {'h':>4}  {'T_max (num)':>12}  "
          f"{'Δ=8J−2h':>10}  {'T_max/Δ':>9}")
    for J, hv, Tmax, Delta in tmax_rows:
        print(f"  {J:>6.2f}  {hv:>4.1f}  {Tmax:>12.4f}  "
              f"{Delta:>10.4f}  {Tmax/Delta:>9.4f}")

    return tmax_rows


# ── Figura 3 — |M| vs h em dois painéis ──────────────────────────────────────
# Valida (A): |M| cresce monotonicamente com h em ambos os regimes.

def fig3_M_vs_h():
    h_arr  = np.linspace(0.01, 5, 600)
    T_vals = [0.5, 1.0, 2.0, 5.0]
    Tlbls  = [r'$T=0.5$', r'$T=1$', r'$T=2$', r'$T=5$']

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)

    for ax, J, title in zip(
        axes,
        [0.1,  1.0],
        [r'Polarized regime  $J=0.1$',
         r'Singlet regime  $J=1.0$'],
    ):
        for i, (T, Tlbl) in enumerate(zip(T_vals, Tlbls)):
            ax.plot(h_arr, absM(J, h_arr, T), color=COLORS[i], lw=1.8, label=Tlbl)

        # Saturação
        ax.axhline(2.0, color='gray', ls=':', lw=1.2, label=r'$|M|=2$ (saturation)')

        # Linha de transição h = 4J
        h_crit = 4 * J
        if h_crit <= 5:
            ax.axvline(h_crit, color='black', ls='--', lw=1.0, alpha=0.5,
                       label=f'$h=4J={h_crit:.1f}$')

        ax.set_xlabel(r'$h$')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 2.15)
        ax.set_title(title)
        ax.legend(fontsize=8.5, loc='lower right')

    axes[0].set_ylabel(r'$|M|$')
    fig.suptitle(r'$|M|$ vs $h$ — monotonic growth with $h$ (validates A)', y=1.01)

    fig.tight_layout()
    _save(fig, 'Fig3_absM_vs_h')


# ── Figura 4 — Heatmap |M|(h, T) em dois painéis ─────────────────────────────
# Síntese visual de todas as afirmações (A)–(F).

def fig4_heatmap():
    h_arr = np.linspace(0.01, 5, 500)
    T_arr = np.linspace(0.05, 10, 500)
    H, Tg = np.meshgrid(h_arr, T_arr)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    vmin, vmax = 0.0, 2.0
    ims = []

    for ax, J, title in zip(
        axes,
        [0.1,  1.0],
        [r'$J=0.1$ (polarized for $h > 0.4$)',
         r'$J=1.0$ (singlet for $h < 4$)'],
    ):
        Mabs = np.abs(M(J, H, Tg))
        im = ax.pcolormesh(h_arr, T_arr, Mabs,
                           cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ims.append(im)

        # Contorno em |M| = 1
        cs = ax.contour(H, Tg, Mabs, levels=[1.0],
                        colors='white', linewidths=1.4)
        ax.clabel(cs, fmt=r'$|M|=1$', fontsize=8, inline=True)

        # Linha de transição h = 4J
        h_crit = 4 * J
        if h_crit <= 5:
            ax.axvline(h_crit, color='yellow', ls='--', lw=1.4, alpha=0.85,
                       label=f'$h=4J={h_crit:.1f}$')
            ax.legend(fontsize=8, loc='upper right',
                      labelcolor='white',
                      framealpha=0.3)

        ax.set_xlabel(r'$h$')
        ax.set_ylabel(r'$T$')
        ax.set_title(title)

    # Colorbar único à direita
    cbar = fig.colorbar(ims[-1], ax=axes,
                        label=r'$|M(J,h,T)|$', shrink=0.88, pad=0.02)

    fig.suptitle(r'Heatmap $|M|(h,T)$ — synthesis of all claims (A)–(F)', y=1.01)
    fig.tight_layout()
    _save(fig, 'Fig4_heatmap_absM')


# ── Figura extra — |M| vs T para série de J específica ───────────────────────

# Paleta de 6 cores (Dark2 ColorBrewer, amigável a daltonismo)
COLORS_6 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']


def fig_MvsT_J_series(h=1.0, T_end=15.0):
    """
    |M| vs T para J = 0, 0.10, 0.24, 0.35, 0.51, 0.70 com h fixo.

    Transição em J_crit = h/4:
        J < h/4  → regime polarizado  (|M|→2 em T→0)
        J > h/4  → regime singleto    (|M|→0 em T→0, pico em T_max)
    """
    J_vals = [0.00, 0.10, 0.24, 0.35, 0.51, 0.70]
    J_crit = h / 4.0

    # grade densa para capturar picos em T baixo (ex.: J=0.35, Δ=0.8)
    T_arr = np.concatenate([
        np.linspace(0.01, 1.0,  800),
        np.linspace(1.0,  T_end, 2200),
    ])

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    tmax_rows = []

    for i, J in enumerate(J_vals):
        vals = absM(J, h, T_arr)

        if J < J_crit:
            regime = 'polarized'
            Delta_str = ''
        elif J > J_crit:
            Delta = 8*J - 2*h
            regime = 'singlet'
            Delta_str = rf', $\Delta={Delta:.2f}$'
        else:
            regime = 'critical'
            Delta_str = ''

        label = rf'$J={J:.2f}$  ({regime}{Delta_str})'
        ax.plot(T_arr, vals, color=COLORS_6[i], lw=1.8, label=label)

        # Marca pico para curvas no regime singleto
        if regime == 'singlet':
            Tmax = find_Tmax(J, h, T_arr)
            Mmax = float(absM(J, h, Tmax))
            Delta = 8*J - 2*h
            # linha vertical pontilhada
            ax.axvline(Tmax, color=COLORS_6[i], ls=':', lw=1.0, alpha=0.75)
            # ponto no pico
            ax.scatter([Tmax], [Mmax], color=COLORS_6[i],
                       s=40, zorder=5, edgecolors='white', linewidths=0.6)
            tmax_rows.append((J, h, Tmax, Delta))

    # Linha de saturação
    ax.axhline(2.0, color='gray', ls=':', lw=1.0, alpha=0.6)
    ax.text(T_end * 0.97, 2.03, r'$|M|=2$', ha='right', fontsize=8.5, color='gray')

    # Linha de transição J_crit (anotação vertical)
    ax.axvline(0, color='none')   # força início em 0
    ax.set_xlabel(r'$k_B T$')
    ax.set_ylabel(r'$|M|$')
    ax.set_xlim(0, T_end)
    ax.set_ylim(0, 2.2)
    ax.set_title(rf'$|M|$ vs $T$  (fixed $h={h}$,  $J_{{c}}=h/4={J_crit:.2f}$)')
    ax.legend(loc='upper right', fontsize=8.5)

    fig.tight_layout()
    _save(fig, 'Fig_MvsT_Jseries')

    # Relatório terminal
    print()
    print(f"  Transição em J_crit = h/4 = {J_crit:.3f}  (h = {h})")
    print(f"  {'J':>6}  {'T_max (num)':>12}  {'Δ=8J−2h':>10}  {'T_max/Δ':>9}")
    for J, hv, Tmax, Delta in tmax_rows:
        print(f"  {J:>6.2f}  {Tmax:>12.4f}  {Delta:>10.4f}  {Tmax/Delta:>9.4f}")

    return tmax_rows


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sep = '=' * 62
    print(sep)
    print('  Magnetization figures for article — 4-level quantum system')
    print(f'  Output → {OUT_DIR}')
    print(sep)

    print('\n[Fig 1] Energy levels diagram')
    print('  Validates: (F) level crossing at h = 4J')
    fig1_energy_levels()

    print('\n[Fig 2] |M| vs T  (h = 1, several J/h)')
    print('  Validates: (B) monotone decay (polarized)')
    print('             (C) non-monotone with maximum (singlet)')
    print('             (D) T→0 limits  (E) T→∞ limit')
    tmax_rows = fig2_M_vs_T()

    print('\n[Fig 3] |M| vs h  (two panels)')
    print('  Validates: (A) monotone growth with h in both regimes')
    fig3_M_vs_h()

    print('\n[Fig 4] Heatmap |M|(h,T)')
    print('  Validates: synthesis of all claims (A)–(F)')
    fig4_heatmap()

    print()
    print(sep)
    print('  SUMMARY')
    print(sep)
    rows = [
        ('Fig1_energy_levels', '(F)',               'Level crossing at h = 4J'),
        ('Fig2_absM_vs_T',     '(B),(C),(D),(E)',    'T-dependence: mono. decay / non-mono. peak / limits'),
        ('Fig3_absM_vs_h',     '(A)',               '|M| monotone increasing with h'),
        ('Fig4_heatmap_absM',  '(A)–(F)',           'Full (h,T) plane synthesis'),
    ]
    for fname, claims, desc in rows:
        print(f'  {fname:<28} {claims:<15} {desc}')

    if tmax_rows:
        print()
        print('  T_max (singlet regime, h=1) vs gap estimate Δ = 8J − 2h:')
        print(f"  {'J':>6}  {'T_max (num)':>12}  {'Δ':>8}  {'T_max/Δ':>9}")
        for J, h, Tmax, Delta in tmax_rows:
            print(f"  {J:>6.2f}  {Tmax:>12.4f}  {Delta:>8.4f}  {Tmax/Delta:>9.4f}")
    print(sep)
