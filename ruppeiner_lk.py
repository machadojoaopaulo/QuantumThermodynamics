"""
ruppeiner_lk.py — Geometria de Ruppeiner em coordenadas naturais (λ,κ).

λ = 2h/T,  κ = 8J/T   →   Z(λ,κ) = 1 + 2cosh(λ) + exp(κ)

Métrica = Hessiano de lnZ = matriz de Fisher das flutuações.

Fórmulas fechadas:
    g_λλ = 2[cosh(λ)(1+e^κ) + 2] / Z²
    g_κκ = e^κ(1 + 2cosh(λ)) / Z²
    g_λκ = -2sinh(λ)e^κ / Z²
    det(g) = 2e^κ(cosh(λ)+2) / Z³
    R = 1/2  (esfera S²(2), K = 1/4, raio 2)

Identidades estatísticas:
    g_λλ = Var(m)/4 = -(1/2) ∂M/∂λ = (T/4)·χ_h
    g_κκ = p_s(1 - p_s)                  [variância de Bernoulli do singleto]
    g_λκ = (M/2)·p_s                      [cross-correlação M ↔ p_s]
    det(g) = (p_s/4)[Var(m) - p_s·<m²>]

Crossover: h = 4J  ↔  λ = κ  (cruzamento de níveis singleto ↔ m=-2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Observáveis básicos ──────────────────────────────────────────────────────

def _zval(lam, kap):
    return 1.0 + 2.0*np.cosh(lam) + np.exp(kap)


def metric(lam, kap):
    """
    Retorna (g_ll, g_lk, g_kk) — componentes exatas da métrica de Ruppeiner.
    Aceita arrays numpy de qualquer shape.
    """
    Z  = _zval(lam, kap)
    ek = np.exp(kap)
    ch = np.cosh(lam)
    sh = np.sinh(lam)
    Z2 = Z * Z
    g_ll = 2.0 * (ch*(1.0 + ek) + 2.0) / Z2
    g_kk = ek * (1.0 + 2.0*ch) / Z2
    g_lk = -2.0 * sh * ek / Z2
    return g_ll, g_lk, g_kk


def det_g(lam, kap):
    """det(g) = 2·e^κ·(cosh λ + 2) / Z³  — forma fechada."""
    Z  = _zval(lam, kap)
    return 2.0 * np.exp(kap) * (np.cosh(lam) + 2.0) / Z**3


def magnetizacao(lam, kap):
    """M = -4·sinh(λ) / Z  (sinal: M < 0 para h > 0)."""
    return -4.0 * np.sinh(lam) / _zval(lam, kap)


def p_singlet(lam, kap):
    """p_s = e^κ / Z — população do singleto."""
    return np.exp(kap) / _zval(lam, kap)


def var_m(lam, kap):
    """Var(m) = 4·g_λλ = 4·Var(T_λ)."""
    g_ll, _, _ = metric(lam, kap)
    return 4.0 * g_ll


# ── Verificações das identidades ─────────────────────────────────────────────

def check_identities(lam, kap, tol=1e-10):
    """
    Verifica numericamente as três identidades estatísticas.
    Retorna True se todas passam.
    """
    g_ll, g_lk, g_kk = metric(lam, kap)
    Z  = _zval(lam, kap)
    ek = np.exp(kap)
    ps = ek / Z
    M  = magnetizacao(lam, kap)
    m2 = 8.0 * np.cosh(lam) / Z          # <m²>

    ok1 = np.allclose(g_kk, ps*(1 - ps), atol=tol)
    ok2 = np.allclose(g_lk, (M/2)*ps,     atol=tol)
    ok3 = np.allclose(det_g(lam, kap),
                      (ps/4)*(var_m(lam, kap) - ps*m2), atol=tol)
    return ok1, ok2, ok3


# ── Mapas 2D em espaço (λ, κ) ────────────────────────────────────────────────

def _make_grid(lam_max=4.0, kap_max=6.0, n=400):
    lam = np.linspace(0.0, lam_max, n)
    kap = np.linspace(0.0, kap_max, n)
    L, K = np.meshgrid(lam, kap)
    return lam, kap, L, K


def plot_manifold_maps(lam_max=4.0, kap_max=6.0, n=400, save=None):
    """
    4 painéis: √det g, g_λλ, M, p_s  no espaço (λ, κ).
    Linha tracejada branca: crossover h=4J  ↔  λ = κ.
    """
    lam, kap, L, K = _make_grid(lam_max, kap_max, n)

    g_ll, g_lk, g_kk = metric(L, K)
    sqdet = np.sqrt(det_g(L, K))
    M     = magnetizacao(L, K)
    ps    = p_singlet(L, K)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(r'Variedade de Ruppeiner — espaço $(\lambda, \kappa)$'
                 r'  [$\lambda=2h/T$, $\kappa=8J/T$]', fontsize=12)

    data = [
        (sqdet, r'$\sqrt{\det g}$',           'viridis',   None),
        (g_ll,  r'$g_{\lambda\lambda}$',       'plasma',    None),
        (M,     r'$M(\lambda,\kappa)$',        'RdBu_r',   'symmetric'),
        (ps,    r'$p_s = e^\kappa/Z$',         'cividis',   None),
    ]

    for ax, (arr, title, cmap, mode) in zip(axes.flat, data):
        if mode == 'symmetric':
            vmax = np.nanpercentile(np.abs(arr), 99)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm = None
        im = ax.pcolormesh(lam, kap, arr, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # crossover  λ = κ
        cross = np.minimum(lam, kap_max)
        ax.plot(lam[lam <= kap_max], lam[lam <= kap_max],
                'w--', lw=1.4, alpha=0.85, label=r'$\lambda=\kappa$ ($h=4J$)')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$\lambda = 2h/T$')
        ax.set_ylabel(r'$\kappa = 8J/T$')
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    return fig


def plot_crossover_slice(kap_max=6.0, n=500, save=None):
    """
    Corte ao longo da linha de crossover λ = κ = t.
    Mostra √det g, g_λλ, M, p_s como função de t (= κ = λ).
    """
    t = np.linspace(1e-4, kap_max, n)
    g_ll, _, g_kk = metric(t, t)
    sd   = np.sqrt(det_g(t, t))
    M    = magnetizacao(t, t)
    ps   = p_singlet(t, t)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    fig.suptitle(r'Crossover $h=4J$ — corte $\lambda=\kappa=t$', fontsize=12)

    axes[0,0].plot(t, sd,   'C0'); axes[0,0].set_ylabel(r'$\sqrt{\det g}$')
    axes[0,1].plot(t, g_ll, 'C1'); axes[0,1].set_ylabel(r'$g_{\lambda\lambda}$')
    axes[1,0].plot(t, M,    'C2'); axes[1,0].set_ylabel(r'$M$')
    axes[1,1].plot(t, ps,   'C3'); axes[1,1].set_ylabel(r'$p_s$')

    for ax in axes.flat:
        ax.set_xlabel(r'$t = \lambda = \kappa$')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5)

    # marca o pico de √det g
    t_peak = t[np.argmax(sd)]
    axes[0,0].axvline(t_peak, color='red', lw=1.2, ls='--',
                      label=f't*={t_peak:.2f}')
    axes[0,0].legend(fontsize=8)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    return fig


def plot_M_vs_sqdetg(lam_max=4.0, kap_max=6.0, n=200, save=None):
    """
    Scatter M vs √det g no espaço inteiro, colorido por p_s.
    Revela a correlação entre observável magnético e volume de informação.
    """
    lam, kap, L, K = _make_grid(lam_max, kap_max, n)
    M   = magnetizacao(L, K).ravel()
    sd  = np.sqrt(det_g(L, K)).ravel()
    ps  = p_singlet(L, K).ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(M, sd, c=ps, cmap='plasma', s=1, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, label=r'$p_s$')
    ax.set_xlabel(r'$M(\lambda,\kappa)$')
    ax.set_ylabel(r'$\sqrt{\det g}$')
    ax.set_title(r'$M$ vs volume de informação $\sqrt{\det g}$, colorido por $p_s$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    return fig


def volume_integral_by_region(lam_max=5.0, kap_max=5.0, n=800):
    """
    Integra √det g dλ dκ sobre as regiões:
        campo-dominante:   λ > κ
        crossover:         |λ - κ| < 0.5
        singlet-dominante: κ > λ
    Útil para quantificar o peso geométrico de cada fase.
    """
    lam, kap, L, K = _make_grid(lam_max, kap_max, n)
    sd = np.sqrt(det_g(L, K))
    dl = lam[1] - lam[0]
    dk = kap[1] - kap[0]
    dA = dl * dk

    field_mask    = L > K
    singlet_mask  = K > L + 0.5
    cross_mask    = np.abs(L - K) <= 0.5

    total   = float(np.sum(sd) * dA)
    field   = float(np.sum(sd[field_mask]) * dA)
    singlet = float(np.sum(sd[singlet_mask]) * dA)
    cross   = float(np.sum(sd[cross_mask]) * dA)

    print(f"Volume total   ∫√det g dλdκ  = {total:.4f}")
    print(f"  Região campo  (λ>κ)         = {field:.4f}  ({100*field/total:.1f}%)")
    print(f"  Crossover     |λ-κ|<0.5     = {cross:.4f}  ({100*cross/total:.1f}%)")
    print(f"  Singleto      (κ>λ+0.5)     = {singlet:.4f}  ({100*singlet/total:.1f}%)")
    return dict(total=total, field=field, singlet=singlet, crossover=cross)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verifica identidades em ponto arbitrário
    lam0, kap0 = 1.3, 2.1
    ok1, ok2, ok3 = check_identities(lam0, kap0)
    print(f"Identidades em (λ={lam0}, κ={kap0}):")
    print(f"  g_κκ = p_s(1-p_s)   : {ok1}")
    print(f"  g_λκ = (M/2)·p_s    : {ok2}")
    print(f"  det  = (p_s/4)[...]  : {ok3}")
    print()

    lams = np.linspace(0.3, 3.0, 8)
    kaps = np.linspace(0.3, 4.0, 8)
    L2, K2 = np.meshgrid(lams, kaps)
    det2 = det_g(L2, K2)
    print(f"det(g) range: [{det2.min():.5f}, {det2.max():.5f}]")
    print()

    # Integrais de volume por região
    volume_integral_by_region()
    print()

    # Gera figuras
    print("Gerando figuras...")
    plot_manifold_maps(save="ruppeiner_manifold_lk.pdf")
    plot_crossover_slice(save="ruppeiner_crossover.pdf")
    plot_M_vs_sqdetg(save="ruppeiner_M_vs_vol.pdf")
    print("Figuras salvas: ruppeiner_manifold_lk.pdf, ruppeiner_crossover.pdf, ruppeiner_M_vs_vol.pdf")
