"""
ruppeiner.py — Ruppeiner scalar curvature for the 2-spin Heisenberg Otto cycle.

Metric in (β, h) coordinates (β = 1/T):
    g_00 = T²C,  g_11 = β χ,  g_01 = -∂U/∂h
Scalar curvature: R = 2 R_{0101} / det(g)
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermodynamics_functions import (
    energia_media, calor_especifico, susceptibilidade,
    entropia, ciclo_classico, eficiencia_classica, eficiencia_quantica,
)


# ── Metric ──────────────────────────────────────────────────────────────────

def metric_components(J, h, T):
    """
    (g_ββ, g_βh, g_hh) of the Fisher/Ruppeiner metric.
    Works on numpy arrays of any shape.
    """
    beta = 1.0 / T
    g00 = T**2 * calor_especifico(J, h, T)
    g11 = beta * susceptibilidade(J, h, T)
    dh = 1e-5
    g01 = -(energia_media(J, h + dh, T) - energia_media(J, h - dh, T)) / (2 * dh)
    return g00, g01, g11


# ── Grid curvature ───────────────────────────────────────────────────────────

def ricci_scalar_grid(J, h_min, h_max, T_min, T_max, nh=250, nT=250):
    """
    Compute R(h, T) on a uniform grid.
    Returns (h_arr, T_arr, R_grid) where R_grid has shape [nT, nh].
    Grid: axis 0 = T direction, axis 1 = h direction.
    """
    h_arr = np.linspace(h_min, h_max, nh)
    T_arr = np.linspace(max(T_min, 1e-3), T_max, nT)

    H, T = np.meshgrid(h_arr, T_arr)   # both [nT, nh]
    B = 1.0 / T

    # Metric components
    g00 = T**2 * calor_especifico(J, H, T)          # g_ββ
    g11 = B * susceptibilidade(J, H, T)              # g_hh
    dh_fd = 1e-5
    g01 = -(energia_media(J, H + dh_fd, T) -
            energia_media(J, H - dh_fd, T)) / (2 * dh_fd)   # g_βh

    det_g = g00 * g11 - g01**2
    safe = np.abs(det_g) > 1e-14
    inv = np.where(safe, 1.0 / np.where(safe, det_g, 1.0), np.nan)

    g00i =  g11 * inv
    g01i = -g01 * inv
    g11i =  g00 * inv

    # Partial derivatives of metric components.
    # In our grid: axis 0 = T, axis 1 = h.
    # ∂/∂β = -T² ∂/∂T  (chain rule from β = 1/T).
    def ddb(f):
        return -T**2 * np.gradient(f, T_arr, axis=0)

    def ddh(f):
        return np.gradient(f, h_arr, axis=1)

    # dg[coord][(a,b)] = ∂g_{ab}/∂x^{coord}  (coord 0=β, 1=h)
    dg = [
        {(0,0): ddb(g00), (0,1): ddb(g01), (1,0): ddb(g01), (1,1): ddb(g11)},
        {(0,0): ddh(g00), (0,1): ddh(g01), (1,0): ddh(g01), (1,1): ddh(g11)},
    ]
    ginv = {(0,0): g00i, (0,1): g01i, (1,0): g01i, (1,1): g11i}

    # Christoffel symbols Γ^k_{ij} = ½ Σ_l g^{kl}(∂_i g_{lj}+∂_j g_{li}-∂_l g_{ij})
    def christoffel(k, i, j):
        res = np.zeros_like(g00)
        for l in range(2):
            res += ginv[(k, l)] * (dg[i][(l, j)] + dg[j][(l, i)] - dg[l][(i, j)])
        return 0.5 * res

    Gam = {(k, i, j): christoffel(k, i, j)
           for k in range(2) for i in range(2) for j in range(2)}

    # ∂_{coord} Γ^k_{ij}
    def dGam(k, i, j, coord):
        G = Gam[(k, i, j)]
        return ddb(G) if coord == 0 else ddh(G)

    # Riemann R^k_{101}  (l=1, i=0, j=1):
    # R^k_{lij} = ∂_i Γ^k_{lj} - ∂_j Γ^k_{li} + Σ_m(Γ^k_{im}Γ^m_{lj} - Γ^k_{jm}Γ^m_{li})
    def riemann_101(k):
        res = dGam(k, 1, 1, 0) - dGam(k, 1, 0, 1)
        for m in range(2):
            res += Gam[(k,0,m)] * Gam[(m,1,1)] - Gam[(k,1,m)] * Gam[(m,1,0)]
        return res

    R0_101 = riemann_101(0)
    R1_101 = riemann_101(1)

    # R_{0101} = g_{0k} R^k_{101}
    R0101 = g00 * R0_101 + g01 * R1_101

    # Scalar curvature R = 2 R_{0101} / det(g)
    R = 2.0 * R0101 * inv

    return h_arr, T_arr, R


# ── Cycle path (classical isentropic strokes) ────────────────────────────────

def cycle_path(J, h_lo, h_hi, T_c, T_h, n=120):
    """
    Classical Otto cycle path in (h, T) space (isentropic adiabatics).
    Returns h_path, T_path, estados [(h,T) for states 1-4].
    """
    estados_d, cam, _, _ = ciclo_classico(J, h_lo, h_hi, T_c, T_h, n)
    h_path = np.concatenate([cam[s]['h'] for s in ('12', '23', '34', '41')])
    T_path = np.concatenate([cam[s]['T'] for s in ('12', '23', '34', '41')])
    pts = [(estados_d[k]['h'], estados_d[k]['T']) for k in (1, 2, 3, 4)]
    return h_path, T_path, pts


# ── Curvature integrals ──────────────────────────────────────────────────────

def cycle_curvature_integrals(J, h_lo, h_hi, T_c, T_h,
                               n_path=150, nh=300, nT=300):
    """
    Returns:
        I_line  — ∮ R |ds| (line integral along cycle, Euclidean ds in (h,T))
        I_area  — ∫∫_{interior} R · (1/T²) dT dh  (area integral in T coords)
        eta_QT  — QT efficiency (NaN if not Motor)
        eta_C   — Classical efficiency (NaN if not Motor)
        eta0    — Otto reference  1 - h_lo/h_hi
        delta_QT, delta_C — η - η0
    """
    h_min = h_lo - 0.5
    h_max = h_hi + 0.5
    T_min = T_c * 0.5
    T_max = T_h * 1.8

    h_arr, T_arr, R_grid = ricci_scalar_grid(J, h_min, h_max, T_min, T_max, nh, nT)

    # Clip extremes for interpolation
    R_clip = np.clip(np.nan_to_num(R_grid, nan=0.0), -1e4, 1e4)
    R_interp = RectBivariateSpline(T_arr, h_arr, R_clip, kx=3, ky=3)

    h_path, T_path, pts = cycle_path(J, h_lo, h_hi, T_c, T_h, n_path)

    R_path = R_interp(T_path, h_path, grid=False)
    dh = np.diff(h_path, append=h_path[0])
    dT = np.diff(T_path, append=T_path[0])
    ds = np.sqrt(dh**2 + dT**2)
    I_line = float(np.sum(R_path * ds))

    # Area integral: mask interior using winding number (polygon test)
    H2D, T2D = np.meshgrid(h_arr, T_arr)
    from matplotlib.path import Path
    poly = Path(list(zip(h_path, T_path)))
    pts_flat = np.column_stack([H2D.ravel(), T2D.ravel()])
    mask = poly.contains_points(pts_flat).reshape(H2D.shape)

    # dT dh element, Jacobian to β: multiply by 1/T²
    dT_step = T_arr[1] - T_arr[0]
    dh_step = h_arr[1] - h_arr[0]
    R_area = np.where(mask, R_clip / T2D**2, 0.0)
    I_area = float(np.sum(R_area) * dT_step * dh_step)

    eta0 = 1.0 - h_lo / h_hi

    res_Q = eficiencia_quantica(J, h_lo, h_hi, T_c, T_h, 80)
    eta_QT = res_Q['eta'] if res_Q['modo'] == 'Motor' else float('nan')

    res_C = eficiencia_classica(J, h_lo, h_hi, T_c, T_h, 80)
    eta_C = res_C['eta'] if res_C['modo'] == 'Motor' else float('nan')

    return dict(
        I_line=I_line, I_area=I_area,
        eta_QT=eta_QT, eta_C=eta_C, eta0=eta0,
        delta_QT=eta_QT - eta0, delta_C=eta_C - eta0,
        modo_QT=res_Q['modo'], modo_C=res_C['modo'],
    )


# ── Heatmap with cycle overlay ───────────────────────────────────────────────

def plot_R_heatmap_with_cycle(J, h_lo, h_hi, T_c, T_h,
                               ax, nh=250, nT=250,
                               R_vmax=None, n_path=120):
    """
    Plot R(h,T) heatmap + classical cycle path on axes `ax`.
    Returns the QuadMesh for colorbar attachment.
    """
    h_min = h_lo - 0.5
    h_max = h_hi + 0.5
    T_min = max(T_c * 0.4, 0.02)
    T_max = T_h * 1.6

    h_arr, T_arr, R_grid = ricci_scalar_grid(J, h_min, h_max, T_min, T_max, nh, nT)

    R_plot = np.clip(R_grid, -1e3, 1e3)
    if R_vmax is None:
        R_vmax = min(np.nanpercentile(np.abs(R_plot), 98), 500)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    norm = mcolors.TwoSlopeNorm(vmin=-R_vmax, vcenter=0, vmax=R_vmax)
    mesh = ax.pcolormesh(h_arr, T_arr, R_plot,
                         cmap='RdBu_r', norm=norm, shading='auto')

    # Level-crossing line h = 4J
    h_cross = 4 * J
    if h_min < h_cross < h_max:
        ax.axvline(h_cross, color='white', lw=1.4, ls='--', alpha=0.85,
                   label=f'$h=4J={h_cross:.3f}$')

    # Classical cycle path
    h_path, T_path, pts = cycle_path(J, h_lo, h_hi, T_c, T_h, n_path)
    ax.plot(np.append(h_path, h_path[0]),
            np.append(T_path, T_path[0]),
            'k-', lw=1.8, zorder=5)

    # Arrows for direction (one per stroke)
    stroke_len = n_path
    for s in range(4):
        mid = s * stroke_len + stroke_len // 2
        nxt = mid + 1
        if nxt < len(h_path):
            ax.annotate('', xy=(h_path[nxt], T_path[nxt]),
                        xytext=(h_path[mid], T_path[mid]),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=1.5), zorder=6)

    # State markers
    labels = ['1', '2', '3', '4']
    for (hp, Tp), lbl in zip(pts, labels):
        ax.scatter(hp, Tp, s=55, color='white', edgecolors='black',
                   linewidths=1.5, zorder=7)
        ax.text(hp + 0.04, Tp + 0.02, lbl, fontsize=9,
                color='white', fontweight='bold', zorder=8)

    ax.set_xlabel('$h$', fontsize=11)
    ax.set_ylabel('$T$', fontsize=11)
    limiar = h_hi / 4  # J threshold: E4=-8J crosses E3=-2h_hi when J=h_hi/4
    ax.set_title(
        f'$J={J:.3f}$  (limiar $J=h_{{hi}}/4={limiar:.3f}$)\n'
        f'$h_{{lo}}={h_lo},\\ h_{{hi}}={h_hi},\\ '
        f'T_c={T_c},\\ T_h={T_h}$',
        fontsize=9, pad=4
    )

    return mesh
