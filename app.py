"""
app.py — Dash web app: Ciclo de Otto Quântico  M(J,h,T) × S(J,h,T)
Execução local:   python app.py
Deploy HF Spaces: ver Dockerfile na mesma pasta
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

from thermodynamics_functions import (
    magnetizacao, entropia,
    ciclo_classico, ciclo_Q,
    eficiencia_classica, eficiencia_quantica,
)

# ─────────────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    title='Ciclo Otto Quântico',
    meta_tags=[{'name': 'viewport',
                'content': 'width=device-width, initial-scale=1'}],
)
server = app.server  # ponto de entrada para gunicorn

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTLY — lógica de construção da figura 3D
# ═════════════════════════════════════════════════════════════════════════════

def _arrow_cone(fig, hx, Tx, Qz, color, row, col, legendgroup, sizeref, frac=0.48):
    n   = len(hx)
    mid = int(n * frac)
    stp = max(2, n // 10)
    nxt = min(mid + stp, n - 1)
    u = float(hx[nxt] - hx[mid])
    v = float(Tx[nxt] - Tx[mid])
    w = float(Qz[nxt] - Qz[mid])
    nm = (u*u + v*v + w*w) ** 0.5
    if nm < 1e-12:
        return
    fig.add_trace(go.Cone(
        x=[float(hx[mid])], y=[float(Tx[mid])], z=[float(Qz[mid])],
        u=[u/nm], v=[v/nm], w=[w/nm],
        sizemode='absolute', sizeref=sizeref,
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=False,
        legendgroup=legendgroup,
        anchor='tail', opacity=0.95, hoverinfo='skip',
    ), row=row, col=col)


def _add_panel(fig, func, fname, flabel, cb_x,
               J, hi, hf, Ta, Tb, Tmin, Tmax, N,
               row, col, show_thermo, show_qt, arrow_sz):
    first = (col == 1)

    # Superfície
    h_g = np.linspace(0.2, max(hf, hi) + 0.6, N)
    T_g = np.linspace(Tmin, Tmax, N)
    H_g, T_g2 = np.meshgrid(h_g, T_g)
    Z_g = func(J, H_g, T_g2)

    fig.add_trace(go.Surface(
        x=H_g, y=T_g2, z=Z_g,
        colorscale='plasma', opacity=0.28, showscale=True,
        colorbar=dict(title=dict(text=flabel, side='right'),
                      thickness=12, len=0.50, x=cb_x, tickfont=dict(size=10)),
        showlegend=False,
        hovertemplate=f'h=%{{x:.2f}}<br>T=%{{y:.2f}}<br>{fname}=%{{z:.3f}}<extra></extra>',
    ), row=row, col=col)

    Q1 = float(func(J, hi, Ta))
    Q3 = float(func(J, hf, Tb))
    n  = 200
    t  = np.linspace(0, 1, n)

    # ── Ciclo Termodinâmico ──────────────────────────────────────────────
    if show_thermo:
        try:
            est_v, cam_v, _, _ = ciclo_classico(J, hi, hf, Ta, Tb, 200)
            QC = ciclo_Q(func, J, cam_v)
            CLR = ({'12': '#2ca02c', '23': '#d62728', '34': '#ff7f0e', '41': '#1f77b4'}
                   if Ta <= Tb else
                   {'12': '#2ca02c', '23': '#1f77b4', '34': '#ff7f0e', '41': '#d62728'})
            LBLS = {'12': '1C→2C (isentr.)', '23': '2C→3C (isoc.)',
                    '34': '3C→4C (isentr.)', '41': '4C→1C (isoc.)'}
            for leg, clr in CLR.items():
                grp = f'thermo_{leg}'
                p   = cam_v[leg]
                hp  = np.asarray(p['h'], float)
                Tp  = np.asarray(p['T'], float)
                Qp  = np.asarray(QC[leg], float)
                fig.add_trace(go.Scatter3d(
                    x=hp, y=Tp, z=Qp, mode='lines',
                    line=dict(color=clr, width=5), name=LBLS[leg],
                    legendgroup=grp,
                    legendgrouptitle_text='<b>Termodinâmico</b>' if (leg == '12' and first) else None,
                    showlegend=first,
                ), row=row, col=col)
                _arrow_cone(fig, hp, Tp, Qp, clr, row, col, grp, arrow_sz)
            for k, v in est_v.items():
                Qv = float(func(J, v['h'], v['T']))
                fig.add_trace(go.Scatter3d(
                    x=[v['h']], y=[v['T']], z=[Qv],
                    mode='markers+text', text=[v['label']], textposition='top center',
                    marker=dict(size=5, color='black'),
                    name='Estados Termod.', legendgroup='thermo_pts',
                    showlegend=(k == 1 and first),
                ), row=row, col=col)
        except Exception:
            pass

    # ── Ciclo Quântico ───────────────────────────────────────────────────
    if show_qt:
        h12 = np.linspace(hi, hf, n)
        h34 = np.linspace(hf, hi, n)

        for i, (hx, Ty, Qz, lbl) in enumerate([
            (h12, np.full(n, Ta), np.full(n, Q1), '1Q→2Q (adiab.)'),
            (h34, np.full(n, Tb), np.full(n, Q3), '3Q→4Q (adiab.)'),
        ]):
            fig.add_trace(go.Scatter3d(
                x=hx, y=Ty, z=Qz, mode='lines',
                line=dict(color='cyan', width=4), name=lbl,
                legendgroup='qt_adiab',
                legendgrouptitle_text='<b>Quântico</b>' if (i == 0 and first) else None,
                showlegend=(i == 0 and first),
            ), row=row, col=col)
            _arrow_cone(fig, hx, Ty, Qz, 'cyan', row, col, 'qt_adiab', arrow_sz)

        for i, (hx, Ty, Qz, lbl) in enumerate([
            (np.full(n, hf), np.linspace(Ta, Tb, n), Q1 + t*(Q3-Q1), '2Q→3Q (isoc.)'),
            (np.full(n, hi), np.linspace(Tb, Ta, n), Q3 + t*(Q1-Q3), '4Q→1Q (isoc.)'),
        ]):
            fig.add_trace(go.Scatter3d(
                x=hx, y=Ty, z=Qz, mode='lines',
                line=dict(color='cyan', width=3, dash='dash'), name=lbl,
                legendgroup='qt_iso', showlegend=(i == 0 and first),
            ), row=row, col=col)
            _arrow_cone(fig, hx, Ty, Qz, 'cyan', row, col, 'qt_iso', arrow_sz)

        for lbl, hv, Tv, Qv, filled in [
            ('1Q', hi, Ta, Q1, True),  ('3Q', hf, Tb, Q3, True),
            ('2Q', hf, Ta, Q1, False), ('4Q', hi, Tb, Q3, False),
        ]:
            fc = 'cyan' if filled else 'rgba(0,0,0,0)'
            fig.add_trace(go.Scatter3d(
                x=[hv], y=[Tv], z=[Qv],
                mode='markers+text', text=[lbl], textposition='top center',
                marker=dict(size=7, color=fc, symbol='square',
                            line=dict(color='cyan', width=2)),
                name='Estados QT', legendgroup='qt_pts',
                showlegend=(lbl == '1Q' and first),
            ), row=row, col=col)


def build_figure(J, hi, hf, Ta, Tb, Tmin, Tmax, N, arrow_sz, show_thermo, show_qt):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Magnetização  M(J,h,T)', 'Entropia  S(J,h,T)'],
        horizontal_spacing=0.02,
    )
    cam = dict(eye=dict(x=0.9, y=-1.4, z=0.75))

    for func, fname, flabel, cb_x, col in [
        (magnetizacao, 'M', 'M(J,h,T)', 0.46, 1),
        (entropia,     'S', 'S(J,h,T)', 1.01, 2),
    ]:
        _add_panel(fig, func, fname, flabel, cb_x,
                   J, hi, hf, Ta, Tb, Tmin, Tmax, N,
                   row=1, col=col,
                   show_thermo=show_thermo, show_qt=show_qt, arrow_sz=arrow_sz)

    lim = hf / 4
    reg = 'suave' if J < lim else 'dobrado'
    fig.update_layout(
        title=dict(
            text=(f'J={J:.2f} | hᵢ={hi:.1f}, h_f={hf:.1f} | '
                  f'Tₐ={Ta:.2f}, T_b={Tb:.2f} | limiar={lim:.2f} → {reg}'),
            x=0.5, font=dict(size=12, color='white'),
        ),
        scene =dict(xaxis_title='h', yaxis_title='T', zaxis_title='M',
                    camera=cam, bgcolor='rgb(10,10,20)', aspectmode='auto'),
        scene2=dict(xaxis_title='h', yaxis_title='T', zaxis_title='S',
                    camera=cam, bgcolor='rgb(10,10,20)', aspectmode='auto'),
        legend=dict(
            x=1.06, xanchor='left', y=0.95, groupclick='togglegroup',
            tracegroupgap=6,
            bgcolor='rgba(20,20,30,0.85)',
            bordercolor='rgba(180,180,180,0.4)', borderwidth=1,
            font=dict(size=11, color='white'),
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        height=750,
        margin=dict(l=0, r=185, t=60, b=10),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  PAINEL DE INFO — eficiência, modos de operação
# ═════════════════════════════════════════════════════════════════════════════

_COR_MODO = {
    'Motor':        ('#d4edda', '#155724'),
    'Refrigerador': ('#cce5ff', '#004085'),
    'Acelerador':   ('#fff3cd', '#856404'),
    'Aquecedor':    ('#f8d7da', '#721c24'),
    'Indefinido':   ('#e2e3e5', '#383d41'),
}


def _tabela_html(titulo, res):
    bg, fg = _COR_MODO.get(res['modo'], _COR_MODO['Indefinido'])
    eta    = f"{res['eta']:.4f}" if res.get('eta') is not None else '—'
    linhas = [
        ('η / COP',          f'<b>{eta}</b>'),
        ('Q_b  (isoc. h_f)', f"{res['Qin']:+.5f}"),
        ('Q_a  (isoc. h_i)', f"{res['Qout']:+.5f}"),
        ('W₁₂  (adiab.)',    f"{res['Win']:+.5f}"),
        ('W₃₄  (adiab.)',    f"{res['Wout']:+.5f}"),
        ('<b>W_tot</b>',     f"<b>{res['W']:+.5f}</b>"),
    ]
    trs = ''.join(
        f'<tr style="background:rgba(255,255,255,{0.04 if i % 2 else 0})">'
        f'<td style="padding:3px 10px;color:#ccc">{k}</td>'
        f'<td style="padding:3px 10px;font-family:monospace;color:#eee">{v}</td></tr>'
        for i, (k, v) in enumerate(linhas)
    )
    return (
        f'<div style="border:1px solid #444;border-radius:6px;overflow:hidden;'
        f'min-width:255px;font-size:12.5px;background:#0d1117">'
        f'<div style="background:{bg};color:{fg};padding:5px 11px;font-weight:bold">'
        f'{titulo} &nbsp;'
        f'<span style="padding:2px 8px;border-radius:4px;border:1px solid {fg}">'
        f'{res["modo"]}</span></div>'
        f'<table style="width:100%;border-collapse:collapse">{trs}</table></div>'
    )


def build_info_html(J, hi, hf, Ta, Tb):
    def _safe(fn):
        try:
            return fn(J, hi, hf, Ta, Tb, 200)
        except Exception:
            return dict(modo='Indefinido', eta=None,
                        Qin=float('nan'), Qout=float('nan'),
                        Win=float('nan'), Wout=float('nan'), W=float('nan'))

    res_Q = _safe(eficiencia_quantica)
    res_C = _safe(eficiencia_classica)
    eta0  = 1 - hi / hf
    lim   = hf / 4
    reg   = 'suave (E₄ não é GS)' if J < lim else 'dobrado (E₄ é GS)'
    M1 = float(magnetizacao(J, hi, Ta)); M3 = float(magnetizacao(J, hf, Tb))
    S1 = float(entropia(J, hi, Ta));     S3 = float(entropia(J, hf, Tb))

    header = (
        f'<div style="background:#0d1117;color:#adb5bd;padding:7px 13px;'
        f'border-radius:5px;margin-bottom:7px;font-size:12.5px">'
        f'<b style="color:white">J={J:.2f}</b> &nbsp;|&nbsp; '
        f'hᵢ={hi:.1f}, h_f={hf:.1f} &nbsp;|&nbsp; '
        f'Tₐ={Ta:.2f}, T_b={Tb:.2f} &nbsp;|&nbsp; '
        f'h_f/4={lim:.2f} → {reg} &nbsp;|&nbsp; '
        f'η₀=1−hᵢ/h_f=<b style="color:white">{eta0:.4f}</b><br>'
        f'<span style="font-size:11.5px">'
        f'#1: M₁={M1:.4f}, S₁={S1:.4f} &nbsp;&nbsp;'
        f'#3: M₃={M3:.4f}, S₃={S3:.4f} &nbsp;&nbsp;'
        f'ΔM={M1-M3:+.4f}, ΔS={S1-S3:+.4f}'
        f'</span></div>'
    )
    tables = (
        f'<div style="display:flex;gap:14px;flex-wrap:wrap">'
        f'{_tabela_html("Quântico (QT)", res_Q)}'
        f'{_tabela_html("Termodinâmico", res_C)}'
        f'</div>'
    )
    return header + tables


# ═════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

_LABEL = {'color': '#adb5bd', 'fontSize': '12px', 'marginBottom': '3px', 'display': 'block'}
_CARD  = {'background': '#0d1117', 'border': '1px solid #2d2d3d',
          'borderRadius': '8px', 'padding': '14px', 'marginBottom': '10px'}


def _slider(sid, min_, max_, step, val, marks=None):
    if marks is None:
        marks = {min_: {'label': str(min_), 'style': {'color': '#777'}},
                 max_: {'label': str(max_), 'style': {'color': '#777'}}}
    return dcc.Slider(
        id=sid, min=min_, max=max_, step=step, value=val,
        marks=marks,
        tooltip={'placement': 'bottom', 'always_visible': True},
        updatemode='mouseup',
    )


app.layout = html.Div(
    style={'background': '#060614', 'minHeight': '100vh', 'padding': '0 16px',
           'fontFamily': 'system-ui, sans-serif'},
    children=[

        # ── Título ──────────────────────────────────────────────────────
        html.H4(
            '⚛  Ciclo de Otto Quântico — M(J,h,T) × S(J,h,T)',
            style={'color': 'white', 'paddingTop': '14px', 'marginBottom': '10px'},
        ),

        # ── Painel de controle ───────────────────────────────────────────
        html.Div(style=_CARD, children=[
            html.Div(style={'display': 'grid',
                            'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))',
                            'gap': '20px'},
                children=[

                    # Coluna 1: acoplamento e campos
                    html.Div([
                        html.P('Física do sistema',
                               style={'color': 'white', 'fontWeight': 'bold',
                                      'marginBottom': '8px', 'fontSize': '13px'}),
                        html.Label('J  (acoplamento spin-spin)', style=_LABEL),
                        _slider('J', 0.0, 3.0, 0.01, 0.51),
                        html.Label('hᵢ  (campo inicial)', style={**_LABEL, 'marginTop': '12px'}),
                        _slider('hi', 0.1, 5.0, 0.1, 1.0),
                        html.Label('h_f  (campo final)', style={**_LABEL, 'marginTop': '12px'}),
                        _slider('hf', 0.2, 5.0, 0.1, 2.0),
                    ]),

                    # Coluna 2: temperaturas
                    html.Div([
                        html.P('Banhos térmicos',
                               style={'color': 'white', 'fontWeight': 'bold',
                                      'marginBottom': '8px', 'fontSize': '13px'}),
                        html.Label('Tₐ  (banho frio)', style=_LABEL),
                        _slider('Ta', 0.01, 20.0, 0.1, 2.0,
                                {0.01: {'label': '0.01', 'style': {'color': '#777'}},
                                 5:   {'label': '5',    'style': {'color': '#777'}},
                                 10:  {'label': '10',   'style': {'color': '#777'}},
                                 20:  {'label': '20',   'style': {'color': '#777'}}}),
                        html.Label('T_b  (banho quente)', style={**_LABEL, 'marginTop': '12px'}),
                        _slider('Tb', 0.05, 10.0, 0.05, 1.0,
                                {0.05: {'label': '0.05', 'style': {'color': '#777'}},
                                 2.5:  {'label': '2.5',  'style': {'color': '#777'}},
                                 5:    {'label': '5',    'style': {'color': '#777'}},
                                 10:   {'label': '10',   'style': {'color': '#777'}}}),
                    ]),

                    # Coluna 3: superfície
                    html.Div([
                        html.P('Superfície',
                               style={'color': 'white', 'fontWeight': 'bold',
                                      'marginBottom': '8px', 'fontSize': '13px'}),
                        html.Label('T min superfície', style=_LABEL),
                        _slider('Tmin', 0.01, 2.0, 0.01, 0.05,
                                {0.01: {'label': '0.01', 'style': {'color': '#777'}},
                                 0.5:  {'label': '0.5',  'style': {'color': '#777'}},
                                 1.0:  {'label': '1',    'style': {'color': '#777'}},
                                 2.0:  {'label': '2',    'style': {'color': '#777'}}}),
                        html.Label('T max superfície', style={**_LABEL, 'marginTop': '12px'}),
                        _slider('Tmax', 1.0, 30.0, 0.5, 10.0,
                                {1:  {'label': '1',  'style': {'color': '#777'}},
                                 10: {'label': '10', 'style': {'color': '#777'}},
                                 20: {'label': '20', 'style': {'color': '#777'}},
                                 30: {'label': '30', 'style': {'color': '#777'}}}),
                        html.Label('Grade N (pontos da superfície)',
                                   style={**_LABEL, 'marginTop': '12px'}),
                        _slider('Nsurf', 30, 120, 10, 60,
                                {30: {'label': '30', 'style': {'color': '#777'}},
                                 60: {'label': '60', 'style': {'color': '#777'}},
                                 90: {'label': '90', 'style': {'color': '#777'}},
                                 120:{'label':'120', 'style': {'color': '#777'}}}),
                    ]),

                    # Coluna 4: setas + ciclos
                    html.Div([
                        html.P('Visualização',
                               style={'color': 'white', 'fontWeight': 'bold',
                                      'marginBottom': '8px', 'fontSize': '13px'}),
                        html.Label('Tamanho das setas', style=_LABEL),
                        _slider('arrow', 0.02, 0.60, 0.02, 0.12,
                                {0.02: {'label': '0.02', 'style': {'color': '#777'}},
                                 0.3:  {'label': '0.3',  'style': {'color': '#777'}},
                                 0.6:  {'label': '0.6',  'style': {'color': '#777'}}}),
                        html.P('Ciclos visíveis',
                               style={'color': 'white', 'fontWeight': 'bold',
                                      'marginTop': '14px', 'marginBottom': '6px',
                                      'fontSize': '13px'}),
                        dcc.Checklist(
                            id='ciclos',
                            options=[
                                {'label': '  Ciclo Termodinâmico', 'value': 'thermo'},
                                {'label': '  Ciclo Quântico (QT)', 'value': 'qt'},
                            ],
                            value=['thermo', 'qt'],
                            style={'color': 'white', 'fontSize': '13px',
                                   'lineHeight': '2.0'},
                        ),
                    ]),

                ]),
        ]),

        # ── Info panel (atualiza rápido, só física) ──────────────────────
        dcc.Markdown(id='info-panel', dangerously_allow_html=True,
                     style={'marginBottom': '10px'}),

        # ── Gráfico 3D ───────────────────────────────────────────────────
        dcc.Loading(
            children=dcc.Graph(
                id='graph-3d',
                config={'scrollZoom': True, 'displayModeBar': True,
                        'modeBarButtonsToRemove': ['toImage']},
                style={'height': '750px'},
            ),
            type='circle',
            color='#7fdbff',
        ),

        # ── Rodapé ───────────────────────────────────────────────────────
        html.P(
            '🖱  Arrastar = rotacionar  |  Scroll = zoom  |  '
            'Shift+arrastar = pan  |  Clique na legenda = ativar/desativar curva',
            style={'color': '#555', 'fontSize': '12px',
                   'textAlign': 'center', 'padding': '8px 0 16px'},
        ),
    ],
)


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

_PHYS = [Input('J','value'), Input('hi','value'), Input('hf','value'),
         Input('Ta','value'), Input('Tb','value')]


@app.callback(Output('info-panel', 'children'), _PHYS)
def cb_info(J, hi, hf, Ta, Tb):
    return build_info_html(
        J or 0.51, hi or 1.0, hf or 2.0, Ta or 2.0, Tb or 1.0,
    )


@app.callback(
    Output('graph-3d', 'figure'),
    _PHYS + [
        Input('Tmin', 'value'), Input('Tmax', 'value'),
        Input('Nsurf', 'value'), Input('arrow', 'value'),
        Input('ciclos', 'value'),
    ],
)
def cb_graph(J, hi, hf, Ta, Tb, Tmin, Tmax, N, arrow, ciclos):
    ciclos = ciclos or []
    return build_figure(
        J     or 0.51,
        hi    or 1.0,
        hf    or 2.0,
        Ta    or 2.0,
        Tb    or 1.0,
        Tmin  or 0.05,
        Tmax  or 10.0,
        int(N or 60),
        arrow or 0.12,
        'thermo' in ciclos,
        'qt'     in ciclos,
    )


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
