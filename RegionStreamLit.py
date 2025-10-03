import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A otimização mais importante!
# Isso impede que o cálculo pesado seja refeito a cada interação.
@st.cache_data
def calcular_regioes(J, hi, hf, passos, temp_max):
    """
    Função que faz o cálculo pesado e retorna a grade de temperaturas e as regiões.
    Separamos o cálculo do plot para usar o cache de forma eficiente.
    """
    Th_values = np.linspace(0.01, temp_max, passos)
    Tc_values = np.linspace(0.01, temp_max, passos)

    # Corrigindo a ordem do meshgrid para corresponder à iteração
    # meshgrid(x, y) -> X tem a forma (len(y), len(x)), Y tem a forma (len(y), len(x))
    Tc_grid, Th_grid = np.meshgrid(Tc_values, Th_values)
    
    regions = np.zeros_like(Tc_grid, dtype=float)
    
    for i in range(passos):  # Iterando sobre Th (linhas)
        for j in range(passos):  # Iterando sobre Tc (colunas)
            
            Thot = Th_grid[i, j]
            Tcold = Tc_grid[i, j]

            # Evita divisões por zero e garante que Thot > Tcold para algumas condições
            if Thot == 0 or Tcold == 0:
                continue

            W, Qin, Qout, _, _ = Otto_Quantico(J, hi, hf, Tcold, Thot)

            # Condição para Máquina
            if W < 0:
                regions[i, j] = 0.1
            # Condição para Refrigerador
            elif (W > 0 and Qout > 0 and Tcold < Thot):
                regions[i, j] = 2.1
                
            # Condição para Refrigerador
            elif (W > 0 and Qin > 0 and Tcold > Thot):
                regions[i,j] = 2.5
                
            # Condição para Acelerador
            elif (W > 0 and Qout < 0 and Qin > 0 and Tcold < Thot):
                regions[i,j] = 1.5          
                
            # Condição para Acelerador
            elif (W > 0 and Qout > 0 and Qin < 0 and Tcold > Thot):
                regions[i, j] = 1.1
                
            # Condição para Aquecedor
            elif (W > 0 and Qin < 0 and Qout < 0):
                regions[i, j] = 3.1

    return Tc_values, Th_values, regions

# --- Funções do Ciclo de Otto (seu código, sem alterações) ---
def Z(J, h, T):
    if T == 0: return np.inf # Evita divisão por zero
    return 1 + np.exp(8*J/T) + 2*np.cosh(2*h/T)

def Win_12(J, hi, hf, Tc):
    Z_val = Z(J,hi,Tc)
    if Z_val == 0 or np.isinf(Z_val): return 0
    return (4*(hi - hf)*np.sinh(2*(hi/Tc)))/Z_val

def Qin_23(J, hi, hf, Tc, Th):
    Z_hi_Tc = Z(J, hi, Tc)
    Z_hf_Th = Z(J, hf, Th)
    if any(z == 0 or np.isinf(z) for z in [Z_hi_Tc, Z_hf_Th]): return 0
    term1 = (((4*hf*np.sinh(2*(hi/Tc))) + (8*J*np.exp(8*(J/Tc))))/Z_hi_Tc)
    term2 = (((4*hf*np.sinh(2*(hf/Th))) + (8*J*np.exp(8*(J/Th))))/Z_hf_Th)
    return term1 - term2

def Wout_34(J, hi, hf, Th):
    Z_val = Z(J,hf,Th)
    if Z_val == 0 or np.isinf(Z_val): return 0
    return (4*(hf - hi)*np.sinh(2*(hf/Th)))/Z_val

def Qout_41(J, hi, hf, Tc, Th):
    Z_hf_Th = Z(J, hf, Th)
    Z_hi_Tc = Z(J, hi, Tc)
    if any(z == 0 or np.isinf(z) for z in [Z_hf_Th, Z_hi_Tc]): return 0
    term1 = (((4*hi*np.sinh(2*(hf/Th))) + (8*J*np.exp(8*(J/Th))))/Z_hf_Th)
    term2 = (((4*hi*np.sinh(2*(hi/Tc))) + (8*J*np.exp(8*(J/Tc))))/Z_hi_Tc)
    return term1 - term2

def Otto_Quantico(J, hi, hf, Tc, Th):
    Wq12 = Win_12(J, hi, hf, Tc)
    Wq34 = Wout_34(J, hi, hf, Th)
    Wq = Wq12 + Wq34
    Qin23 = Qin_23(J, hi, hf, Tc, Th)
    Qout41 = Qout_41(J, hi, hf, Tc, Th)
    return Wq, Qin23, Qout41, 0, 0 # Eficiência não é usada no plot

# --- Interface do Streamlit ---
st.title('Regiões de Funcionamento de um Motor de Otto Quântico')

st.sidebar.header('Parâmetros do Ciclo')
J = st.sidebar.slider('Acoplamento (J)', min_value=0.0, max_value=5.0, value=1.0, step=0.01)
hi = st.sidebar.slider('Campo Magnético Inicial (hi)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
hf = st.sidebar.slider('Campo Magnético Final (hf)', min_value=0.1, max_value=5.0, value=2.0, step=0.1)

st.sidebar.markdown('---')
st.sidebar.header('Parâmetros do Gráfico')
temp_max = st.sidebar.slider('Temperatura Máxima (Eixos)', min_value=1.0, max_value=20.0, value=10.0, step=1.0)
PASSOS = st.sidebar.slider('Resolução (Nº de Passos)', min_value=50, max_value=2000, value=100, step=50)

# Botão para iniciar o cálculo pesado
if st.button('Gerar Gráfico de Regiões'):
    with st.spinner('Calculando regiões... Isso pode levar um momento.'):
        # 1. Chamar a função com cache
        Tc_values, Th_values, regions = calcular_regioes(J, hi, hf, PASSOS, temp_max)

        # 2. Plotar os resultados
        fig, ax = plt.subplots(figsize=(10, 8))

        # Usando um mapa de cores discreto
        cmap = plt.get_cmap('viridis', 4) 
        contour = ax.contourf(Tc_values, Th_values, regions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap)
        
        # Colorbar com os rótulos corretos
        cbar = plt.colorbar(contour, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Máquina', 'Acelerador', 'Refrigerador', 'Aquecedor'])
        
        ax.set_xlabel('Temperatura Fria ($T_c$)', fontsize=14)
        ax.set_ylabel('Temperatura Quente ($T_h$)', fontsize=14)
        ax.set_title('Diagrama de Fases do Ciclo de Otto Quântico', fontsize=16)

        # Adicionar os parâmetros no gráfico
        param_text = (
            f"$J = {J:.2f}$\n"
            f"$h_i = {hi:.2f}$\n"
            f"$h_f = {hf:.2f}$"
        )
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        st.pyplot(fig)
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Gerar Gráfico de Regiões'.")