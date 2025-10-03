# salve este código como app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Título do Dashboard
st.title('Meu Primeiro Dashboard Interativo!')

# Adicionando um controle interativo (slider) na barra lateral
st.sidebar.header('Opções de Controle')
num_pontos = st.sidebar.slider(
    'Selecione o número de pontos:',
    min_value=10, 
    max_value=1000, 
    value=50, 
    step=10
)

# Gerando dados aleatórios com base no slider
st.write(f"Gerando um gráfico com {num_pontos} pontos.")
dados = pd.DataFrame({
    'x': np.random.randn(num_pontos),
    'y': np.random.randn(num_pontos)
})

# Criando e exibindo um gráfico de dispersão (scatterplot)
fig, ax = plt.subplots()
ax.scatter(dados['x'], dados['y'])
ax.set_title('Gráfico de Dispersão Aleatório')
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')

# Exibe o gráfico do Matplotlib no dashboard
st.pyplot(fig)

# Exibe a tabela de dados
st.write("Dados gerados:")
st.dataframe(dados)