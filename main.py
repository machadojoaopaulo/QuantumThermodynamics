import matplotlib
matplotlib.use('TkAgg')  # Definir o backend para 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np

# Definir as constantes
S = 1
K = 1
J = 1

# Definir a função
def f(T, h):
    term1 = -S
    term2 = K * np.log(1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T)))
    numerator = 8 * J * np.exp(8 * J / (K * T)) + 4 * h * np.sinh(2 * h / (K * T))
    denominator = T * (1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T)))
    term3 = numerator / denominator
    return term1 + term2 - term3

# Definir o intervalo de h e T
h_values = np.linspace(1, 2, 100)
T_values = np.linspace(0.1, 50, 100)

# Criar uma grade de valores de h e T
H, T = np.meshgrid(h_values, T_values)

# Calcular f(T, h) para cada ponto na grade
Z = f(T, H)

# Plotar o gráfico de contorno
plt.contourf(H, T, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(T, h)')
plt.title('Gráfico de f(T, h)')
plt.xlabel('h (campo magnético)')
plt.ylabel('T (temperatura)')
plt.grid(True)
plt.show()