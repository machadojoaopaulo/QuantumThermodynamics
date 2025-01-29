import numpy as np
import matplotlib.pyplot as plt

# Definir a função S(T)
def S(T, J, K, h):
    Z = 1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T))
    term1 = K * np.log(Z)
    numerator = 8 * J * np.exp(8 * J / (K * T)) + 4 * h * np.sinh(2 * h / (K * T))
    denominator = T * Z
    term2 = numerator / denominator
    return term1 - term2

# Definir os intervalos de T
T_values = np.linspace(-1, 10, 10000)

# Definir os valores para h, K e J que serão usados nas plotagens
h_values = [-4, 1, 4]
K_values = [1]
J_values = [-4, 1, 4]

# Criar o gráfico
plt.figure(figsize=(20, 10))

# Loop para calcular S(T) para cada combinação de h, K e J e plotar os resultados
for h in h_values:
    for K in K_values:
        for J in J_values:
            S_values = S(T_values, J, K, h)
            plt.plot(T_values, S_values, label=f'h = {h}, J = {J}, K = {K}')

# Adicionar título e rótulos
plt.title('Gráfico de S(T) para diferentes valores de h, J e K')
plt.xlabel('T (Temperatura)')
plt.ylabel('S (Entropia)')

# Incluir a equação no gráfico (opcional)
equation_text = r'$S(T) = K \log\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right) - \frac{8J e^{\frac{8J}{KT}} + 4h \sinh\left(\frac{2h}{KT}\right)}{T\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right)}$'
plt.text(6, 0.1, equation_text, fontsize=15, bbox=dict(facecolor='white', alpha=1))

# Exibir a grade, a legenda e o gráfico
plt.grid(True)
plt.legend()
plt.show()
