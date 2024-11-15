import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Definindo constantes
K = 1
J = 1
S = 1  # S tem que ser constante porque o processo é adiabático

# Definindo a função Temp(T, h)
def Temp(T, h):
    exp_mais = np.exp((2 * h / (K * T)))
    exp_menos = np.exp(- ((2 * h / (K * T))))
    Z = 1 + np.exp(8 * J / (K * T)) +  (exp_mais + exp_menos )
    term1 = -S + K * np.log(Z)
    numerator = 8 * J * np.exp(8 * J / (K * T)) + (2 * h * (exp_mais - exp_menos))
    denominator = T * Z
    term2 = numerator / denominator
    return term1 - term2

# Definindo os intervalos de h
h_values = np.linspace(-2, 2,10)
T_values = []

# Encontrando os valores de T para cada h
for h in h_values:
    T_initial_guess = 1 # Melhorando o chute inicial para um valor maior
    try:
        # Usando uma função lambda para manter h fixo e encontrar T
        T_solution = fsolve(lambda T: Temp(T, h), T_initial_guess)
        T_values.append(T_solution[0])  # Adiciona o valor encontrado para T
    except RuntimeWarning:
        T_values.append(np.nan)  # Se houver erro, adiciona NaN para evitar falha

# Configuração do gráfico
plt.figure(figsize=(10, 10))
plt.plot(h_values, T_values, label='T vs h')

# Adicionar título e rótulos
plt.title('Gráfico de T(S,h,J)')
plt.xlabel('h (Campo Magnético)')
plt.ylabel('T (Temperatura)')

# Incluir a equação no gráfico (opcional)
equation_text = r'$f(T,S,h,J) = -S + K \log\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right) - \frac{8J e^{\frac{8J}{KT}} + 4h \sinh\left(\frac{2h}{KT}\right)}{T\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right)}$'
plt.text(-1.5, 0.2, equation_text, fontsize=15, bbox=dict(facecolor='white', alpha=1))

# Exibir a grade, a legenda e o gráfico
plt.grid(True)
plt.legend()
plt.show()
