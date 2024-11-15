import numpy as np
import matplotlib.pyplot as plt

# Definir as constantes
#h = 1
#K = 0.1
#J = -110

# Definir a função S(T)
def S(T,J,K,h):
    Z = 1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T))
    term1 = K * np.log(Z)
    numerator = 8 * J * np.exp(8 * J / (K * T)) + 4 * h * np.sinh(2 * h / (K * T))
    denominator = T * (Z)
    term2 = numerator / denominator
    return term1 - term2

# Definir o intervalo de T e h
T_values = np.linspace(1, 1000, 10000)

#J_values = np.linspace(1,10,100)
#h_values = np.linspace(-50, 50, 101)  # 5 valores de h entre -2 e 2

# Criar o gráfico
plt.figure(figsize=(20, 10))

#for h in h_values:
#   S_values = S(T_values, h)
#    plt.plot(T_values, S_values, label=f'h = {h}')

#for J in J_values:
S_values = S(T_values,J=1,K=0.005,h=1)



plt.plot(T_values, S_values)

# Adicionar título e rótulos
plt.title('Gráfico de S(T)')
plt.xlabel('T (Temperatura)')
plt.ylabel('S (Entropia)')

# Incluir a equação no gráfico
equation_text = r'$S(T) = K \log\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right) - \frac{(8Je^{\frac{8J}{KT}}) + (4h\sinh\left(\frac{2h}{KT}\right))}{T(1 + e^{\frac{8J}{KT}} + 2\cosh\left(\frac{2h}{KT}\right))}$'
plt.text(5, 0.3101, equation_text, fontsize=20, bbox=dict(facecolor='white', alpha=1))

# Exibir a grade e a legenda
plt.grid(True)
plt.legend()
plt.show()
