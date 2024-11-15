import numpy as np
import matplotlib.pyplot as plt

# Definir as constantes
S = 1
K = 1
J = 1

# Definir a função que resolve T para cada h
def calculate_T(h, target_f):
    # Usar uma aproximação numérica para resolver a equação
    # Definindo uma função que retorna a diferença para encontrar a raiz
    def equation(T):
        term1 = -S
        term2 = K * np.log(1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T)))
        numerator = 8 * J * np.exp(8 * J / (K * T)) + 4 * h * np.sinh(2 * h / (K * T))
        denominator = T * (1 + np.exp(8 * J / (K * T)) + 2 * np.cosh(2 * h / (K * T)))
        term3 = numerator / denominator
        return term1 + term2 - term3 - target_f

    # Usar um valor inicial para a busca pela raiz
    T_initial_guess = 1.0
    T_solution = None

    # Usar um loop para buscar a raiz numericamente
    for T in np.linspace(0.1, 10, 1000):
        if equation(T) * equation(T + 0.01) < 0:  # Verificando a mudança de sinal
            T_solution = T
            break

    return T_solution

# Valores de h e valores-alvo para f
h_values = np.linspace(-10, 10, 100)
target_f_values = [0,-1,1,-2,2,-3,3]  # Exemplos de valores para f

# Criar o gráfico
plt.figure(figsize=(10, 6))

for target_f in target_f_values:
    T_results = [calculate_T(h, target_f) for h in h_values]
    plt.plot(h_values, T_results, label=f'f = {target_f}')

plt.title('Gráfico de T(h) para diferentes valores de f')
plt.xlabel('h (campo magnético)')
plt.ylabel('T (temperatura)')
plt.legend()
plt.grid(True)
#plt.ylim(0, 500)  # Limitar o eixo y para melhor visualização
plt.show()
