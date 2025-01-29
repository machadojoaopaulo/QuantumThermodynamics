import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os


#Definindo constantes
BOLTZMANN=1 # boltzmann constant -> Kb
ANTIFERROMAGNETIC_EXCHANGE_COUPLING=20 #J>=0
ENTROPY=1 #S tem que ser constante pq o processo é adibático
TEMPERATURE_INITIAL_GUESS = 1 # melhorando o chute inicial para um valor maior
MAGNETIC_FIELD_INITIAL = (-4) # O valor inicial do campo magnético
MAGNETIC_FIELD_FINAL = 4 #O valor final do campo magnético

def log_detailed_calculation(magnetic_field):
    return (magnetic_field > 1 and magnetic_field < 1.0001)


#definindo a função T(h), o único parâmetro é h pois é a única coisa que vai mudar de h_i até h_f
def Temp(temperature, magnetic_field):
    if (log_detailed_calculation(magnetic_field)):
        print(f"Temperatures [{temperature}] Magnetic Field [{magnetic_field}]")
    Z = (                                                          
        1 + 
        np.exp(8 * ANTIFERROMAGNETIC_EXCHANGE_COUPLING / (BOLTZMANN * temperature)) + 
        (
            2 * 
            np.cosh(2 * magnetic_field / (BOLTZMANN * temperature))
        )
    )
    term1 = (ENTROPY * -1) + (BOLTZMANN * np.log(Z))
    numerator = (                
            (
                8 * ANTIFERROMAGNETIC_EXCHANGE_COUPLING * 
                np.exp(8 * ANTIFERROMAGNETIC_EXCHANGE_COUPLING / (BOLTZMANN * temperature))
            ) + 
            (
                4 * magnetic_field * 
                np.sinh(2 * magnetic_field / (BOLTZMANN * temperature))
            )
        )
    denominator = temperature * Z
    term2 =  numerator / denominator
    calculated_temperature = term1 - term2
    if (log_detailed_calculation(magnetic_field)):
        print(f"Temperatures [{temperature}] Magnetic Field [{magnetic_field}] result[{calculated_temperature}]")
    return calculated_temperature


#definindo os intervalos de h
magnetic_fields = np.linspace(MAGNETIC_FIELD_INITIAL,MAGNETIC_FIELD_FINAL,100_000)
temperatures = []
temperature_guess = TEMPERATURE_INITIAL_GUESS


for magnetic_field in magnetic_fields:
    if (log_detailed_calculation(magnetic_field)):
        print(f"========== MAGNETIC FIELD = {magnetic_field} ==========")
    try:
        calculated_temperature = fsolve(lambda temperature: Temp(temperature, magnetic_field), temperature_guess)
        if (log_detailed_calculation(magnetic_field)):
            print(f"Final Result={calculated_temperature}")
        
        #Update temperature guess with the result of previuos systems
        temperature_guess = calculated_temperature[0]
        
        #Append the system result to the list of temperatures to be plotted
        temperatures.append(calculated_temperature[0])  # Adiciona o valor encontrado para T
    except RuntimeWarning:
        temperatures.append(np.nan)  # Se houver erro, adiciona NaN para evitar falha
    if (log_detailed_calculation(magnetic_field)):
        print(f"========== MAGNETIC FIELD = {magnetic_field} ==========")


plt.figure(figsize=(10,10))

# Incluir a equação no gráfico (opcional)
equation_text = r'$f(T,S,h,J) = -S + K \log\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right) - \frac{8J e^{\frac{8J}{KT}} + 4h \sinh\left(\frac{2h}{KT}\right)}{T\left(1 + e^{\frac{8J}{KT}} + 2 \cosh\left(\frac{2h}{KT}\right)\right)}$'
plt.text(0.3, 0.95, equation_text, fontsize=10, bbox=dict(facecolor='white', alpha=1), clip_on=False, transform=plt.gcf().transFigure)


plt.plot(magnetic_fields, temperatures)

# Adicionar título e rótulos
plt.title(f"Gráfico de T(S,h,J) T_i=[{TEMPERATURE_INITIAL_GUESS}], h_i=[{MAGNETIC_FIELD_INITIAL}] , h_f=[{MAGNETIC_FIELD_FINAL}], J=[{ANTIFERROMAGNETIC_EXCHANGE_COUPLING}],S=[{ENTROPY}],k_b=[{BOLTZMANN}]")
plt.xlabel('h (Campo Magnético)')
plt.ylabel('T (Temperatura)')


# Exibir a grade, a legenda e o gráfico
plt.grid(True)
plt.legend()
plt.show()



