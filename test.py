from scipy.optimize import fsolve

# Define the function representing the system of equations
def my_equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 4   # Example equation 1
    eq2 = x * y - 1         # Example equation 2
    print(f"vars[{vars}], eq1[{eq1}], eq2[{eq2}]")
    return [eq1, eq2]

# Initial guess for the solution
initial_guess = [1, 1]

# Use fsolve to find the roots
solution = fsolve(my_equations, initial_guess)

print("The solution is:", solution)