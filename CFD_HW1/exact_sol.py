import numpy as np
import matplotlib.pyplot as plt

r=0.01 # epsilon

# Given coefficients (for epsilon = r)
c0 = -7.031594052*10**(-68)
c1 = 7.031594052*10**(-68)
c2 = -1.383896527*10**(-91)
c3 = 3.710200000
c4 = -1.398265322*10**(-174)
c5 = 1.730100000

# Define functions f1, f2, and f3
def f1(x):
    return c0 * np.exp(x/r) + c1 + (2 - r) * x  - x**2 / 2

def f2(x):
    return c2 * np.exp(x/r) + c3 + (r - 2) * x  + x**2 / 2

def f3(x):
    return c4 * np.exp(x/r) + c5

# Generate x values for each interval
x_values_f1 = np.linspace(0, 1.5, 500)
x_values_f2 = np.linspace(1.5, 2, 500)
x_values_f3 = np.linspace(2, 4, 500)

# Calculate function values for each interval
f1_values = f1(x_values_f1)
f2_values = f2(x_values_f2)
f3_values = f3(x_values_f3)

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x_values_f1, f1_values, label='f1(x)', color='blue')
plt.plot(x_values_f2, f2_values, label='f2(x)', color='green')
plt.plot(x_values_f3, f3_values, label='f3(x)', color='red')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title('Plot with $\epsilon$ = 0.01')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('forcing_term_eps_001.png')
