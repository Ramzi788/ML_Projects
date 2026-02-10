import numpy as np
from gradientdescent import gradient_descent
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 10*np.sin(x)

x0 = np.random.uniform(-10, 10)
x_min, x_history = gradient_descent(f, x0, learning_rate=0.1)

print(f"Starting point: x0 = {x0:.6f}")
print(f"Minimum found at: x = {x_min:.6f}")
print(f"Function value: f(x) = {f(x_min):.6f}")

x_plot = np.linspace(-10, 10, 500)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² + 10sin(x)')
plt.plot(x_history, [f(x) for x in x_history], 'ro-', markersize=5, 
         linewidth=1.5, label='Gradient Descent Path')
plt.plot(x0, f(x0), 'g*', markersize=15, label=f'Start (x={x0:.2f})')
plt.plot(x_min, f(x_min), 'r*', markersize=15, label=f'Minimum (x={x_min:.2f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Gradient Descent', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()