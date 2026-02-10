import numpy as np
from gdmultivariate import gradient_descent_multivariate
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + x[1]**2 + 10*np.sin(x[0]*x[1])


x0 = np.random.uniform(-5, 5, size=2)

x_min, history = gradient_descent_multivariate(f, x0, learning_rate=0.05, tolerance=1e-4)

print(f"Starting point: x0 = [{x0[0]:.6f}, {x0[1]:.6f}]")
print(f"Minimum found at: x = [{x_min[0]:.6f}, {x_min[1]:.6f}]")
print(f"Function value: f(x) = {f(x_min):.6f}")
print(f"Number of iterations: {len(history) - 1}")


fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + Y**2 + 10*np.sin(X*Y)

surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

history = np.array(history)
x_traj = history[:, 0]
y_traj = history[:, 1]
z_traj = np.array([f(point) for point in history])

ax.plot(x_traj, y_traj, z_traj, 'r-o', linewidth=2, markersize=4, label='GD Trajectory')

ax.scatter(x0[0], x0[1], f(x0), color='green', s=200, marker='*', 
           label=f'Start ({x0[0]:.2f}, {x0[1]:.2f})', zorder=5)
ax.scatter(x_min[0], x_min[1], f(x_min), color='red', s=200, marker='*', 
           label=f'Minimum ({x_min[0]:.2f}, {x_min[1]:.2f})', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('f(x,y)', fontsize=12)
ax.set_title('Gradient Descent', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()