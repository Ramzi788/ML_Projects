import numpy as np
from gdmultivariate import gradient_descent_multivariate

def f(x):
    return x[0]**2 + x[1]**2 + 10*np.sin(x[0]*x[1])

print("1. Learning Rate Comparison")
print("-"*50)
for lr in [0.1, 0.05, 0.01, 0.001]:
    x0 = np.array([3.0, -3.0])
    x_min, history = gradient_descent_multivariate(f, x0, learning_rate=lr, max_iterations=1000)
    print(f"lr={lr:5} → iters={len(history)-1:4}, f(x)={f(x_min):7.3f}")

print("\n2. Different Starting Points")
print("-"*50)
for _ in range(5):
    x0 = np.random.uniform(-5, 5, size=2)
    x_min, history = gradient_descent_multivariate(f, x0, learning_rate=0.01, max_iterations=1000)
    print(f"x0=({x0[0]:5.2f},{x0[1]:5.2f}) → x=({x_min[0]:5.2f},{x_min[1]:5.2f}), f(x)={f(x_min):7.3f}")

## MY FINDINGS:

# 1. LEARNING RATE:
#    - lr=0.1: Did NOT converge (1000 iterations), found f(x)=-6.075
#    - lr=0.05: Converged quickly in 17 iterations, found best minimum f(x)=-7.059
#    - lr=0.01: Converged in 26 iterations, but found worse minimum f(x)=5.507
#    - lr=0.001: Converged in 110 iterations, same minimum as lr=0.01 f(x)=5.507
#    Conclusion: Larger steps aren't just faster, they can also escape poor local minima. Too small steps get trapped in nearest minimum regardless of quality.

# 2. STARTING POINTS:
#    - Found 4 different local minima ranging from f(x)=-7.059 to f(x)=20.663
#    - Best minimum: x≈(-1.17, 1.17) or (1.17, -1.17), f(x)≈-7.059
#    - Worst minimum: x≈(3.63, -3.63), f(x)≈20.663
#    - Starting points closer to origin tend to find better minima
#    Conclusion: Gradient descent only finds local minima, not global. The algorithm depends on how we initialize, it rolls downhill to nearest valley.

# 3. MAX ITERATIONS:
#    - Needed 17-404 iterations depending on learning rate and starting point
#    - lr=0.05 converges fastest (17 iters)
#    - lr=0.01 needs 200-400 iterations typically
#    Conclusion: 500-1000 max iterations is safe to make sure of convergence

# 4. OVERALL:
#    - Learning rate affects how fast we converge AND which solution we find
#    - Different starting points lead to different solutions
#    - Need to try multiple random starts to find the best answer
#    - Set max iterations high enough (500-1000) to ensure convergence