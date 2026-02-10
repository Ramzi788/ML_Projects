import numpy as np
from gradientdescent import gradient_descent

def f(x):
    return x**2 + 10*np.sin(x)

learning_rates = [1, 0.1, 0.01, 0.001]
x0 = 5.0

print("Learning Rate Comparison")
print("-" * 50)

for lr in learning_rates:
    x_min, history = gradient_descent(f, x0, learning_rate=lr, max_iterations=1000)
    iterations = len(history) - 1
    
    print(f"lr={lr:5} → x={x_min:.4f}, f(x)={f(x_min):.4f}, iters={iterations}")

##Observations:
# Learning rate heavely affects gradient descent performance:
#
# lr = 1.0: 
#   - Algorithm DIVERGED - ended at x=880 instead of correct minimum
#   - Steps were too large, causing overshooting and instability
#
# lr = 0.1 (Optimal):
#   - Converged correctly to x=3.84, f(x)=8.32 in only 9 iterations
#   - Fast and stable, best balance between speed and accuracy
#
# lr = 0.01:
#   - Converged to correct answer but took 128 iterations, way too slow.
#   - Inefficient - wastes computation time
#
# lr = 0.001 (Way Too Small):
#   - Did not converge even after 1000 iterations
#   - Steps too tiny, we would need thousands more iterations