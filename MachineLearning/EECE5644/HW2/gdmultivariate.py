import numpy as np
from scipy.optimize import approx_fprime

def gradient_descent_multivariate(f, x0, learning_rate=0.01, epsilon=1e-8, 
                                   max_iterations=1000, tolerance=1e-6):
    """
    Gradient descent for multivariate functions.
    
    Returns:
    - x_min: minimum point found
    - history: list of points during optimization
    """
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iterations):
        gradient = approx_fprime(x, f, epsilon)
        
        x_new = x - learning_rate * gradient
        history.append(x_new.copy())
        
        if np.linalg.norm(gradient) < tolerance:
            print(f"Convergence reached at iteration {i}")
            break
        
        x = x_new
    
    return x, history