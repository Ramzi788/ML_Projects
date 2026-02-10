import numpy as np

def gradient_descent(f, x0, learning_rate=0.01, h=1e-5, 
                                max_iterations=1000, tolerance=1e-6):
    """
    Performs gradient descent to learn theta
    Returns the final theta and the history of cost function values
    """
    x = x0
    x_history = [x]

    for i in range(max_iterations):
        derivative = (f(x + h) - f(x)) / h  
        x_upd = x - learning_rate * derivative
        x_history.append(x_upd)  

        if abs(x_history[-1] - x_history[-2]) < tolerance:
            print(f"Convergence reached at iteration {i}")
            break
        x = x_upd

    return x, x_history