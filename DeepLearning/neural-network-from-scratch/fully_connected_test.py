from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE
    y = full_connected(X, W, B)
    z = y.mean()
    dz_dy = torch.autograd.grad(z, y, retain_graph=True)[0]
    z.backward()

    #Storing analytical gradients
    analytical_dzdx = X.grad.clone()
    analytical_dzdw = W.grad.clone()
    analytical_dzdb = B.grad.clone()

    # Initialize tensors to store numerical gradients
    numerical_dzdx = torch.zeros_like(X)
    numerical_dzdw = torch.zeros_like(W)
    numerical_dzdb = torch.zeros_like(B)

    # Compute numerical gradients for X
    with torch.no_grad():
        # Numerical gradient for X
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                x_plus = X.clone()
                x_minus = X.clone()
                
                x_plus[t, i] += DELTA
                x_minus[t, i] -= DELTA
                
                y_plus = full_connected(x_plus, W, B)
                y_minus = full_connected(x_minus, W, B)
                
                diff = (y_plus - y_minus) / (2 * DELTA)
                numerical_dzdx[t, i] = (dz_dy * diff).sum()
        # Numerical gradient for W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w_plus = W.clone()
                w_minus = W.clone()
                
                w_plus[i, j] += DELTA
                w_minus[i, j] -= DELTA
                
                y_plus = full_connected(X, w_plus, B)
                y_minus = full_connected(X, w_minus, B)
                
                diff = (y_plus - y_minus) / (2 * DELTA)
                numerical_dzdw[i, j] = (dz_dy * diff).sum()
        
        # Numerical gradient for B
        for k in range(B.shape[0]):
            b_plus = B.clone()
            b_minus = B.clone()
            
            b_plus[k] += DELTA
            b_minus[k] -= DELTA
            
            y_plus = full_connected(X, W, b_plus)
            y_minus = full_connected(X, W, b_minus)
            
            diff = (y_plus - y_minus) / (2 * DELTA)
            numerical_dzdb[k] = (dz_dy * diff).sum()
    
    # Computing errors
    e_x = torch.max(torch.abs(analytical_dzdx - numerical_dzdx)).item()
    e_w = torch.max(torch.abs(analytical_dzdw - numerical_dzdw)).item()
    e_b = torch.max(torch.abs(analytical_dzdb - numerical_dzdb)).item()

    # Gradcheck step
    X_check = X.clone().detach().requires_grad_(True)
    W_check = W.clone().detach().requires_grad_(True)
    B_check = B.clone().detach().requires_grad_(True)

    gradcheck_result = torch.autograd.gradcheck(
        full_connected,
        (X_check, W_check, B_check),
        eps=DELTA,
        atol=TOL
    )
    is_correct = (e_x < TOL) and (e_w < TOL) and (e_b < TOL) and gradcheck_result

    err = {
        'dzdx': e_x,
        'dzdw': e_w,
        'dzdb': e_b
    }

    return is_correct, err

if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    torch.save([tests_passed, errors], 'fully_connected_test_results.pt')
    assert tests_passed
    print(errors)
