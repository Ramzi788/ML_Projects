from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%
    y = generalized_logistic(X, L, U, G)
    y_true = torch.tanh(X)
    err_y = torch.max(torch.abs(y - y_true)).item()
    z = y.mean()
    dz_dy = torch.autograd.grad(z, y, retain_graph=True)[0]
    z.backward()

    analytical_dzdx = X.grad.clone()
    analytical_dzdl = L.grad.clone()
    analytical_dzdu = U.grad.clone()
    analytical_dzdg = G.grad.clone()

    numerical_dzdx = torch.zeros_like(X)
    numerical_dzdl = torch.zeros_like(L)
    numerical_dzdu = torch.zeros_like(U)
    numerical_dzdg = torch.zeros_like(G)

    with torch.no_grad():
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                x_plus = X.clone()
                x_minus = X.clone()
                x_plus[t, i] += DELTA
                x_minus[t, i] -= DELTA
                y_plus = generalized_logistic(x_plus, L, U, G)
                y_minus = generalized_logistic(x_minus, L, U, G) 
                diff = (y_plus - y_minus) / (2 * DELTA)
                numerical_dzdx[t, i] = torch.sum(dz_dy * diff)

        l_plus = L.clone() + DELTA
        l_minus = L.clone() - DELTA
        y_plus = generalized_logistic(X, l_plus, U, G)
        y_minus = generalized_logistic(X, l_minus, U, G)
        diff = (y_plus - y_minus) / (2 * DELTA)
        numerical_dzdl = torch.sum(dz_dy * diff)  

        u_plus = U.clone() + DELTA
        u_minus = U.clone() - DELTA
        y_plus = generalized_logistic(X, L, u_plus, G)
        y_minus = generalized_logistic(X, L, u_minus, G)
        diff = (y_plus - y_minus) / (2 * DELTA)
        numerical_dzdu = torch.sum(dz_dy * diff)  

        g_plus = G.clone() + DELTA
        g_minus = G.clone() - DELTA
        y_plus = generalized_logistic(X, L, U, g_plus)
        y_minus = generalized_logistic(X, L, U, g_minus)
        diff = (y_plus - y_minus) / (2 * DELTA)
        numerical_dzdg = torch.sum(dz_dy * diff)  

    e_x = torch.max(torch.abs(analytical_dzdx - numerical_dzdx)).item()
    e_l = torch.max(torch.abs(analytical_dzdl - numerical_dzdl)).item()
    e_u = torch.max(torch.abs(analytical_dzdu - numerical_dzdu)).item()
    e_g = torch.max(torch.abs(analytical_dzdg - numerical_dzdg)).item()

    X_check = X.clone().detach().requires_grad_(True)
    L_check = L.clone().detach().requires_grad_(True)
    U_check = U.clone().detach().requires_grad_(True)
    G_check = G.clone().detach().requires_grad_(True)   

    gradcheck_result = torch.autograd.gradcheck(
        generalized_logistic,
        (X_check, L_check, U_check, G_check),
        eps=DELTA,
        atol=TOL2 
    )

    is_correct = (err_y < TOL1) and \
             (e_x < TOL2) and (e_l < TOL2) and (e_u < TOL2) and (e_g < TOL2) and \
             gradcheck_result 
    
    err = {
        'dzdx': e_x,
        'dzdl': e_l,
        'dzdu': e_u,
        'dzdg': e_g,
        'y': err_y
    }

    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    torch.save([test_passed, errors], 'generalized_logistic_test_results.pt')
    assert test_passed
    print(errors)
