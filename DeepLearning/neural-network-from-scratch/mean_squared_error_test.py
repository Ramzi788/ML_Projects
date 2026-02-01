from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%
    y = mean_squared_error(X1, X2)
    z = y 
    dz_dy = torch.autograd.grad(z, y, retain_graph=True)[0]
    z.backward()

    analytical_dzdx1 = X1.grad.clone()
    analytical_dzdx2 = X2.grad.clone()

    numerical_dzdx1 = torch.zeros_like(X1)
    numerical_dzdx2 = torch.zeros_like(X2)

    with torch.no_grad():
        for t in range(X1.shape[0]):
            for i in range(X1.shape[1]):
                x1_plus = X1.clone()
                x1_minus = X1.clone()
                x1_plus[t, i] += DELTA
                x1_minus[t, i] -= DELTA
                y_plus = mean_squared_error(x1_plus, X2)
                y_minus = mean_squared_error(x1_minus, X2)
                numerical_dzdx1[t, i] = (y_plus - y_minus) / (2 * DELTA)

        for t in range(X2.shape[0]):
            for i in range(X2.shape[1]):
                x2_plus = X2.clone()
                x2_minus = X2.clone()
                x2_plus[t, i] += DELTA
                x2_minus[t, i] -= DELTA
                y_plus = mean_squared_error(X1, x2_plus)
                y_minus = mean_squared_error(X1, x2_minus)
                numerical_dzdx2[t, i] = (y_plus - y_minus) / (2 * DELTA)

    err_dzdx1 = torch.max(torch.abs(analytical_dzdx1 - numerical_dzdx1)).item()
    err_dzdx2 = torch.max(torch.abs(analytical_dzdx2 - numerical_dzdx2)).item()

    X1_check = X1.clone().detach().requires_grad_(True)
    X2_check = X2.clone().detach().requires_grad_(True)

    gradcheck_result = torch.autograd.gradcheck(
        mean_squared_error,
        (X1_check, X2_check),
        eps=DELTA,
        atol=TOL
    )
    is_correct = (err_dzdx1 < TOL) and (err_dzdx2 < TOL) and gradcheck_result
    err = {
        'dzdx1': err_dzdx1,
        'dzdx2': err_dzdx2
    }
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    torch.save([tests_passed, errors], 'mean_squared_error_test_results.pt')
    assert tests_passed
    print(errors)
