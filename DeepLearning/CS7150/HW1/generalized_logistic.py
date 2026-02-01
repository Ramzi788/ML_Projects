import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        """
        Computes the generalized logistic function

        Arguments
        ---------
        ctx: A PyTorch context object
        x: (Tensor) of size (T x n), the input features
        l, u, and g: (scalar tensors) representing the generalized logistic function parameters.

        Returns
        -------
        y: (Tensor) of size (T x n), the outputs of the generalized logistic operator

        """
        ctx.save_for_backward(x, l, u, g)
        y = l + (u - l) / (1 + torch.exp(-g * x))
        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagate the gradients with respect to the inputs

        Arguments
        ----------
        ctx: a PyTorch context object
        dzdy (Tensor): of size (T x n), the gradients with respect to the outputs y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdl, dzdu, and dzdg: the gradients with respect to the generalized logistic parameters
        """
        x, l, u, g = ctx.saved_tensors
        exp_term = torch.exp(-g * x)
        denom = (1 + exp_term) ** 2
        dydx = (u - l) * g * exp_term / denom
        dydl = 1 - 1 / (1 + exp_term) 
        dydu = 1 / (1 + exp_term)
        dydg = (u-l) * (x * exp_term) / denom

        #Compute gradients 
        dzdx = dzdy * dydx
        dzdl = (dzdy * dydl).sum()  
        dzdu = (dzdy * dydu).sum()  
        dzdg = (dzdy * dydg).sum()
        return dzdx, dzdl, dzdu, dzdg
