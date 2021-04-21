import torch
from torch import nn

# Loss Function ---------------------------------------------------------------------------
def KL_Loss(mu, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return loss

def RMSE_Loss(y_pred, y_true):
    mse = nn.MSELoss()
    loss = torch.sqrt(mse(y_pred, y_true))
    return loss

def KL_R_Loss(y_pred, y_true, mu, log_var, r_factor):
    kl_loss = KL_Loss(mu, log_var)
    r_loss = RMSE_Loss(y_pred, y_true)

    return r_factor * r_loss + kl_loss


def Wasserstein(y_pred, y_true):
    return -1 * torch.mean(y_true * y_pred)


class GradientPaneltyLoss(nn.Module):
    def __init__(self):
        super(GradientPaneltyLoss, self).__init__()

    def forward(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)