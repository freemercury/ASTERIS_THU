
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonGaussianLoss(nn.Module):
    """
    Generalized Anscombe-based loss for Poisson-Gaussian noise.
    Applies a variance-stabilizing transform and minimizes MSE in the transformed domain.
    """

    def __init__(self, alpha=1.0, sigma=0.0, beta=1.0, reduction='mean', eps=1e-8):
        """
        Parameters:
        - alpha: Poisson gain (default=1.0)
        - sigma: Gaussian noise std (e.g., read noise)
        - beta: scaling factor for Gaussian term (default=1.0)
        - reduction: 'mean', 'sum', or 'none'
        - eps: small constant for numerical stability
        """
        super(PoissonGaussianLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.reduction = reduction
        self.eps = eps

    def generalized_anscombe(self, x):
        """
        Apply Generalized Anscombe Transform to input tensor x.
        """  
        return 2.0 / (self.alpha ** 0.5) * torch.sqrt(
            torch.clamp(self.alpha * x + 3.0 / 8.0 + self.beta * self.sigma**2, min=self.eps)
        )

    def forward(self, prediction, target):
        """
        Parameters:
        - prediction: model output (should be positive, e.g. use softplus/exp)
        - target: observed noisy image
        Returns:
        - loss value (scalar)
        """
        prediction = prediction.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        
        A_pred = self.generalized_anscombe(prediction)
        A_target = self.generalized_anscombe(target)
        if A_pred.dim() != 4:
            A_pred_mean = A_pred.mean(dim=2)
            A_target_mean = A_target.mean(dim=2)
            loss = (A_pred_mean - A_target_mean) ** 2
        else:
            loss = (A_pred - A_target) ** 2      

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


