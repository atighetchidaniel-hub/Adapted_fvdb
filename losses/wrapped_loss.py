import torch

try:
    import spconv.pytorch as spconv
except Exception:
    spconv = None

from utils.tensor import to_dense


def get_loss_module(loss_type: torch.nn.Module) -> torch.nn.Module:
    class WrappedLoss(loss_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, input, target: torch.Tensor, data: dict = {}) -> torch.Tensor:
            """
            Args:
                input (Tensor): Input tensor of shape (B, C, H, W, Z).
                target (Tensor): Target tensor of shape (B, C, H, W, Z).
            """
            input = to_dense(input, target.shape)
            return super().forward(input, target)

    return WrappedLoss
