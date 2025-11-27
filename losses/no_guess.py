import torch
from torch import nn

from utils.train import extract_data
from utils.tensor import to_dense

class NoGuessLoss(nn.Module):
    def __init__(self, sigmoid_normalization: bool = True):
        super().__init__()
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, inputs, targets, data):
        eps = 1e-3
        inputs = to_dense(inputs, targets.shape)
        inputs = self.normalization(inputs)
        gv = extract_data(data, 'gv')
        pvv = extract_data(data, 'pvv')
        inputs = torch.where(gv > eps, inputs, torch.zeros_like(inputs, dtype=inputs.dtype))

        return inputs[pvv < eps].sum() / gv.sum() + 1 - (inputs[pvv > eps].sum() / gv.sum())
