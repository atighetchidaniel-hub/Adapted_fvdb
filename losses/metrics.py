import torch
from torch import nn
import spconv.pytorch as spconv

from utils.tensor import flatten, expand_as_one_hot, to_dense
from utils.train import extract_data


class Metrics(nn.Module):
    def __init__(self, classes=1, sigmoid_normalization=True):
        super().__init__()
        self.classes = classes
        self.skip_index_after = None
        # self.threshold = 0.73105
        self.threshold = 0.5
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def false_positive_negative(self, input, target, data = {}):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        input = (input > self.threshold).float()
        target = flatten(target)

        fp = (input * (1 - target)).sum(-1)  # FP = P & ~GT
        fn = ((1 - input) * target).sum(-1)  # FN = ~P & GT
        tn = ((1 - input) * (1 - target)).sum(-1)
        tp = (input * target).sum(-1)

        fp_ratio = -1
        gv = extract_data(data, 'gv')
        if gv is not None:
            fp_ratio = (fp / gv.sum())

        # denominator = (input * input).sum(-1) + (target * target).sum(-1)
        return {
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'tp': tp,
            'fp_rate': fp / (fp + tn),
            'fn_rate': fn / (fn + tp),
            'fp_ratio': fp_ratio,
        }

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input: torch.Tensor | spconv.SparseConvTensor, target: torch.Tensor, data = {}) -> dict:
        input = to_dense(input, target.shape)

        # Expand to one hot added extra for consistency reasons
        target = expand_as_one_hot(target, self.classes)

        assert input.dim() == target.dim(
        ) == 5, "'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            print("Target {} after skip index {}".format(
                before_size, target.size()))

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # if using amp, make sure the input is float so that `log` does not produce NaN
        input = input.float()
        # get probabilities from logits
        if self.normalization is not None:
            input = self.normalization(input)

        gv = extract_data(data, 'gv')
        if gv is not None:
            # Remove points that are not in the target
            input = torch.where(gv > 0, input, torch.zeros_like(input, dtype=input.dtype))

        metrics = self.false_positive_negative(input, target, data)

        metrics['gv_ratio'] = (input > self.threshold).sum() / gv.sum()
        # Each metric should be converted to numpy array
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().cpu().numpy()
        return metrics
