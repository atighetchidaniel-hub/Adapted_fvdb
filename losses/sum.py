from torch import nn

from utils.tensor import flatten, expand_as_one_hot


# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class WeightedSumLoss(nn.Module):

    def __init__(self, sigmoid_normalization=True, alpha=0.1):
        super(WeightedSumLoss, self).__init__()
        self.classes = None
        self.skip_index_after = None
        self.alpha = alpha
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def weighted_sum(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        fp = (input * (1 - target)).sum(-1)  # FP = P & ~GT
        fn = ((1 - input) * target).sum(-1)  # FN = ~P & GT
        weighted_sum = fp * self.alpha + fn * (1 - self.alpha)

        return weighted_sum

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = expand_as_one_hot(target, self.classes)

        assert input.dim() == target.dim(
        ) == 5, "'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            print("Target {} after skip index {}".format(
                before_size, target.size()))

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        input = self.normalization(input)

        weighted_sum = self.weighted_sum(input, target)
        return weighted_sum
