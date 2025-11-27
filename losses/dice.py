import torch
from torch import nn
import spconv.pytorch as spconv

from utils.tensor import flatten, expand_as_one_hot, to_dense


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    # Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.classes = None
        self.skip_index_after = None
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight) -> dict[str, torch.Tensor]:
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target, data: dict):
        # Usually we use the sparse version of the loss. Just for debugging
        input = to_dense(input, target.shape)

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

        # target = self.sub_sample(target)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        if self.normalization is not None:
            input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        return loss, {
            "dice": per_channel_dice,
        }


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=4, skip_index_after=None, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


# Code was further adapted and mofified from https://github.com/black0017/MedicalZooPytorch/tree/master

class WeightedDiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    """

    def __init__(self, classes=4, skip_index_after=None, weight=None, sigmoid_normalization=True, alpha=0.1, debug=False):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        self.alpha = alpha
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after
        self.debug = debug

    def dice(self, input, target, weight=None):
        epsilon = 1e-6
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        fp = (input**2 * (1 - target)**2).sum(-1)  # FP = P & ~GT
        fn = ((1 - input)**2 * target**2).sum(-1)  # FN = ~P & GT

        if weight is not None:
            intersect = weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        # denominator = (input * input).sum(-1) + (target * target).sum(-1)
        denominator = (2*intersect) + fp * self.alpha + fn * (1 - self.alpha)

        if self.debug:
            print(
                f"intersect: {intersect}, fp: {fp}, fn: {fn}, denominator: {denominator}")

        return 2 * (intersect / denominator.clamp(min=epsilon))


class SparseWeightedDiceLoss(nn.Module):
    def __init__(self, classes=4, skip_index_after=None, weight=None, sigmoid_normalization=True, alpha=0.1, debug=False):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        self.debug = debug

        # For dense input compatibility during evaluation
        self._dense_dice = WeightedDiceLoss(
            classes=classes, skip_index_after=skip_index_after, weight=weight, sigmoid_normalization=sigmoid_normalization, alpha=alpha, debug=debug)

    def dice(self, input: spconv.SparseConvTensor, target: torch.Tensor, weight: float | None = None) -> torch.Tensor:
        epsilon = 1e-6
        # Extract coordinates from input SparseConvTensor
        coords = input.indices  # Shape: (N, 4) where columns are [batch, x, y, z]

        # Extract features from the target tensor using input's coordinates
        # Target is expected to be a dense tensor of shape (B, C, D, H, W)
        # Convert coordinates to indices for the target tensor
        batch_indices = coords[:, 0].long()
        x = coords[:, 1].long()
        y = coords[:, 2].long()
        z = coords[:, 3].long()

        # Gather the features from target tensor
        # Note: For spconv, spatial dimensions are in order [D, H, W] in the target tensor
        target_features = target[batch_indices, :, x, y, z]

        # Create a sparse tensor using the input's coordinates and the gathered features
        sparse_target = spconv.SparseConvTensor(
            features=target_features,
            indices=coords,
            spatial_shape=input.spatial_shape,
            batch_size=input.batch_size
        )

        # Predicted values (can be soft probabilities)
        input_feat = input.features
        target_feat = sparse_target.features.type(input_feat.dtype)  # Target values

        # Compute per-channel sums over all sparse points.
        intersect = (input_feat * target_feat).sum(dim=0)

        # Squared terms in FP and FN as in your original implementation.
        fp = ((input_feat ** 2) * ((1 - target_feat) ** 2)
              ).sum(dim=0)  # FP = prediction & ~target
        fn = (((1 - input_feat) ** 2) * (target_feat ** 2)
              ).sum(dim=0)   # FN = ~prediction & target

        # Optionally weight the intersect (e.g., for class balancing).
        if weight is not None:
            intersect = weight * intersect

        # Calculate the denominator using your formulation.
        denominator = (2 * intersect) + (self.alpha * fp) + \
            ((1 - self.alpha) * fn)

        if self.debug:
            print(
                f"intersect: {intersect}, fp: {fp}, fn: {fn}, denominator: {denominator}")

        # Compute the Dice coefficient (per channel).
        dice = 2 * (intersect / denominator.clamp(min=epsilon))
        return dice

    def forward(self, input, target, data: dict = {}):
        """
        Expand to one hot added extra for consistency reasons
        """
        if not isinstance(input, spconv.SparseConvTensor):
            # Fallback to dense Dice loss for dense inputs (e.g., during evaluation)
            return self._dense_dice(input, target, data)
        
        # get probabilities from logits
        if self.normalization is not None:
            # Apply normalization to sparse features
            normalized_features = self.normalization(input.features)
            input = input.replace_feature(normalized_features)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        return loss, {
            "dice": per_channel_dice,
        }
