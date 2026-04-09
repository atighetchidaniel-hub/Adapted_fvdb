import setup_paths  # noqa: F401
import torch

from losses.dice import WeightedDiceLoss, SparseWeightedDiceLoss
from utils.tensor import to_sparse, to_dense


def rand_tensor(shape=(1, 1, 32, 32, 32), active=0.05):
    B, C, H, W, Z = shape
    tensor = torch.zeros((B, C, H, W, Z), device="cuda")
    mask = torch.rand(B, C, H, W, Z, device="cuda") < active
    tensor[mask] = 1.0
    return tensor


def test():
    shape = (1, 1, 32, 32, 32)
    active = 0.05
    alpha = 0.1

    input_tensor = rand_tensor(shape, active).requires_grad_()
    sparse_input = to_sparse(input_tensor.detach(), "fvdb")
    sparse_input.data.requires_grad_()
    target_tensor = rand_tensor(shape, active)

    # Compare dense Dice, densified sparse input, and true sparse Dice on the
    # same random binary example to catch inconsistencies in the loss port.
    dense_criterion = WeightedDiceLoss(alpha=alpha, debug=True)
    dense_loss = dense_criterion(input_tensor, target_tensor, {})[0]
    print("Dense Loss:", dense_loss.item())

    dense_convert = to_dense(sparse_input, torch.Size(shape))
    dense_convert_loss = dense_criterion(dense_convert, target_tensor, {})[0]
    print("Dense Convert Loss:", dense_convert_loss.item())

    sparse_criterion = SparseWeightedDiceLoss(alpha=alpha, debug=True)
    sparse_loss = sparse_criterion(sparse_input, target_tensor, {})[0]
    print("Sparse Loss:", sparse_loss.item())


if __name__ == "__main__":
    test()
