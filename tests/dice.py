import setup_paths  # noqa: F401
import torch
import spconv.pytorch as spconv
from torchviz import make_dot
from utils.tensor import dense_to_spconv

from losses.dice import WeightedDiceLoss, SparseWeightedDiceLoss


def rand_tensor(shape=(1, 1, 4, 4, 4), active=0.5):
    """
    Generate a random tensor of shape (B, C, H, W, Z) with some values set to 1.
    """
    # Construct B, C, H, W, Z tensor
    B, C, H, W, Z = shape
    tensor = torch.zeros((B, C, H, W, Z))
    # Randomly set some values to 1
    mask = torch.rand(B, C, H, W, Z) < active
    tensor[mask] = 1.0
    return tensor


def test():
    W = 256
    shape = (1, 1, W, W, W)
    active = 0.05
    ALPHA = 0.1

    input_tensor = rand_tensor(shape, active)
    sparse_input = dense_to_spconv(input_tensor)
    sparse_input.features.requires_grad_()
    input_tensor.requires_grad_()
    target_tensor = rand_tensor(shape, active)

    input_tensor = input_tensor * 1  # Create backward graph
    sparse_input = spconv.SparseConvTensor(
        sparse_input.features, sparse_input.indices * 1, 
        sparse_input.spatial_shape, sparse_input.batch_size)

    dense_criterion = WeightedDiceLoss(alpha=ALPHA, debug=True)
    dense_loss = dense_criterion(input_tensor, target_tensor, {})[0]
    print("Dense Loss:", dense_loss.item())

    dense_convert = sparse_input.dense(
        torch.Size(shape))[0]
    dense_convert_loss = dense_criterion(
        dense_convert, target_tensor, ALPHA)[0]
    print("Dense Convert Loss:", dense_convert_loss.item())

    sparse_criterion = SparseWeightedDiceLoss(alpha=ALPHA, debug=True)
    sparse_loss = sparse_criterion(sparse_input, target_tensor, {})[0]
    print("Sparse Loss:", sparse_loss.item())

    dot = make_dot(dense_loss, params=dict(dense_criterion.named_parameters()))
    dot.view("dense_loss_graph", "data/vis/graphs")
    dot = make_dot(dense_convert_loss)
    dot.view("dense_convert_loss_graph", "data/vis/graphs")
    dot = make_dot(sparse_loss)
    dot.view("sparse_loss_graph", "data/vis/graphs")

    # sparse_test = ME.SparseTensor(
    #     coordinates=torch.tensor([[0, 0], [0, 1]]),
    #     features=torch.tensor([[0], [1.]]),
    # )

    # sparse_mul = ME.SparseTensor(
    #     coordinates=sparse_test.C,
    #     features=torch.rand([2,1]),
    #     coordinate_manager=sparse_test.coordinate_manager,
    #     device=sparse_test.device,
    # )

    # sparse_test.requires_grad_()

    # print(sparse_mul, sparse_test)

    # sparse_res = sparse_test * sparse_mul

    # dot = make_dot(sparse_res.F[0])
    # dot.view("dense_convert_active_graph")

    # dot = make_dot(sparse_res.F[1])
    # dot.view("dense_convert_void_graph")


if __name__ == "__main__":
    test()
