import setup_paths  # noqa: F401
import torch
import torch.utils.benchmark as benchmark
import spconv.pytorch as spconv
import argparse
from utils.init import init_cuda, init_model
from utils.tensor import dense_to_spconv


def prepare_input(B, C, H, W, Z, sparse=False):
    # Create a random binary tensor (for example purposes)
    input_tensor = torch.randint(0, 2, (B, C, H, W, Z)).float()

    mask = (torch.rand(B, C, H, W, Z) < 0.1).float()
    # Expand mask to match (B, C, H, W, Z) and apply it.
    input_dense = input_tensor * mask
    input_dense = input_dense.cuda()
    if sparse:
        input_sparse = dense_to_spconv(input_dense)
        return input_sparse

    return input_dense


def test():
    pass

@torch.no_grad()
def timing():
    init_cuda(True, False, 0, True)

    B, C, H, W, Z = 1, 1, 256, 256, 256
    input_tensor = prepare_input(B, C, H, W, Z, sparse=True)

    vnet = init_model('VNet', backend_type='spconv',
                      in_channels=C, classes=1)
    vnet = vnet.cuda()
    vnet.eval()

    vnet_time = _benchmark(vnet, input_tensor)

    print(f"VNet inference time: {vnet_time:.2f} µs")

    vnet_interleaver = init_model(
        'VNetInterleaved', backend_type='spconv', in_channels=C, classes=1)
    vnet_interleaver = vnet_interleaver.cuda()
    vnet_interleaver.eval()

    vnet_interleaved_time = _benchmark(vnet_interleaver, input_tensor)

    print(f"VNetInterleaved inference time: {vnet_interleaved_time:.2f} µs")

    vnet_dyt = init_model('VNet', backend_type='spconv',
                          in_channels=C, classes=1, args=argparse.Namespace(use_dyt=True))
    vnet_dyt = vnet_dyt.cuda()
    vnet_dyt.eval()

    vnet_dyt_time = _benchmark(vnet_dyt, input_tensor)

    print(f"VNet inference time with DYT: {vnet_dyt_time:.2f} µs")

    # Compute speedup compared to VNet
    speedup_interleaved = vnet_time / vnet_interleaved_time

    print(f"Speedup of VNetInterleaved over VNet: {speedup_interleaved:.2f}x")


def _benchmark(model_component, input_tensor):
    timer = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model_component, "input_tensor": input_tensor},
    )

    REPEAT_COUNT = 20
    return timer.timeit(REPEAT_COUNT).mean * 1e6


if __name__ == "__main__":
    timing()
