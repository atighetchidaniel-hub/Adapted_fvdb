import setup_paths  # noqa: F401
import argparse

import torch
import torch.utils.benchmark as benchmark

from utils.init import init_cuda, init_model
from utils.tensor import to_sparse


def prepare_input(B, C, H, W, Z, sparse=False):
    input_tensor = torch.randint(0, 2, (B, C, H, W, Z), device="cuda").float()
    mask = (torch.rand(B, C, H, W, Z, device="cuda") < 0.1).float()
    # Use a sparse-ish binary volume so the benchmark is closer to the actual
    # NeuralPVS operating regime than a fully dense random tensor.
    input_dense = input_tensor * mask
    if sparse:
        return to_sparse(input_dense, "fvdb")
    return input_dense


def test():
    pass


@torch.no_grad()
def timing():
    init_cuda(True, False, 0, True)

    B, C, H, W, Z = 1, 1, 64, 64, 64
    input_tensor = prepare_input(B, C, H, W, Z, sparse=True)

    # This script is a small backend timing comparison between the plain fvdb
    # VNet and the interleaved variant on the same sparse input.
    vnet = init_model("VNet", backend_type="fvdb", in_channels=C, classes=1)
    vnet = vnet.cuda().eval()
    vnet_time = _benchmark(vnet, input_tensor)
    print(f"VNet inference time: {vnet_time:.2f} µs")

    vnet_interleaver = init_model(
        "VNetInterleaved", backend_type="fvdb", in_channels=C, classes=1,
        args=argparse.Namespace(interleaver_r=2)
    )
    vnet_interleaver = vnet_interleaver.cuda().eval()
    vnet_interleaved_time = _benchmark(vnet_interleaver, input_tensor)
    print(f"VNetInterleaved inference time: {vnet_interleaved_time:.2f} µs")

    speedup_interleaved = vnet_time / vnet_interleaved_time
    print(f"Speedup of VNetInterleaved over VNet: {speedup_interleaved:.2f}x")


def _benchmark(model_component, input_tensor):
    timer = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model_component, "input_tensor": input_tensor},
    )
    repeat_count = 20
    return timer.timeit(repeat_count).mean * 1e6


if __name__ == "__main__":
    timing()
