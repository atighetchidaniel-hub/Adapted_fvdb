import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

try:
    import spconv
except Exception:
    spconv = None

from losses.dice import SparseWeightedDiceLoss, WeightedDiceLoss
from losses.sum import WeightedSumLoss
from losses.focal import FocalLoss
from losses.metrics import Metrics
from losses.no_guess import NoGuessLoss
from losses.wrapped_loss import get_loss_module
from utils.train import count_params
from models.vnet import VNet, VNetInterleaved, VNetLighter, VNetLight
from modules.dataset import PVSVoxelDataset


def init_cuda(cuda: bool, cupy: bool, seed: int, inference: bool = False):
    if not inference:
        # In case CUDA configuration error, see https://github.com/pytorch/pytorch/issues/48573#issuecomment-1970868120
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # cusolver throws an error when using torch.linalg.eigh in shape descriptor code
        torch.backends.cuda.preferred_linalg_library("magma")
    if cuda:
        torch.cuda.manual_seed(seed)
    if cupy:
        torch.multiprocessing.set_start_method('spawn')

    if spconv is not None:
        spconv.constants.SPCONV_ALLOW_TF32 = True
        spconv.constants.SPCONV_CPP_GEMM = True


def init_model(model_type: str = 'VNet', backend_type: str = 'torchnn',
               in_channels: int = 1, classes: int = 1, model_depth: int = 3, args=None):
    if model_type == 'VNet':
        model = VNet(elu=False, in_channels=in_channels, classes=classes,
                     backend_type=backend_type)
    elif model_type == 'VNetLighter':
        model = VNetLighter(elu=False,  in_channels=in_channels,
                            classes=classes, backend_type=backend_type)
    elif model_type == 'VNetLight':
        model = VNetLight(elu=False, in_channels=in_channels,
                          classes=classes, backend_type=backend_type)
    elif model_type == 'VNetInterleaved':
        model = VNetInterleaved(elu=False, in_channels=in_channels,
                                classes=classes, backend_type=backend_type, r=args.interleaver_r if args else 2)
    elif model_type == 'OACNNs':
        if spconv is None:
            raise ImportError("OACNNs requires spconv, but spconv is not installed.")
        from models.oacnn import OACNNs
        model = OACNNs(in_channels=in_channels, classes=classes,
                       backend_type=backend_type, depth=model_depth)
    elif model_type == 'OACNNsInterleaved':
        if spconv is None:
            raise ImportError("OACNNsInterleaved requires spconv, but spconv is not installed.")
        from models.oacnn import OACNNsInterleaved
        model = OACNNsInterleaved(
            depth=model_depth, r=args.interleaver_r if args else 2)
    else:
        raise ValueError('Model not supported')

    # Dynamic tanh conversion removed
    # if args and args.use_dyt:
    #     model = convert_bn_to_dyt(model)

    print("Num of params:{}".format(count_params(model)))
    return model


def init_loss(loss_type: str | list[str] = 'dice', backend_type: str = 'torchnn',
              classes: int = 1, args=None):
    if isinstance(loss_type, list):
        if len(loss_type) == 1:
            loss_type = loss_type[0]
        else:
            return [init_loss(l, backend_type, classes, args) for l in loss_type]
    if loss_type == 'dice':
        alpha = args.dice_alpha if args else 0.1
        if backend_type == 'spconv':
            criterion = SparseWeightedDiceLoss(classes=classes, alpha=alpha)
        else:  # torchnn backend
            criterion = WeightedDiceLoss(classes=classes, alpha=alpha)
    elif loss_type == 'focal':
        criterion = FocalLoss()
    elif loss_type == 'sum':
        criterion = WeightedSumLoss(alpha=args.dice_alpha if args else 0.1)
    elif loss_type == 'bce':
        criterion = get_loss_module(torch.nn.BCELoss)()
    elif loss_type == 'bce_logits':
        criterion = get_loss_module(torch.nn.BCEWithLogitsLoss)()
    elif loss_type == 'cross_entropy':
        criterion = get_loss_module(torch.nn.CrossEntropyLoss)()
    elif loss_type == 'l1':
        criterion = get_loss_module(torch.nn.L1Loss)()
    elif loss_type == 'mse':
        criterion = get_loss_module(torch.nn.MSELoss)()
    elif loss_type == 'huber':
        criterion = get_loss_module(torch.nn.HuberLoss)()
    elif loss_type == 'nll':
        criterion = get_loss_module(torch.nn.NLLLoss)()
    elif loss_type == 'no_guess':
        criterion = NoGuessLoss()
    return criterion


def init_optimizer(optmizer_type: str = 'adam', model: nn.Module = None, lr: float = 1e-2, weight_decay: float = 0.0000000001, no_scheduler: bool = False):
    if optmizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.5, weight_decay=weight_decay)
    elif optmizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    elif optmizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'adamax':
        optimizer = optim.Adamax(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'asgd':
        optimizer = optim.ASGD(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optmizer_type == 'lbfgs':
        optimizer = optim.LBFGS(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not supported')

    scheduler = None
    if not no_scheduler:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 0.95)

    return optimizer, scheduler


def init_metric():
    return Metrics()


def init_dataloader_args(cuda: bool, cupy: bool):
    args = {}
    if cuda:
        args = {'num_workers': 1, 'pin_memory': True}
    if cupy:
        args = {'num_workers': 1}
    return args


def init_train_dataloader(dataset_path, batch_size, seed=0, cuda=True, cupy=False, amp=False, test_fraction=0.05, z_size=256):
    dataloader_args = init_dataloader_args(cuda, cupy)
    train_set = PVSVoxelDataset(root=dataset_path, mode='train', seed=seed,
                                cupy=cupy, amp=amp, test_fraction=test_fraction, z_size=z_size)
    valid_set = PVSVoxelDataset(root=dataset_path, mode='test', seed=seed,
                                    cupy=cupy, amp=False, test_fraction=test_fraction, z_size=z_size)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **dataloader_args)
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False, **dataloader_args)
    return train_loader, valid_loader


def init_infer_dataloader(dataset_path, seed=0, cuda=True, cupy=False, amp=False, z_size=256):
    dataloader_args = init_dataloader_args(cuda, cupy)
    infer_set = PVSVoxelDataset(root=dataset_path, mode='infer', seed=seed,
                                cupy=cupy, amp=amp, z_size=z_size)
    infer_loader = DataLoader(
        infer_set, batch_size=1, shuffle=False, **dataloader_args)
    return infer_loader
