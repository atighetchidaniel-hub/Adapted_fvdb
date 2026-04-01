import traceback
import os
import time

import numpy as np
import torch
import torch.utils.benchmark as benchmark

try:
    import spconv.pytorch as spconv
except Exception:
    spconv = None

from modules.writer import TrainingLogger

from utils.args import save_arguments
from utils.train import prepare_data, save_checkpoint, save_volume, extract_data
from utils.init import init_metric
from utils.tensor import requires_grad, to_dense, to_sparse


class Runner:
    """
    Trainer class
    """

    def __init__(self, args, model: torch.nn.Module,
                 criterion: torch.nn.Module | list[torch.nn.Module] = None,
                 optimizer: torch.optim.Optimizer = None,
                 train_data_loader: torch.utils.data.DataLoader = None,
                 valid_data_loader: torch.utils.data.DataLoader = None,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 start_epoch: int = 1,
                 mode: str = 'train'
                 ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_weights = args.loss_weights
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.writer = TrainingLogger(args.save)

        self.device, device_ids, self.n_gpu = self._prepare_device()
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.save_freq = 10
        self.save_all_freq = args.save_all_freq if args.save_all_freq else 50
        self.log_freq = self.args.terminal_show_freq
        self.start_epoch = start_epoch
        self.scaler = torch.amp.GradScaler(self.device)
        self.metrics = init_metric()
        if args.cuda:
            self.metrics = self.metrics.cuda()
        self.mode = mode
        self.start_time = time.time()

        self.forward_times = {
            "total": 0,
            "count": 0
        }

        self.val_loss = np.inf

        if mode == 'train':
            save_arguments(args, args.save)

    def train(self):
        self.mode = 'train'
        for epoch in range(self.start_epoch, self.args.nEpochs+1):
            cur_time = time.time() - self.start_time
            print(f'time elapsed: {cur_time}')
            if (cur_time > 60*60*120):
                print('Stopping training due to time limit about to reached')
                return
            self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.validate_epoch(epoch)

            if self.args.save is not None:
                if (self.n_gpu > 1):
                    model_save = self.model.module
                else:
                    model_save = self.model
                if (epoch + 1) % self.save_freq == 0:
                    save_checkpoint(model_save, self.args.save,
                                    epoch, self.val_loss,
                                    optimizer=self.optimizer,
                                    scheduler=self.lr_scheduler)
                if self.save_all_freq > 0 and (epoch + 1) % self.save_all_freq == 0:
                    save_name = "{}_{}_epoch.pth".format(
                        os.path.basename(self.args.save),
                        epoch)
                    save_checkpoint(model_save, self.args.save,
                                    epoch, self.val_loss,
                                    optimizer=self.optimizer,
                                    scheduler=self.lr_scheduler,
                                    name=save_name)
                

    @torch.no_grad()
    def infer(self):
        self.mode = 'infer'
        if not self.valid_data_loader:
            return

        self.model.eval()

        metrics_all = {}
        i = 0

        pred_caches = []

        n_caches = self.args.cache_size

        for batch_idx, data_dict in enumerate(self.valid_data_loader):
            if self.args.n_frames is not None and i >= self.args.n_frames:
                break
            i += 1
            
            data_dict = prepare_data(data_dict)
            input = extract_data(data_dict, 'input')
            target = extract_data(data_dict, 'target')

            if not torch.any(input > 0):
                print("Empty input, skipping batch")
                continue

            input = to_sparse(input, self.args.backend)

            requires_grad(input, False)

            try:
                if self.args.amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        pred = self.model(input, data_dict)
                        loss, metrics = self._compute_criterion(
                            pred, target, data_dict)
                        metrics.update(self.metrics(pred, target, data_dict))

                else:
                    pred = self.model(input, data_dict)
                    pred_dense = to_dense(pred, target.shape)

                    if n_caches > 0:
                        # Cache the prediction
                        if len(pred_caches) < n_caches:
                            pred_caches.append(pred_dense)
                        else:
                            # pop the oldest cache
                            pred_caches.pop(0)
                            pred_caches.append(pred_dense)
                    
                    # If we have caches, do logical OR on them
                    if len(pred_caches) > 0:
                        pred_dense = torch.stack(pred_caches, dim=0).max(dim=0)[0]

                    # Apply max pooling dilation if enabled
                    max_pool_size = self.args.max_pool_size
                    if max_pool_size == 0:  # Use default size
                        max_pool_size = 11
                    
                    if max_pool_size > 0:
                        pred_pool = max_pool_dilate3d(pred_dense, max_pool_size)
                        pred = pred_pool
                    else:
                        pass
                        
                    loss, metrics = self._compute_criterion(
                        pred, target, data_dict)
                    metrics.update(self.metrics(pred, target, data_dict))
            except Exception as e:
                # With too less input data, convolution will fail
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                raise e
                # continue

            # for each attr in metrics, add to metrics_all
            for key in metrics:
                if key not in metrics_all:
                    metrics_all[key] = []
                metrics_all[key].append(metrics[key])

            self.writer.log_metrics(
                metrics, batch_idx, mode='infer', log_to_stdout=True
            )

            output = pred
            output = to_dense(output, target.shape)
            self._save_output(output, 0, batch_idx)

        _metrics = {}
        for key in metrics_all:
            # add a `-hist` suffix to the key
            key_hist = key + '-hist'
            _metrics[key_hist] = metrics_all[key]
        self.writer.log_metrics(_metrics, 0, mode='infer', log_to_stdout=False, log_to_csv=False)

        self.writer.compute_stats()

    @torch.no_grad()
    def timing(self):
        self.mode = 'infer'
        if not self.valid_data_loader:
            return

        self.model.eval()

        density_data = []

        computed = False

        i = 0

        for batch_idx, data_dict in enumerate(self.valid_data_loader):
            if self.args.n_frames is not None and i >= self.args.n_frames:
                break
            i += 1

            input_tensor = data_dict['input'].to(self.device)
            density_data.append(input_tensor.sum().cpu().item() / input_tensor.numel())

            if computed:
                # Skip the first batch to avoid warmup time
                continue
            computed = True
            # input_tensor = to_sparse(input_tensor, self.args.backend)
            requires_grad(input_tensor, False)
            metrics = {}

            # Benchmark inference time
            metrics.update(self._benchmark_time(
                self.model, input_tensor))

            # Benchmark interleaver/deinterleaver/shapecriptor and exclude from total time
            if hasattr(self.model, 'interleaver'):
                excluded_time = 0.

                interleave_time = self._benchmark_time(
                    self.model.interleaver, input_tensor)
                interleaved = self.model.interleaver(input_tensor)
                metrics['interleaver_time_mean'] = interleave_time["infer_time_mean"]
                excluded_time += interleave_time["infer_time_mean"]

                if hasattr(self.model, 'shapecriptor'):
                    shapecriptor_time = self._benchmark_time(
                        self.model.shapecriptor, interleaved)
                    metrics['shapecriptor_time_mean'] = shapecriptor_time["infer_time_mean"]
                    excluded_time += shapecriptor_time["infer_time_mean"]

                if hasattr(self.model, 'deinterleaver'):
                    deinterleave_time = self._benchmark_time(
                        self.model.deinterleaver, interleaved)
                    metrics['deinterleaver_time_mean'] = deinterleave_time["infer_time_mean"]
                    excluded_time += deinterleave_time["infer_time_mean"]

                metrics.update({
                    'infer_time_pure_mean': metrics['infer_time_mean'] - excluded_time
                })

            # Benchmark memory usage
            metrics.update(self._benchmark_memory(
                self.model, input_tensor))
            
            # Get number of parameters
            metrics.update(self._get_num_params())
        
        density_data = np.array(density_data)
        metrics.update({
            "density_mean": np.mean(density_data),
            "density_std": np.std(density_data),
            "density_min": np.min(density_data),
            "density_max": np.max(density_data),
        })

        self.writer.log_metrics(
            metrics, batch_idx, mode='infer', log_to_stdout=True)
        return

    def _benchmark_time(self, model_component, input_tensor):
        """
        Benchmark the model component in ms
        """
        # model_component =model_component.eval().half()
        # input_tensor = input_tensor.to(torch.float16)
        timer = benchmark.Timer(
            stmt="model(input_tensor)",
            globals={"model": model_component, "input_tensor": input_tensor},
        )

        # results = timer.timeit(number=REPEAT_COUNT)

        m = timer.blocked_autorange(min_run_time=5.0)
        times = m.times

        return {
            "infer_time_mean": np.mean(times) * 1000,
            "infer_time_std": np.std(times) * 1000,
            "infer_time_min": np.min(times) * 1000,
            "infer_time_max": np.max(times) * 1000,
        }
    
    def _benchmark_memory(self, model_component, input_tensor):
        """
        Run a forward pass through `model_component` with `input_tensor`
        and return GPU memory stats in MB.

        Returns a dict with:
            - start_mem: memory before forward
            - end_mem: memory after forward
            - peak_mem: max memory during forward
        """
        device = 'cuda:0' if self.args.cuda else 'cpu'

        # Make sure stats are clean
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Record memory before
        start_mem = torch.cuda.memory_allocated(device)

        # Forward pass (no grad to minimize extra usage)
        model_component.eval()
        with torch.no_grad():
            _ = model_component(input_tensor)
        torch.cuda.synchronize(device)

        # Record memory after and peak
        end_mem = torch.cuda.memory_allocated(device)
        peak_mem = torch.cuda.max_memory_allocated(device)

        # Convert bytes to megabytes
        to_mb = lambda bytes: bytes / (1024 ** 2)
        return {
            "start_mem": to_mb(start_mem),
            "end_mem":   to_mb(end_mem),
            "peak_mem":  to_mb(peak_mem),
        }
    
    def _get_num_params(self):
        """
        Get the number of parameters in the model
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "num_params": num_params,
        }

    def train_epoch(self, epoch):
        if not self.train_data_loader:
            return

        self.model.train()

        for batch_idx, data_dict in enumerate(self.train_data_loader):

            start_time = time.time()

            self.optimizer.zero_grad()

            data_dict = prepare_data(data_dict)
            input = extract_data(data_dict, 'input')
            target = extract_data(data_dict, 'target')

            if not torch.any(input > 0):
                print("Empty input, skipping batch")
                continue

            input = to_sparse(input, self.args.backend)

            requires_grad(input, True)

            try:
                if self.args.amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        pred = self.model(input, data_dict)
                        loss, metrics = self._compute_criterion(
                            pred, target, data_dict)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(input, data_dict)
                    loss, metrics = self._compute_criterion(
                        pred, target, data_dict)

                    loss.backward()
                    self.optimizer.step()
            except Exception as e:
                # With too less input data, convolution will fail
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

            metrics.update({'epoch': epoch})

            if self.lr_scheduler is not None:
                last_lr = self.lr_scheduler.get_last_lr()[-1]
                metrics.update({'lr': last_lr})
            self.writer.log_metrics(
                metrics, (epoch - 1) * len(self.train_data_loader) + batch_idx, mode='train', log_to_stdout=True)

            self.forward_times["total"] += time.time() - start_time
            self.forward_times["count"] += 1
            print(
                f"Average batch time: {self.forward_times['total'] / self.forward_times['count']}")

            # Required by Minkowski Engine
            torch.cuda.empty_cache()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        if not self.valid_data_loader:
            return

        self.model.eval()

        for batch_idx, data_dict in enumerate(self.valid_data_loader):
            data_dict = prepare_data(data_dict)
            input = extract_data(data_dict, 'input')
            target = extract_data(data_dict, 'target')

            if not torch.any(input > 0):
                print("Empty input, skipping batch")
                continue

            input = to_sparse(input, self.args.backend)

            requires_grad(input, False)

            try:
                if self.args.amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        pred = self.model(input, data_dict)
                        loss, metrics = self._compute_criterion(
                            pred, target, data_dict)
                        metrics.update(self.metrics(pred, target, data_dict))

                else:
                    pred = self.model(input, data_dict)
                    loss, metrics = self._compute_criterion(
                        pred, target, data_dict)
                    metrics.update(self.metrics(pred, target, data_dict))
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

            metrics.update({'epoch': epoch})

            self.writer.log_metrics(
                metrics, (epoch - 1) * len(self.valid_data_loader) + batch_idx, mode='eval', log_to_stdout=True)

            self.val_loss = loss.detach().item()

            output = pred
            output = to_dense(output, target.shape)
            self._save_output(output, epoch, batch_idx)

    def _compute_criterion(self, pred: torch.Tensor, target: torch.Tensor, extras: dict) -> tuple:
        if isinstance(self.criterion, list):
            loss = 0
            metrics = {}
            for i, crit in enumerate(self.criterion):
                _loss, _metrics = self._call_criterion(
                    crit, pred, target, extras, record_loss_name=True)
                loss += _loss * \
                    self.criterion_weights[i] if self.criterion_weights else _loss
                metrics.update(_metrics)
            metrics['loss'] = loss.detach().item()
        else:
            loss, metrics = self._call_criterion(
                self.criterion, pred, target, extras)
        return loss, metrics

    def _call_criterion(self, crit, pred, target, extras, record_loss_name=False):
        metrics = {}
        res = crit(pred, target, extras)
        if isinstance(res, tuple):
            loss, metrics = res
            if not isinstance(metrics, dict):
                metrics = {}
        else:
            loss = res
            metrics = {}
        metrics['loss_{}'.format(crit.__class__.__name__)
                if record_loss_name else 'loss'] = loss.detach().item()
        return loss, metrics

    @torch.no_grad()
    def _save_output(self, output, epoch: int, batch_idx: int):
        output_save = output.sigmoid()
        output_save = (output_save > 0.5).int()
        save_volume(output_save.cpu(), os.path.join(
            self.args.save, "inference", str(epoch), "{}_predicted_pvv.bin.gz".format(batch_idx)))

    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = 0
        if self.args.cuda:
            n_gpu = torch.cuda.device_count()
        print(f'Using {n_gpu} GPUs')
        if n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                  "training will be performed on CPU.")
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        list_ids = list()
        for i in range(0, n_gpu):
            list_ids.append(f'cuda:{i}')

        return device, list_ids, n_gpu

def max_pool_dilate3d(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Perform a 3D dilation on a binary tensor using max-pooling.

    Args:
        x: Input tensor of shape [B, 1, H, W, Z] with binary values (0 or 1).
        n: Size of the cubic structuring element (n x n x n).

    Returns:
        Tensor of the same shape as `x`, where each voxel is set to 1 if any voxel
        in its n x n x n neighborhood in the input is 1.
    """
    if x.dim() != 5 or x.size(1) != 1:
        raise ValueError(f"Expected input of shape [B, 1, H, W, Z], got {tuple(x.shape)}")

    # Compute padding for each side: floor(n/2)
    pad = n // 2
    # max_pool3d expects input in [B, C, D, H, W], which matches [B, 1, H, W, Z]
    # Use stride=1 to slide the window by one voxel and padding to preserve shape
    pooled = torch.nn.functional.max_pool3d(x, kernel_size=n, stride=1, padding=pad)

    # Convert pooled result to binary (0 or 1) matching input dtype
    return (pooled > 0).to(x.dtype)
