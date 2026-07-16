from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.torch_trainer import resolve_device


def _batch_norm_modules(model):
    return [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]


@contextmanager
def _preserve_module_modes(model):
    original_modes = {module: module.training for module in model.modules()}
    try:
        yield
    finally:
        for module, was_training in original_modes.items():
            module.train(was_training)


@contextmanager
def _cumulative_batch_norm_stats(batch_norm_layers):
    original_momenta = {module: module.momentum for module in batch_norm_layers}
    try:
        for module in batch_norm_layers:
            module.reset_running_stats()
            module.momentum = None
        yield
    finally:
        for module, momentum in original_momenta.items():
            module.momentum = momentum


def adapt_batch_norm(model, X_target_unlabeled, config):
    """Update BatchNorm running statistics from unlabeled target-subject epochs."""
    batch_norm_layers = _batch_norm_modules(model)
    if not batch_norm_layers:
        return {
            "adabn_applied": False,
            "adabn_batch_norm_layers": 0,
            "adabn_target_samples": int(len(X_target_unlabeled)),
            "adabn_target_batches": 0,
        }

    device = resolve_device(config.device)
    model = model.to(device)

    dataset = TensorDataset(torch.from_numpy(X_target_unlabeled).float())
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    with _preserve_module_modes(model), _cumulative_batch_norm_stats(batch_norm_layers):
        model.eval()
        for module in batch_norm_layers:
            module.train()

        batch_count = 0
        with torch.no_grad():
            for (X_batch,) in loader:
                _ = model(X_batch.to(device))
                batch_count += 1

    model.eval()
    return {
        "adabn_applied": True,
        "adabn_batch_norm_layers": len(batch_norm_layers),
        "adabn_target_samples": int(len(X_target_unlabeled)),
        "adabn_target_batches": batch_count,
    }
