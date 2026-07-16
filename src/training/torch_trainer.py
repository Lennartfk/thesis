from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.experiments.evaluate import binary_metrics, binary_predictions_from_scores, safe_roc_auc


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loader(X, y, batch_size, shuffle, num_workers=0):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def class_weights_from_labels(y, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = y.size / (n_classes * safe_counts)
    return weights.astype(np.float32)


def run_epoch(model, loader, criterion, device, optimizer=None, decision_threshold=0.5):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_score = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        scores = probabilities[:, 1].cpu().numpy()
        predictions = binary_predictions_from_scores(scores, threshold=decision_threshold)

        batch_size = y_batch.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.append(y_batch.detach().cpu().numpy())
        y_pred.append(predictions)
        y_score.append(scores)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    metrics = binary_metrics(y_true, y_pred)
    metrics["roc_auc"] = safe_roc_auc(y_true, y_score)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, y_true, y_pred, y_score


def train_torch_model(model, X_train, y_train, X_val, y_val, config):
    device = resolve_device(config.device)
    print(f"Training on device: {device}")
    model = model.to(device)

    train_loader = make_loader(X_train, y_train, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_loader(X_val, y_val, config.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.use_class_weights:
        weights = torch.from_numpy(class_weights_from_labels(y_train)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_metric_value = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start = perf_counter()
    
    # Determine if metric should be minimized or maximized
    metric_name = config.early_stopping_metric
    minimize_metric = metric_name == "loss"

    for epoch in tqdm(range(1, config.max_epochs + 1), desc="Epochs", unit="epoch", leave=False):
        train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics, _, _, _ = run_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)

        # Check if metric exists in validation metrics
        if metric_name not in val_metrics:
            raise KeyError(f"Early stopping metric '{metric_name}' not found in validation metrics. Available: {list(val_metrics.keys())}")
        
        current_value = val_metrics[metric_name]
        
        # Check for improvement based on metric direction
        is_improvement = False
        if best_metric_value is None:
            is_improvement = True
        elif minimize_metric:
            is_improvement = current_value < best_metric_value
        else:  # maximize metric
            is_improvement = current_value > best_metric_value
        
        if is_improvement:
            best_metric_value = current_value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break


    if best_state is not None:
        model.load_state_dict(best_state)

    training_info = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric_value,
        "best_val_metric_name": metric_name,
        "trained_epochs": len(history),
        "training_seconds": perf_counter() - start,
        "device": str(device),
    }
    return model, pd.DataFrame(history), training_info


def predict_torch_model(model, X, y, config, decision_threshold=None):
    device = resolve_device(config.device)
    print(f"Predicting on device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X, y, config.batch_size, shuffle=False, num_workers=config.num_workers)
    if decision_threshold is None:
        decision_threshold = config.decision_threshold
    metrics, y_true, y_pred, y_score = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        decision_threshold=decision_threshold,
    )
    return metrics, y_true, y_pred, y_score
