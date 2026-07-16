from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from src.data.epochs import infer_epoch_shape, load_epochs_from_index, ChannelStandardizer
from src.experiments.deep_loso import build_deep_model, apply_euclidean_alignment
from src.experiments.evaluate import binary_metrics, safe_roc_auc
from src.models.adaptation.adabn import adapt_batch_norm
from src.experiments.loso import iter_loso_splits
from src.data.prepare import prepare_epoch_dataset
from src.training.torch_trainer import predict_torch_model
from src.tracking.mlflow_utils import configure_mlflow, log_config, log_artifacts
import mlflow

def format_fraction_tag(fraction):
    return f"f{int(round(float(fraction) * 100)):02d}"

def build_run_name(base_name, adaptation_name, cv_suffix="cv"):
    if base_name:
        return f"{base_name}_{adaptation_name}_{cv_suffix}"
    return f"{adaptation_name}_{cv_suffix}"

def load_checkpoint(config, adaptation_name, subject_id, model):
    if adaptation_name == "ea":
        checkpoint_dir = Path(config.output_dir) / config.ea_baseline_run_name / "checkpoints"
    else:
        checkpoint_dir = Path(config.output_dir) / config.baseline_run_name / "checkpoints"
        
    ckpt_path = checkpoint_dir / f"eegnet_subject_{subject_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for subject {subject_id} at {ckpt_path}")
        
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model

def run_fixed_window_sweep(config, valid_minutes_list, methods):
    """
    Evaluates adaptation using sliding windows of a fixed absolute length in valid minutes.
    """
    configure_mlflow(config)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = prepare_epoch_dataset(config)
    epoch_index = dataset.df
    n_channels, n_samples, sfreq, channel_names = infer_epoch_shape(config.epoch_dir)
    
    loso_splits = list(iter_loso_splits(epoch_index))
    
    for adaptation_name in methods:
        run_name = build_run_name(config.run_name, adaptation_name, "fixed_window_sweep")
        output_dir = Path(config.output_dir) / run_name
        
        # Load baseline summary to get tuned decision thresholds (Zero-Leakage)
        baseline_dir = Path(config.output_dir) / (config.ea_baseline_run_name if adaptation_name == "ea" else config.baseline_run_name)
        summary_path = baseline_dir / "fold_summary.csv"
        baseline_thresholds = {}
        if summary_path.exists():
            import pandas as pd
            df = pd.read_csv(summary_path)
            if "decision_threshold" in df.columns:
                for _, row in df.iterrows():
                    baseline_thresholds[str(row["test_subject"])] = row["decision_threshold"]
                    
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with mlflow.start_run(run_name=run_name):
            log_config(config)
            
            fold_metrics_rows = []
            
            for minutes in valid_minutes_list:
                # 1 minute = 60 seconds = 7.5 epochs -> round to nearest integer
                num_adapt_blocks = int(max(1, round((minutes * 60) / 8)))
                
                print(f"Fixed CV {adaptation_name} {minutes} valid mins: ")
                for fold_index, (subject_id, train_idx, test_idx) in tqdm(
                    enumerate(loso_splits, start=1),
                    total=len(loso_splits),
                    desc=f"CV {adaptation_name.upper()} {minutes}vmin",
                    unit="fold"
                ):
                    test_subject_id = str(subject_id)
                    test_df = epoch_index.iloc[test_idx].reset_index(drop=True)
                    
                    # Fetch zero-leakage threshold optimized during training
                    decision_threshold = baseline_thresholds.get(test_subject_id, config.decision_threshold)
                    
                    # Also load train data to fit the ChannelStandardizer if needed
                    # Actually, if we use the baseline checkpoints, they were trained with channel standardization!
                    # Wait, we need to load the standardizer. The standardizer is fit on X_train.
                    # We must fit it on X_train of this fold just like during training.
                    outer_train_df = epoch_index.iloc[train_idx].reset_index(drop=True)
                    val_mask = outer_train_df["subject_id"].astype(str) == str(test_subject_id) # val subject logic simplified: actually we just need X_train which is everything except val. But val doesn't affect standardizer much, we can just fit standardizer on all train.
                    # Let's be exact:
                    from src.experiments.deep_loso import choose_validation_subject
                    subject_order = [str(s) for s, _, _ in loso_splits]
                    val_subject_id = choose_validation_subject(test_subject_id, subject_order, candidate_df=outer_train_df, strategy=config.validation_subject_strategy, seed=config.seed, min_samples_per_class=getattr(config, 'val_min_class_samples', 0))
                    val_mask = outer_train_df["subject_id"].astype(str) == str(val_subject_id)
                    train_df = outer_train_df.loc[~val_mask].reset_index(drop=True)
                    X_train, _, _ = load_epochs_from_index(train_df)
                    
                    if config.adaptation_name in {"ea", "ea_adabn"}: # wait, we use adaptation_name from the loop
                        pass # handled inside run logic
                        
                    X_test, y_test, test_metadata = load_epochs_from_index(test_df)
                    
                    # Pre-compute normalizer once per fold
                    normalizer = None
                    if config.normalization != "none":
                        if adaptation_name == "ea":
                            X_train_norm, _ = apply_euclidean_alignment(X_train, train_df)
                            normalizer = ChannelStandardizer().fit(X_train_norm)
                        else:
                            normalizer = ChannelStandardizer().fit(X_train)
                    
                    # For fixed window, we want 10 contiguous blocks of `num_adapt_blocks` length
                    n_total = len(X_test)
                    
                    # Ensure we have room for at least 1 epoch of future evaluation
                    max_start_idx = n_total - num_adapt_blocks - 1
                    if max_start_idx < 0:
                        print(f"Subject {test_subject_id} has {n_total} epochs, skipping for {minutes} min requirement.")
                        continue
                        
                    rng = random.Random(f"{config.seed}:{test_subject_id}:{minutes}")
                    # Pick 10 random starting points
                    start_indices = [rng.randint(0, max_start_idx) for _ in range(10)]
                    
                    run_metrics = []
                    
                    # 10 Runs for Statistical Averaging
                    for start_idx in start_indices:
                        adapt_indices = np.arange(start_idx, start_idx + num_adapt_blocks)
                        
                        # Evaluate on the immediate next block (True Sliding Window)
                        eval_start = start_idx + num_adapt_blocks
                        eval_end = min(n_total, eval_start + num_adapt_blocks)
                        eval_indices = np.arange(eval_start, eval_end)
                        
                        if len(eval_indices) == 0:
                            eval_indices = adapt_indices
                            
                        X_adapt = X_test[adapt_indices]
                        X_eval = X_test[eval_indices]
                        y_eval = y_test[eval_indices]
                        meta_adapt = test_metadata.iloc[adapt_indices].reset_index(drop=True)
                        meta_eval = test_metadata.iloc[eval_indices].reset_index(drop=True)
                        
                        # Load Pristine Model
                        base_model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
                        model = load_checkpoint(config, adaptation_name, test_subject_id, base_model)
                        model.to(device)
                        
                        # Data pre-processing
                        if adaptation_name == "ea":
                            X_adapt_aligned, test_aligner = apply_euclidean_alignment(X_adapt, meta_adapt)
                            X_eval_aligned = test_aligner.transform(X_eval, meta_eval["subject_id"].astype(str).to_numpy())
                            
                            X_adapt = X_adapt_aligned
                            X_eval = X_eval_aligned
                        
                        if normalizer is not None:
                            X_adapt = normalizer.transform(X_adapt)
                            X_eval = normalizer.transform(X_eval)
                            
                        # Adaptation
                        if adaptation_name == "adabn":
                            adapt_batch_norm(model, X_adapt, config)
                            
                        # Evaluation
                        test_metrics, _, y_pred, y_score = predict_torch_model(
                            model, X_eval, y_eval, config, decision_threshold=decision_threshold
                        )
                            
                        metrics = binary_metrics(y_eval, y_pred)
                        if len(np.unique(y_eval)) > 1:
                            metrics["roc_auc"] = safe_roc_auc(y_eval, y_score)
                        else:
                            metrics["roc_auc"] = np.nan
                        run_metrics.append(metrics)
                    
                    # Average over the 10 runs
                    avg_metrics = {k: np.mean([rm[k] for rm in run_metrics]) for k in run_metrics[0].keys()}
                    avg_metrics.update({
                        "fold": fold_index,
                        "test_subject_id": test_subject_id,
                        "valid_minutes": minutes,
                        "adaptation_name": adaptation_name,
                        "n_target_adapt": num_adapt_blocks,
                        "n_target_eval": len(eval_indices)
                    })
                    fold_metrics_rows.append(avg_metrics)
                    
            if fold_metrics_rows:
                fold_df = pd.DataFrame(fold_metrics_rows)
                fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
                
                # Create Summary
                summary_rows = []
                for minutes in valid_minutes_list:
                    frac_df = fold_df[np.isclose(fold_df["valid_minutes"], minutes)]
                    if not frac_df.empty:
                        sum_row = frac_df.mean(numeric_only=True).to_dict()
                        sum_row["adaptation_name"] = adaptation_name
                        sum_row["valid_minutes"] = minutes
                        sum_row["n_folds"] = len(frac_df)
                        summary_rows.append(sum_row)
                        
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(output_dir / "summary.csv", index=False)
                
                for _, row in summary_df.iterrows():
                    step = int(round(row["valid_minutes"]))
                    for key, val in row.items():
                        if isinstance(val, (int, float)) and pd.notna(val):
                            mlflow.log_metric(key, float(val), step=step)
                            
                log_artifacts([str(output_dir / "fold_metrics.csv"), str(output_dir / "summary.csv")])
                print(f"Finished strict CV sweep for {adaptation_name}")
