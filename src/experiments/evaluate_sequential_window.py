from pathlib import Path

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

def run_sequential_window_sweep(config, valid_minutes_list, methods):
    """
    Evaluates adaptation using a sequential window approach (stride equals window length).
    The model calibrates on window N, evaluates on immediately following window N, and aggregates
    predictions across the entire drive for final metric calculation.
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
        run_name = build_run_name(config.run_name, adaptation_name, "sequential_window_sweep")
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
            prediction_rows = []
            
            for minutes in valid_minutes_list:
                # 1 minute = 60 seconds = 7.5 epochs -> round to nearest integer
                num_adapt_blocks = int(max(1, round((minutes * 60) / 8)))
                
                print(f"Sequential CV {adaptation_name} {minutes} valid mins: ")
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
                    
                    # Fit ChannelStandardizer on training data for this fold
                    outer_train_df = epoch_index.iloc[train_idx].reset_index(drop=True)
                    from src.experiments.deep_loso import choose_validation_subject
                    subject_order = [str(s) for s, _, _ in loso_splits]
                    val_subject_id = choose_validation_subject(
                        test_subject_id, 
                        subject_order, 
                        candidate_df=outer_train_df, 
                        strategy=config.validation_subject_strategy, 
                        seed=config.seed, 
                        min_samples_per_class=getattr(config, 'val_min_class_samples', 0)
                    )
                    val_mask = outer_train_df["subject_id"].astype(str) == str(val_subject_id)
                    train_df = outer_train_df.loc[~val_mask].reset_index(drop=True)
                    X_train, _, _ = load_epochs_from_index(train_df)
                        
                    X_test, y_test, test_metadata = load_epochs_from_index(test_df)
                    
                    # Pre-compute normalizer once per fold
                    normalizer = None
                    if config.normalization != "none":
                        if adaptation_name == "ea":
                            X_train_norm, _ = apply_euclidean_alignment(X_train, train_df)
                            normalizer = ChannelStandardizer().fit(X_train_norm)
                        else:
                            normalizer = ChannelStandardizer().fit(X_train)
                    
                    n_total = len(X_test)
                    
                    # Sequential window iterates strictly chronologically.
                    # Stride equals the window length (num_adapt_blocks).
                    # Loop terminates if there is no data left for a calibration block.
                    start_indices = list(range(0, n_total - num_adapt_blocks, num_adapt_blocks))
                    
                    if not start_indices:
                        print(f"Subject {test_subject_id} has {n_total} epochs, skipping for {minutes} min requirement.")
                        continue
                    
                    all_y_true = []
                    all_y_pred = []
                    all_y_score = []
                    
                    for start_idx in start_indices:
                        adapt_indices = np.arange(start_idx, start_idx + num_adapt_blocks)
                        
                        # Evaluate on the immediate contiguous block (which is the trailing partial block if at end)
                        eval_start = start_idx + num_adapt_blocks
                        eval_end = min(n_total, eval_start + num_adapt_blocks)
                        eval_indices = np.arange(eval_start, eval_end)
                        
                        if len(eval_indices) == 0:
                            break
                            
                        X_adapt = X_test[adapt_indices]
                        X_eval = X_test[eval_indices]
                        y_eval = y_test[eval_indices]
                        meta_adapt = test_metadata.iloc[adapt_indices].reset_index(drop=True)
                        meta_eval = test_metadata.iloc[eval_indices].reset_index(drop=True)
                        
                        # Load Pristine Model before EVERY block to prevent drift
                        base_model = build_deep_model(config, n_channels=n_channels, n_samples=n_samples)
                        model = load_checkpoint(config, adaptation_name, test_subject_id, base_model)
                        model.to(device)
                        
                        # Data pre-processing
                        if adaptation_name == "ea":
                            # EA is recomputed entirely on current calibration block
                            X_adapt_aligned, test_aligner = apply_euclidean_alignment(X_adapt, meta_adapt)
                            X_eval_aligned = test_aligner.transform(X_eval, meta_eval["subject_id"].astype(str).to_numpy())
                            
                            X_adapt = X_adapt_aligned
                            X_eval = X_eval_aligned
                        
                        if normalizer is not None:
                            X_adapt = normalizer.transform(X_adapt)
                            X_eval = normalizer.transform(X_eval)
                            
                        # Adaptation
                        if adaptation_name == "adabn":
                            # AdaBN is recomputed entirely on current calibration block
                            adapt_batch_norm(model, X_adapt, config)
                            
                        # Evaluation
                        test_metrics, _, y_pred, y_score = predict_torch_model(
                            model, X_eval, y_eval, config, decision_threshold=decision_threshold
                        )
                        
                        all_y_true.append(y_eval)
                        all_y_pred.append(y_pred)
                        all_y_score.append(y_score)
                        
                        for i in range(len(y_eval)):
                            prediction_rows.append({
                                "test_subject_id": test_subject_id,
                                "sequential_minutes": minutes,
                                "epoch_index": meta_eval.iloc[i]["epoch_index"],
                                "y_true": int(y_eval[i]),
                                "y_pred": int(y_pred[i]),
                                "y_score": float(y_score[i]) if y_score is not None else np.nan,
                                "adaptation_name": adaptation_name
                            })
                    
                    if len(all_y_true) > 0:
                        y_true_subj = np.concatenate(all_y_true)
                        y_pred_subj = np.concatenate(all_y_pred)
                        y_score_subj = np.concatenate(all_y_score)
                        
                        metrics = binary_metrics(y_true_subj, y_pred_subj)
                        if len(np.unique(y_true_subj)) > 1:
                            metrics["roc_auc"] = safe_roc_auc(y_true_subj, y_score_subj)
                        else:
                            metrics["roc_auc"] = np.nan
                            
                        metrics.update({
                            "fold": fold_index,
                            "test_subject_id": test_subject_id,
                            "valid_minutes": minutes,
                            "adaptation_name": adaptation_name,
                            "n_target_adapt": num_adapt_blocks,
                            "n_target_eval": len(y_true_subj)
                        })
                        fold_metrics_rows.append(metrics)
                    
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
                            
                if prediction_rows:
                    pred_df = pd.DataFrame(prediction_rows)
                    pred_df.to_csv(output_dir / "predictions.csv", index=False)
                    log_artifacts([str(output_dir / "fold_metrics.csv"), str(output_dir / "summary.csv"), str(output_dir / "predictions.csv")])
                else:
                    log_artifacts([str(output_dir / "fold_metrics.csv"), str(output_dir / "summary.csv")])
                    
                print(f"Finished sequential CV sweep for {adaptation_name}")
