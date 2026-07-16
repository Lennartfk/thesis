#!/usr/bin/env python3
"""
Compute pairwise statistical tests between model results.

Usage:
    source thesis/bin/activate
    python3 scripts/stats_pairwise_models.py \
        --base data/results/experiments \
        --out results/pairwise_stats.csv

The script expects each experiment folder to contain `fold_metrics.csv` with
columns `test_subject_id` (or `fold`) and `balanced_accuracy`.

It performs paired t-test, Wilcoxon signed-rank test, computes Cohen's d for
paired samples, and applies Holm-Bonferroni correction to the t-test p-values.
Outputs a CSV and prints a summary table.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from functools import reduce


DEFAULT_MAP = {
    'EEGNet': 'Baselines/Deep/Cross-subject/EEGNet_SEED-VIG',
    'EEGNet + EA': 'Adaptation/EA_SEED-VIG',
    'EEGNet + AdaBN': 'Adaptation/Adabn_SEED-VIG',
    'EEGNet (New Hyperparameters)': 'Baselines/Deep/Cross-subject/EEGNet_New_Hyperparameters_SEED-VIG',
}


def load_model_df(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    key_col = 'test_subject_id' if 'test_subject_id' in df.columns else 'fold'
    res = df[[key_col, 'balanced_accuracy']].rename(columns={key_col: 'test_subject_id', 'balanced_accuracy': 'balanced_accuracy'})
    return res


def cohen_d_paired(x, y):
    d = x - y
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    return d.mean() / d.std(ddof=1)


def holm_bonferroni(pvals):
    # pvals: array-like
    p = np.array(pvals)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.ones_like(p)
    for k, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (m - k))
    for i in range(1, m):
        prev = adjusted[order[i-1]]
        cur_idx = order[i]
        adjusted[cur_idx] = max(prev, adjusted[cur_idx])
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='data/results/experiments', help='Base experiments folder')
    parser.add_argument('--map', nargs='*', help='Optional explicit mapping entries NAME=REL_PATH', default=None)
    parser.add_argument('--out', default='data/results/experiments/pairwise_stats.csv', help='Output CSV path (relative to repo)')
    args = parser.parse_args()

    base = Path(args.base)

    mapping = DEFAULT_MAP.copy()
    if args.map:
        for item in args.map:
            if '=' in item:
                k, v = item.split('=', 1)
                mapping[k.strip()] = v.strip()

    dfs = {}
    for name, rel in mapping.items():
        folder = base / Path(rel)
        csvp = folder / 'fold_metrics.csv'
        if not csvp.exists():
            print(f'Warning: missing {csvp} for {name} -- skipping')
            continue
        df = load_model_df(csvp)
        if df is None or df.empty:
            print(f'Warning: empty dataframe for {name} at {csvp} -- skipping')
            continue
        # rename balanced_accuracy column to model name
        df = df.rename(columns={'balanced_accuracy': name})
        dfs[name] = df

    if not dfs:
        print('No model data found. Exiting.')
        return

    # merge on test_subject_id
    merged = reduce(lambda a, b: pd.merge(a, b, on='test_subject_id', how='inner'), dfs.values())
    print(f'Merged data shape: {merged.shape}')
    if merged.shape[0] == 0:
        print('No overlapping test_subject_id between models. Exiting.')
        return

    models = list(dfs.keys())

    records = []
    pvals = []

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            a = models[i]
            b = models[j]
            x = merged[a].to_numpy(dtype=float)
            y = merged[b].to_numpy(dtype=float)
            n = int(np.sum(~np.isnan(x - y)))
            mean_a = np.nanmean(x)
            mean_b = np.nanmean(y)
            diff = x - y
            mean_diff = np.nanmean(diff)
            std_diff = np.nanstd(diff, ddof=1) if n > 1 else np.nan
            cohen_d = cohen_d_paired(x, y)

            # paired t-test
            try:
                t_stat, p_t = stats.ttest_rel(x, y, nan_policy='omit')
            except Exception:
                t_stat, p_t = np.nan, np.nan

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, p_w = stats.wilcoxon(x, y)
            except Exception:
                w_stat, p_w = np.nan, np.nan

            records.append({
                'model_a': a,
                'model_b': b,
                'n': n,
                'mean_a': mean_a,
                'mean_b': mean_b,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'cohen_d': cohen_d,
                't_stat': t_stat,
                'p_t': p_t,
                'w_stat': w_stat,
                'p_w': p_w,
            })
            pvals.append(p_t if not np.isnan(p_t) else 1.0)

    # Holm-Bonferroni correction for p_t
    adjusted = holm_bonferroni(pvals)
    for rec, adj in zip(records, adjusted):
        rec['p_t_holm'] = adj

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)

    # Print summary
    print('\nPairwise comparisons:')
    print('A vs B | n | meanA | meanB | meanDiff | stdDiff | Cohen d | t-stat | p_t | p_t_holm | w-stat | p_w')
    for r in records:
        print(f"{r['model_a']} vs {r['model_b']} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['mean_diff']:.4f} | {r['std_diff']:.4f} | {r['cohen_d']:.4f} | {r['t_stat']:.4f} | {r['p_t']:.4g} | {r['p_t_holm']:.4g} | {r['w_stat']:.4f} | {r['p_w']:.4g}")

    print(f'Written results to {out_path}')


if __name__ == '__main__':
    main()
