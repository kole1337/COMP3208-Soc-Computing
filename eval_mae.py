# not for submission - made by claude
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Local MAE evaluation script for COMP3208
# -----------------------------------------
# The handin system evaluates your results.csv against a hidden test set.
# Since you don't have that, this script gives you two ways to estimate
# your MAE locally before submitting:
#
#   MODE 1 (recommended) — holdout evaluation
#     Split the training file into a local train/test split BEFORE running
#     part2.py, train on the local train split, predict on the local test
#     split, then run this script.  MAE will be representative of the real
#     submission score.
#
#   MODE 2 — training-set self-check
#     Run part2.py on the full training file, then compare results.csv
#     against the training file itself.  MAE will be optimistic (the model
#     has seen these ratings) but confirms predictions are in the right range
#     and no rows are missing.
#
# Usage:
#   python eval_mae.py --pred results.csv --truth train_100k_withratings.csv
#   python eval_mae.py --pred results.csv --truth my_local_test.csv
#
# results.csv format  : userid, itemid, predicted_rating, timestamp
# ground-truth format : userid, itemid, rating, timestamp

import codecs
import sys
import argparse

def load_predictions(filepath):
    """
    Load predicted ratings from results.csv.

    Expected format: userid, itemid, predicted_rating, timestamp
    Returns dict: (uid_str, iid_str) -> predicted_rating (float)
    """
    preds = {}
    with codecs.open(filepath, 'r', 'utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                preds[(parts[0], parts[1])] = float(parts[2])
            except ValueError:
                continue   # skip header if present
    return preds


def load_ground_truth(filepath):
    """
    Load ground-truth ratings from a training/test CSV with ratings.

    Expected format: userid, itemid, rating, timestamp
    Returns dict: (uid_str, iid_str) -> true_rating (float)
    """
    truth = {}
    with codecs.open(filepath, 'r', 'utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                truth[(parts[0], parts[1])] = float(parts[2])
            except ValueError:
                continue
    return truth


def compute_mae(preds, truth):
    """
    Compute Mean Absolute Error between predictions and ground truth.

    MAE = (1/N) * sum(|r_true - r_pred|)

    Missing predictions are treated as 0.0 (matching the handin system
    behaviour), which heavily penalises incomplete results.csv files.
    """
    total_err = 0.0
    missing   = 0

    for key, true_rating in truth.items():
        if key in preds:
            total_err += abs(true_rating - preds[key])
        else:
            # handin system defaults missing predictions to 0
            total_err += abs(true_rating - 0.0)
            missing   += 1

    n   = len(truth)
    mae = total_err / n if n > 0 else float('inf')
    return mae, n, missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute MAE for COMP3208 submission')
    parser.add_argument('--pred',  required=True, help='Path to your results.csv')
    parser.add_argument('--truth', required=True, help='Path to ground-truth CSV (with ratings)')
    args = parser.parse_args()

    print(f'Loading predictions from : {args.pred}')
    preds = load_predictions(args.pred)
    print(f'  {len(preds):,} predictions loaded')

    print(f'Loading ground truth from: {args.truth}')
    truth = load_ground_truth(args.truth)
    print(f'  {len(truth):,} ground-truth ratings loaded')

    mae, n, missing = compute_mae(preds, truth)

    print()
    print('=' * 40)
    print(f'  Total ratings evaluated : {n:,}')
    print(f'  Missing predictions     : {missing:,}  (scored as 0.0)')
    print(f'  MAE                     : {mae:.6f}')
    print('=' * 40)

    if missing > 0:
        print(f'\nWARNING: {missing:,} rows have no prediction.')
        print('  The handin system will default these to 0.0, which inflates MAE.')
        print('  Make sure every row in the test file has a corresponding prediction.')