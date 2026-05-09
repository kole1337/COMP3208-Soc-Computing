# not for submission - made by claude
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Holdout split script for COMP3208 local evaluation
# ----------------------------------------------------
# Splits train_100k_withratings.csv into:
#   local_train.csv  — rows used to train your model  (with ratings)
#   local_test.csv   — rows used to evaluate MAE      (without ratings)
#   local_test_truth.csv — same rows but WITH ratings (for eval_mae.py)
#
# Usage:
#   python make_holdout.py --input train_100k_withratings.csv --ratio 0.1
#
# Then:
#   1. Train part2.py on local_train.csv  -> produces results.csv
#   2. python eval_mae.py --pred results.csv --truth local_test_truth.csv
#
# --ratio controls the fraction of rows held out for testing (default 0.1 = 10%)

import codecs
import random
import argparse

def make_holdout(input_file, ratio=0.1, seed=42):
    random.seed(seed)

    train_out = 'local_train.csv'
    test_out  = 'local_test.csv'        # no ratings — feed this to part2.py
    truth_out = 'local_test_truth.csv'  # with ratings — feed this to eval_mae.py

    n_train = n_test = 0

    with codecs.open(input_file, 'r', 'utf-8', errors='replace') as fin, \
         codecs.open(train_out,  'w', 'utf-8') as f_train, \
         codecs.open(test_out,   'w', 'utf-8') as f_test, \
         codecs.open(truth_out,  'w', 'utf-8') as f_truth:

        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(',')
            if len(parts) < 4:
                continue

            # format: userid, itemid, rating, timestamp
            uid, iid, rating, ts = parts[0], parts[1], parts[2], parts[3]

            if random.random() < ratio:
                # held-out row: write without rating for part2.py input
                f_test.write(f'{uid},{iid},{ts}\n')
                # write with rating for eval_mae.py ground truth
                f_truth.write(f'{uid},{iid},{rating},{ts}\n')
                n_test += 1
            else:
                f_train.write(f'{uid},{iid},{rating},{ts}\n')
                n_train += 1

    print(f'Split complete (ratio={ratio}, seed={seed})')
    print(f'  Training rows : {n_train:,}  -> {train_out}')
    print(f'  Test rows     : {n_test:,}   -> {test_out}  /  {truth_out}')
    print()
    print('Next steps:')
    print(f'  1. Point part2.py train_file to "{train_out}" and test_file to "{test_out}"')
    print(f'  2. Run part2.py to produce results.csv')
    print(f'  3. python eval_mae.py --pred results.csv --truth {truth_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a local holdout split for COMP3208')
    parser.add_argument('--input', default='train_100k_withratings.csv',
                        help='Path to the training CSV with ratings')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='Fraction of rows to hold out as test (default: 0.1)')
    args = parser.parse_args()
    make_holdout(args.input, ratio=args.ratio)