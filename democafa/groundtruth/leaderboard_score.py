#!/usr/bin/env python3
"""Aggregate evaluator scores into the CAFA leaderboard score.

The script expects evaluator outputs under ``evaluation_w_NK``,
``evaluation_w_LK``, and ``evaluation_w_PK``. For each prediction file, it
averages ``f_micro_w`` within each subset and then averages the three subset
scores into ``final_score``.
"""

import os
import pandas as pd
import argparse


def calculate_average_aspect(data_dir):
    """Return per-prediction mean weighted F-measure for one evaluation subset."""
    result_file = os.path.join(data_dir, "evaluation_best_f_micro_w.tsv")
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return {}
    df = pd.read_csv(result_file, sep='\t',header=0)
    filenames = dict()
    if 'f_micro_w' in df.columns:
        for filename in set(df['filename']):
            score = df[df['filename'] == filename]['f_micro_w'].mean()
            print(f"File: {filename}, f_micro_w: {score}")
            filenames[filename] = score
    return filenames

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calculate leaderboard score from prediction and ground truth files.")
    parser.add_argument("--DATA_DIR", "--data-dir", dest="DATA_DIR", type=str, required=True, help="Directory containing evaluation_w_NK/LK/PK folders.")
    parser.add_argument("--output_file", type=str, default="leaderboard_score.tsv", help="Output file for leaderboard score.")
    return parser.parse_args(argv)

def main():
    args = parse_args()
    NK_dir = os.path.join(args.DATA_DIR, "test_cafaeval_NK")
    LK_dir = os.path.join(args.DATA_DIR, "test_cafaeval_LK")
    PK_dir = os.path.join(args.DATA_DIR, "test_cafaeval_PK")

    score_dict_NK = calculate_average_aspect(NK_dir)
    score_dict_LK = calculate_average_aspect(LK_dir)
    score_dict_PK = calculate_average_aspect(PK_dir)

    all_methods = set(score_dict_NK.keys()).union(set(score_dict_LK.keys())).union(set(score_dict_PK.keys()))
    for filename in all_methods:
        if filename in score_dict_NK and filename in score_dict_LK and filename in score_dict_PK:
            avg_score = (score_dict_NK[filename] + score_dict_LK[filename] + score_dict_PK[filename]) / 3
            print(f"File: {filename}, Average Score: {avg_score}")
        else:
            print(f"File: {filename} is missing in one of the evaluations.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as out_f:
        out_f.write("filename\tf_micro_w_NK\tf_micro_w_LK\tf_micro_w_PK\tfinal_score\n")
        for filename in sorted(all_methods):
            f_micro_w_NK = score_dict_NK.get(filename, 0.0)
            f_micro_w_LK = score_dict_LK.get(filename, 0.0)
            f_micro_w_PK = score_dict_PK.get(filename, 0.0)
            f_micro_w_final = (f_micro_w_NK + f_micro_w_LK + f_micro_w_PK) / 3
            out_f.write(f"{filename}\t{f_micro_w_NK}\t{f_micro_w_LK}\t{f_micro_w_PK}\t{f_micro_w_final}\n")

if __name__ == "__main__":
    main()
