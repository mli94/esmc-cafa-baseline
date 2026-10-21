#!/bin/bash
# Re-evaluate PK with CAFA-evaluator-PK (-known), replacing the earlier workaround.
# The workaround removed t0-known (protein, term) pairs from predictions before cafaeval,
# but cafaeval propagates both ground truth and predictions to ancestors, which re-adds
# known terms on both sides and inflates PK. -known excludes them after propagation.
# Old outputs are kept as test_cafaeval_PK_workaround/ and leaderboard_score_workaround.tsv.
# Usage (from rotation2/): bash runs/human/redo_pk.sh
set -e

GT=data/human/groundtruth/groundtruth
OBO=data/human/raw/go-basic_t0.obo
IA=data/human/processed/IA.txt
CAFA_PK=/Users/a1/miniconda3/envs/cafapk/bin/cafaeval
PY=/Users/a1/miniconda3/envs/esmc/bin/python

for EVAL in runs/human/eval runs/human/sweep/eval runs/human/layers/eval; do
    echo "== $EVAL"
    if [[ ! -d "$EVAL/test_cafaeval_PK_workaround" ]]; then
        mv "$EVAL/test_cafaeval_PK" "$EVAL/test_cafaeval_PK_workaround"
        cp "$EVAL/leaderboard_score.tsv" "$EVAL/leaderboard_score_workaround.tsv"
    fi
    "$CAFA_PK" "$OBO" "$EVAL/predictions" "${GT}_PK.tsv" \
        -ia "$IA" -prop max -norm cafa -th_step 0.01 -threads 4 \
        -known "${GT}_PK_known.tsv" -out_dir "$EVAL/test_cafaeval_PK"
    "$PY" -m democafa.groundtruth.leaderboard_score --DATA_DIR "$EVAL" \
        --output_file "$EVAL/leaderboard_score.tsv"
done
