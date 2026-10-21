#!/bin/bash
# Evaluate every prediction file in runs/human/predictions/ separately for NK, LK and PK.
#
# NK/LK use the public cafaeval (esmc env). PK uses CAFA-evaluator-PK (cafapk env,
# https://github.com/claradepaolis/CAFA-evaluator-PK) with -known, which excludes
# t0-known terms after propagation. (Removing known pairs from the predictions instead
# does not work: propagation re-adds known ancestor terms to both sides.)
#
# Usage: bash runs/human/evaluate.sh        (run from the rotation2/ directory, esmc env)
set -e

GT=data/human/groundtruth/groundtruth
OBO=data/human/raw/go-basic_t0.obo
IA=data/human/processed/IA.txt
CAFA_PK=${CAFA_PK:-/Users/a1/miniconda3/envs/cafapk/bin/cafaeval}
PRED=${PRED:-runs/human/predictions}   # override, e.g. PRED=runs/human/sweep/predictions
EVAL=${EVAL:-runs/human/eval}
mkdir -p "$EVAL"

# cafaeval reads every file under pred_dir recursively, so collect only the final
# prediction files (skip esmc_results_* similarity tables and the embeddings/ folder)
ALL_PRED="$EVAL/predictions"
rm -rf "$ALL_PRED" && mkdir -p "$ALL_PRED"
for f in "$PRED"/*.tsv; do
    [[ "$(basename "$f")" == esmc_results_* ]] && continue
    cp "$f" "$ALL_PRED/"
done
PRED="$ALL_PRED"

for S in NK LK; do
    echo "== Evaluating $S"
    cafaeval "$OBO" "$PRED" "${GT}_${S}.tsv" \
        -ia "$IA" -prop max -norm cafa -th_step 0.01 -threads 4 \
        -out_dir "$EVAL/test_cafaeval_$S"
done

echo "== Evaluating PK (with -known)"
"$CAFA_PK" "$OBO" "$PRED" "${GT}_PK.tsv" \
    -ia "$IA" -prop max -norm cafa -th_step 0.01 -threads 4 \
    -known "${GT}_PK_known.tsv" -out_dir "$EVAL/test_cafaeval_PK"

# Combine into one leaderboard table (same script the project uses)
python -m democafa.groundtruth.leaderboard_score --DATA_DIR "$EVAL" --output_file "$EVAL/leaderboard_score.tsv"
