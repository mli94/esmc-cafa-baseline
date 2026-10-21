#!/bin/bash
# Step-1 experiment: does ESM C's gap to ProtT5 come from the settings?
# Re-uses cached embeddings; sweeps metric (euclidean, cosine) x k (1, 3, 5, 10) for both models.
# Usage (from rotation2/, esmc env on PATH): bash runs/human/sweep.sh
set -e

EMB=runs/human/predictions/embeddings
OUT=runs/human/sweep
TERMS=data/human/processed/train_terms.tsv
QUERY=data/human/processed/test_superset.fasta
CODE=democafa/baselines/esm_container
mkdir -p "$OUT/similarity" "$OUT/predictions"

for model in esmc_300m prot_t5; do
  for metric in euclidean cosine; do
    for k in 1 3 5 10; do
      name="${model}_${metric}_k${k}"
      echo "== $name"
      python "$CODE/process_embeddings.py" \
        "$EMB/test_superset.fasta.$model.h5" "$EMB/train_sequences.fasta.$model.h5" \
        "$OUT/similarity/$name.tsv" --top_k "$k" --metric "$metric"
      python "$CODE/esmc_predict.py" -a "$TERMS" -q "$QUERY" \
        -r "$OUT/similarity/${name}_norm.tsv" -o "$OUT/predictions/$name.tsv"
    done
  done
done

PRED="$OUT/predictions" EVAL="$OUT/eval" bash runs/human/evaluate.sh
