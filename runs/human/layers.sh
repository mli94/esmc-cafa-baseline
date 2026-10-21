#!/bin/bash
# Step-2 experiment: is ESM C's gap to ProtT5 the model (size) or the layer?
#   esmc_300m: blocks 10, 15, 20, 25 + last (30 blocks)
#   esmc_600m: blocks 12, 18, 24, 30 + last (36 blocks)
# Each embedding variant -> k-NN (k = 3, euclidean and cosine) -> GO transfer -> cafaeval.
# Usage (from rotation2/, esmc env on PATH): bash runs/human/layers.sh
set -e

OUT=runs/human/layers
EMB=$OUT/embeddings
TERMS=data/human/processed/train_terms.tsv
QUERY=data/human/processed/test_superset.fasta
TRAIN=data/human/processed/train_sequences.fasta
CODE=democafa/baselines/esm_container
mkdir -p "$EMB" "$OUT/similarity" "$OUT/predictions"

embed() {  # model, layers
  for fasta in "$QUERY" "$TRAIN"; do
    h5="$EMB/$(basename "$fasta").$1.h5"
    [[ -f "$h5" ]] && { echo "Reusing $h5"; continue; }
    python "$CODE/esmc_embedder.py" -i "$fasta" -o "$h5" --model_name "$1" --layers "$2"
  done
}
embed esmc_300m 10,15,20,25
embed esmc_600m 12,18,24,30

for variant in esmc_300m.L10 esmc_300m.L15 esmc_300m.L20 esmc_300m.L25 esmc_300m \
               esmc_600m.L12 esmc_600m.L18 esmc_600m.L24 esmc_600m.L30 esmc_600m; do
  for metric in euclidean cosine; do
    name="${variant/./_}_${metric}_k3"
    echo "== $name"
    python "$CODE/process_embeddings.py" \
      "$EMB/$(basename "$QUERY").$variant.h5" "$EMB/$(basename "$TRAIN").$variant.h5" \
      "$OUT/similarity/$name.tsv" --top_k 3 --metric "$metric"
    python "$CODE/esmc_predict.py" -a "$TERMS" -q "$QUERY" \
      -r "$OUT/similarity/${name}_norm.tsv" -o "$OUT/predictions/$name.tsv"
  done
done

PRED="$OUT/predictions" EVAL="$OUT/eval" bash runs/human/evaluate.sh
