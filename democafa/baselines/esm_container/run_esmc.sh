#!/bin/bash

# ESM C helper: embed query and database FASTAs, then find top-k neighbors
# Embeddings are cached in --work_dir and reused when the same file/model was embedded before

set -e  # Exit on any error

# Set default values
QUERY_FILE=""
DATABASE_FILE=""
OUTPUT_FILE=""
WORK_DIR=""
MODEL_NAME="esmc_300m"
TOP_K=3
METRIC="euclidean"
KEEP_SELF_HITS=""
OVERWRITE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}  # esmc_main.py sets this to its own interpreter

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)          QUERY_FILE="$2"; shift 2 ;;
        --database)       DATABASE_FILE="$2"; shift 2 ;;
        --output)         OUTPUT_FILE="$2"; shift 2 ;;
        --work_dir)       WORK_DIR="$2"; shift 2 ;;
        --model_name)     MODEL_NAME="$2"; shift 2 ;;
        --top_k)          TOP_K="$2"; shift 2 ;;
        --metric)         METRIC="$2"; shift 2 ;;
        --keep_self_hits) KEEP_SELF_HITS="--keep_self_hits"; shift ;;
        --overwrite)      OVERWRITE=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$QUERY_FILE" || -z "$DATABASE_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "Usage: $0 --query <query_fasta> --database <database_fasta> --output <output.tsv> [options]"
    echo "  --work_dir:       Directory for cached embeddings (default: <output dir>/embeddings)"
    echo "  --model_name:     esmc_300m or esmc_600m (default: esmc_300m)"
    echo "  --top_k:          Neighbors kept per query (default: 3)"
    echo "  --metric:         euclidean or cosine (default: euclidean)"
    echo "  --keep_self_hits: Allow a query to be its own neighbor"
    echo "  --overwrite:      Recompute embeddings even if cached files exist"
    exit 1
fi

if ! [[ "$TOP_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --top_k must be a positive integer"
    exit 1
fi

for f in "$QUERY_FILE" "$DATABASE_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Input file '$f' not found"
        exit 1
    fi
done

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
WORK_DIR=${WORK_DIR:-"$OUTPUT_DIR/embeddings"}
mkdir -p "$OUTPUT_DIR" "$WORK_DIR"

# Cache name = input file name + model, so a different input or model never reuses a stale file
QUERY_H5="$WORK_DIR/$(basename "$QUERY_FILE").${MODEL_NAME}.h5"
DB_H5="$WORK_DIR/$(basename "$DATABASE_FILE").${MODEL_NAME}.h5"

embed() {
    local fasta="$1" h5="$2"
    if [[ -f "$h5" && "$OVERWRITE" -eq 0 ]]; then
        echo "Reusing cached embeddings: $h5"
    else
        "$PYTHON" "$SCRIPT_DIR/esmc_embedder.py" --input "$fasta" --output "$h5" --model_name "$MODEL_NAME"
    fi
}

echo "Step 1: Embedding query set with $MODEL_NAME..."
embed "$QUERY_FILE" "$QUERY_H5"

echo "Step 2: Embedding database set with $MODEL_NAME..."
embed "$DATABASE_FILE" "$DB_H5"

echo "Step 3: Finding top-$TOP_K neighbors ($METRIC)..."
"$PYTHON" "$SCRIPT_DIR/process_embeddings.py" "$QUERY_H5" "$DB_H5" "$OUTPUT_FILE" \
    --top_k "$TOP_K" --metric "$METRIC" $KEEP_SELF_HITS

OUTPUT_NORM=${OUTPUT_FILE/.tsv/_norm.tsv}
if [[ -f "$OUTPUT_NORM" ]]; then
    echo "ESM C analysis completed: $OUTPUT_NORM ($(($(wc -l < "$OUTPUT_NORM") - 1)) neighbor rows)"
else
    echo "Error: Output file was not created"
    exit 1
fi
