# ESM C Container for Protein Function Prediction

A CAFA-style baseline that uses ESM-Cambrian (ESM C) embeddings to find similar training proteins
and transfers their GO terms, with the same transfer rule as the BLAST and ProtT5 containers.

The pipeline is:
1. Embed query and training proteins with ESM C (mean-pooled over residues; one vector per protein).
2. For each query, find the top-k nearest training proteins (Euclidean or cosine distance).
   Self-hits (same accession) are removed **before** taking the top k, so proteins that are also
   in the training set still get k real neighbors.
3. Convert distances to 0-100 similarity scores:
   - Euclidean: `(max_d - d) / max_d * 100`, with `max_d` the largest retained distance (same as ProtT5)
   - cosine: `(1 - cosine_distance) * 100`, clipped at 0
4. Transfer GO terms from the neighbors with score = similarity / 100, keeping the maximum score per term.

Embeddings are cached, so changing k or the metric re-uses them.

## Files

- `esmc_main.py`: entry point; runs `run_esmc.sh`, then the GO transfer
- `run_esmc.sh`: embeds both FASTA files (or re-uses cached `.h5`) and runs the neighbor search
- `esmc_embedder.py`: FASTA → `.h5` embeddings (key = UniProt accession, value = 1-D float32 vector)
- `process_embeddings.py`: top-k neighbor search and distance normalization (sklearn, or cuml if installed)
- `esmc_predict.py`: GO term transfer, adapted from `blast_chunks_optimized.py` (CPU, or GPU with CuPy)
- `retrieve_terms.py`, `ontology.py`, `config.yaml`: annotation helpers copied from the ProtT5 container
- `requirements.txt`, `Dockerfile`

## Run

```bash
python esmc_main.py \
  --annot_file train_terms.tsv \
  --query_file test_superset.fasta \
  --train_sequences train_sequences.fasta \
  --graph go-basic.obo \
  --output_baseline output/esmc_predictions.tsv
```

Re-use an existing similarity table:

```bash
python esmc_main.py --annot_file train_terms.tsv --query_file test_superset.fasta \
  --graph go-basic.obo --esmc_results output/esmc_results_esmc_300m_euclidean_k3_norm.tsv \
  --output_baseline output/esmc_predictions.tsv
```

Docker:

```bash
docker build -t esmc_predictor democafa/baselines/esm_container
docker run --rm --gpus all -v /path/to/data:/app/data -v /path/to/output:/app/output \
  esmc_predictor --annot_file /app/data/train_terms.tsv --query_file /app/data/test_superset.fasta \
  --train_sequences /app/data/train_sequences.fasta --graph /app/data/go-basic.obo \
  --output_baseline /app/output/esmc_predictions.tsv
```

## Arguments (`esmc_main.py`)

Required: `--annot_file` (3-column `.tsv`, `.gaf`, `.dat`, or `.npz` with `--indices`), `--query_file`,
`--graph`, `--output_baseline`.

Optional:
- `--train_sequences`: training FASTA; required unless `--esmc_results` points to an existing table
- `--esmc_results`: re-use a normalized similarity table (`*_norm.tsv`)
- `--model_name`: `esmc_300m` (default), `esmc_600m`, or `prot_t5` (for comparison through the same pipeline)
- `--top_k` (default 3), `--metric` (`euclidean` default, or `cosine`)
- `--work_dir`: embedding cache (default `<output dir>/embeddings`); `--overwrite_embeddings` to recompute
- `--keep_self_hits`, `--n_terms`, `--num_threads`, `--indices`, `--add_graph`

## Middle-layer embeddings

`esmc_embedder.py --layers 20,25` also saves the output of those transformer blocks
(1-based, before the final norm) to `<output>.L20.h5`, `<output>.L25.h5`, in the same forward pass.
On a human-only test set, middle layers scored clearly better than the last layer
(ESM C 600M layer 24: 0.222 vs 0.174 average IA-weighted F-max), but the layer was chosen
on the test set and should be picked on a validation split. To use a layer file, run
`process_embeddings.py` and `esmc_predict.py` on it directly (see `runs/human/layers.sh`).

## Output

Tab-separated, no header: `<query_id>  <go_term>  <score>`.

## Notes

- Device: CUDA, then Apple MPS, then CPU. On GPU/MPS the esm package runs ESM C in bfloat16.
- Sequences longer than `--max_len` (2048) residues are truncated before embedding.
- The first run downloads the ESM C weights from Hugging Face (`EvolutionaryScale/esmc-300m-2024-12`).
