# Provenance: human-only ESM C baseline experiment

Where every input came from, which code produced each file, and what was changed.
All paths are relative to `rotation2/`.

## 1. Raw data (`data/human/raw/`)

| File | Source | Version / date |
|---|---|---|
| `goa_human_t0.gaf.gz` | EBI GOA archive: `https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN/goa_human.gaf.228.gz` | release 228, generated 2025-05-03 |
| `goa_human_t1.gaf.gz` | EBI GOA current: `https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz` | generated 2026-07-28 (downloaded 2026-10-05) |
| `go-basic_t0.obo` | Copy of `tests/test_data/go-basic-20250601.obo` from the lab repo | GO release 2025-06-01 |
| `go-basic_t1.obo` | `http://purl.obolibrary.org/obo/go/go-basic.obo` | GO release 2026-07-26 |
| `swissprot_human.fasta.gz` | UniProt REST API (reviewed, taxonomy 9606) via `download_fasta_by_taxon.py` | downloaded 2026-10-05; 20,432 sequences (UniProt release at the time likely 2026_03) |
| `taxon.tsv` | Written by hand: `taxon_id<TAB>name` / `9606<TAB>Homo sapiens` | header row is required |

Note: goa_human is built on the UniProt human reference proteome (one protein per gene).

## 2. Processed data and ground truth (lab repo modules, unmodified)

Commands run from `rotation2/` with the `democafaenv` environment:

| Output | Command (module) | Key options |
|---|---|---|
| `processed/goa_human_t{0,1}_filtered.gaf.gz` | `democafa.datacollection.filter_gaf` | `-q swissprot_human.fasta.gz` |
| `processed/t0_terms.tsv` | `democafa.datacollection.retrieve_terms` | `-g go-basic_t0.obo -ag go-basic_t0.obo` (Experimental, IC, TAS) |
| `processed/t1_terms.tsv` | `democafa.datacollection.retrieve_terms` | `-g go-basic_t0.obo -ag go-basic_t1.obo` |
| `processed/train_sequences.fasta`, `train_taxonomy.tsv`, `test_superset.fasta` | `democafa.datacollection.create_test_set` | `--include_all` |
| `groundtruth/groundtruth_{NK,LK,PK,PK_known,targets,terms_of_interest}` | `democafa.groundtruth.classify_ground_truth` | `-g go-basic_t0.obo -g2 go-basic_t1.obo` |
| `processed/IA.txt` | `democafa.datacollection.propagate_and_ia` (uses `utils/ia.py`) | t0 terms, t0 GO |
| `processed/train_terms.tsv` | `cut -f1-3 t0_terms.tsv` | drops the `date` column |

Why t0 also gets `-ag`: it makes t0 propagated with `date = 0`, like t1. `classify_ground_truth`
merges t0 and t1 on all columns, so without it every t1 annotation would count as new.

Counts: 16,991 training proteins, 20,432 queries; NK 160, LK 577, PK 6,107 proteins
(1,350 binding-only MF proteins removed by `classify_ground_truth`).

## 3. Methods and code

| Method | Code | Status |
|---|---|---|
| Naive | `democafa/baselines/naive_container/naive.py` | lab code, unmodified; run on ground-truth targets only, scores < 0.01 dropped |
| BLAST | `democafa/baselines/blast_container/blast_main.py` + `run_blast.sh` + `blast_chunks_optimized.py` | lab code; **one fix** in `blast_chunks_optimized.py` (lines 73-80): accept query IDs without `|` |
| ProtT5 | `esm_container/esmc_main.py --model_name prot_t5` | my pipeline with the ProtT5 encoder; embeddings match `prott5_container/prott5_embedder.py` (cosine 1.000000 on test data) |
| ESM C | `democafa/baselines/esm_container/` | **new code written for this rotation** (below) |

New files in `democafa/baselines/esm_container/`:
- `esmc_embedder.py`: FASTA → mean-pooled embeddings (.h5); `--layers` saves middle layers
- `process_embeddings.py`: top-k nearest neighbors (sklearn, or cuml if present); self-hits removed before top-k; distance → 0-100 similarity
- `esmc_predict.py`: GO transfer, adapted from `blast_chunks_optimized.py` (score = similarity/100, max per term)
- `esmc_main.py`, `run_esmc.sh`: one-command pipeline, adapted from the ProtT5 container
- `retrieve_terms.py`, `ontology.py`, `config.yaml`: copied from the ProtT5 container

Experiment scripts in `runs/human/`: `evaluate.sh`, `sweep.sh` (metric × k), `layers.sh`
(ESM C layers, 300M and 600M), `redo_pk.sh`, `figures/make_figures.py`.

## 4. Evaluation

- NK, LK: `cafaeval` 1.2.1 from PyPI (env `esmc`)
- PK: CAFA-evaluator-PK, `https://github.com/claradepaolis/CAFA-evaluator-PK`, commit `e386b08`
  (env `cafapk`), with `-known groundtruth_PK_known.tsv`
- Options for all: `-ia IA.txt -prop max -norm cafa -th_step 0.01`
- Score: IA-weighted F-max (`f_micro_w`), averaged over NK, LK, PK with `democafa.groundtruth.leaderboard_score`
- The first PK version (known pairs removed from predictions) is kept as `test_cafaeval_PK_workaround/`
  for reference; it overestimated PK because propagation re-adds known ancestor terms.

## 5. Models and software

| Item | Version |
|---|---|
| ESM C weights | Hugging Face `EvolutionaryScale/esmc-300m-2024-12`, `EvolutionaryScale/esmc-600m-2024-12` |
| ProtT5 weights | Hugging Face `Rostlab/prot_t5_xl_half_uniref50-enc` |
| `esm` package | 3.2.1.post1 |
| PyTorch / transformers | 2.14.1 / 4.48.1 |
| scikit-learn / h5py | 1.9.1 / 3.16.0 |
| BLAST+ | 2.17.0 (bioconda, env `blastplus`) |
| Hardware | MacBook (Apple Silicon): embeddings on MPS (ESM C bf16, ProtT5 fp32), everything else on CPU |
