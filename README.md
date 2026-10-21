# democafa

> **About this copy.** This repository is based on [anphan0828/democafa_package](https://github.com/anphan0828/democafa_package)
> and adds an ESM-Cambrian (ESM C) baseline, developed during a lab rotation:
> - `democafa/baselines/esm_container/`: the new ESM C baseline (see its README)
> - `runs/human/`: experiment scripts for a human-only CAFA-style evaluation; `runs/human/PROVENANCE.md` lists where all data and code came from
> - `democafa/baselines/blast_container/blast_chunks_optimized.py`: one fix, accepting query IDs without `|`
> - `statistics/`: GO term and protein information-content analysis (zebrafish), with data and results
>
> Data and results are not included (about 12 GB); they can be regenerated with the scripts in `runs/human/`.

`democafa` is a Python package for building CAFA-style protein function prediction data releases, baseline inputs, ground-truth splits, and leaderboard scores. The code is organized as terminal-runnable modules under `democafa/`; most steps can be run with `python -m democafa.<subpackage>.<module>`.

## Features

* Data processing from UniProt FASTA, UniProt-GOA GAF, UniProt DAT, and GO OBO files.
* Filtering for large GAF files using streaming and optional multiprocessing paths.
* GO evidence-code filtering, alternate-term normalization, ontology pivot filtering, term propagation, and information accretion calculation.
* Training/test sequence preparation for CAFA-style evaluation.
* Ground-truth processing and NK/LK/PK classification.
* Baseline prediction modules and container-oriented baseline folders.

## Installation

Create the main environment and install the package from the repository root:

```bash
conda env create -f democafaenv.yaml
conda activate democafaenv
pip install .
```

For ProtT5-specific baseline work, use the separate environment described in the baseline container notes or create another environment with PyTorch, Transformers, h5py, tqdm, joblib, scikit-learn, scipy, and sentencepiece.

## Package Structure

```text
democafa_package/
    democafa/
        __init__.py
        config.py
        config.yaml
        main.py
        datacollection/
            __init__.py
            compare_fasta.py
            create_test_set.py
            create_train_set.py
            download_fasta_by_taxon.py
            filter_gaf.py
            propagate_and_ia.py
            retrieve_sequences.py
            retrieve_terms.py
        groundtruth/
            __init__.py
            classify_ground_truth.py
            process_ground_truth.py
        baselines/
            blast_container/
            goa_nonexp_container/
            naive_container/
            prott5_container
        utils/
            compare_tsv.py
            constants.py
            ia.py
            ontology.py
            performance.py
    democafaenv.yaml
    pyproject.toml
    setup.py
    README.md
```

## Data Collection Modules

Run modules from the repository root:

```bash
python -m democafa.datacollection.filter_gaf --help
python -m democafa.datacollection.retrieve_terms --help
python -m democafa.datacollection.create_test_set --help
python -m democafa.datacollection.propagate_and_ia --help
```

Main modules:

* `filter_gaf.py`: streams or parses a GAF file and filters records by query accessions and/or taxonomy IDs while preserving GAF headers.
* `retrieve_terms.py`: extracts `EntryID`, `term`, and `aspect` from GAF or DAT annotations after evidence-code filtering and ontology cleanup.
* `download_fasta_by_taxon.py`: downloads reviewed UniProtKB FASTA records for taxonomy IDs listed in a TSV file.
* `create_train_set.py`: writes annotated SwissProt training sequences and an `EntryID<TAB>taxon_id` mapping.
* `create_test_set.py`: creates training FASTA/taxonomy files and optionally a test-superset FASTA, including TrEMBL accessions downloaded from UniProt when needed.
* `propagate_and_ia.py`: propagates terms through a supplied OBO graph and writes propagated annotations plus IA values. Uses official IA calculation from [here](https://github.com/claradepaolis/InformationAccretion).
* `retrieve_sequences.py`: extracts selected sequences from a FASTA file using an accession list.
* `compare_fasta.py`: compares two FASTA files by UniProt accession and writes sequence differences.

## Ground Truth Modules

Run modules from the repository root:

```bash
python -m democafa.groundtruth.process_ground_truth --help
python -m democafa.groundtruth.classify_ground_truth --help
```

Main modules:

* `process_ground_truth.py`: converts raw holdout annotation TSVs into the package terms schema and applies evidence, ontology, isoform, and binding-only filters.
* `classify_ground_truth.py`: compares known and later annotations for a query set and writes NK, LK, PK, PK-known, target, and terms-of-interest files.

## Typical Flow

1. Download or select SwissProt FASTA records with `download_fasta_by_taxon.py`.
2. Filter UniProt-GOA to the selected FASTA/query set with `filter_gaf.py`.
3. Extract training terms with `retrieve_terms.py`.
4. Create train/test sequence files with `create_test_set.py` or `create_train_set.py`.
5. Propagate training terms and compute IA with `propagate_and_ia.py`.
6. Build later-timepoint or holdout ground truth with `process_ground_truth.py` and `classify_ground_truth.py`.

## Notes

* Baselines are constructed against the submission-time ontology. Containerized baselines are available on [Dockerhub](https://hub.docker.com/repositories/anphan0828).
* Prediction and ground-truth files should be propagated and filtered against the same pivot ontology before comparison.
* For ontology pivots, use `retrieve_terms.py --add_graph` to remove terms not present in the frozen graph.
* Proteins whose sequences changed between release and evaluation should be removed from the evaluation set before final scoring.
