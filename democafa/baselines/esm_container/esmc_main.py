#!/usr/bin/env python3
"""
ESM C Container Main Script

This script orchestrates the ESM C-based protein function prediction pipeline:
1. Runs run_esmc.sh to embed sequences and find top-k nearest training proteins
2. Transfers annotations from the neighbors and keeps the maximum score per term

Usage:
    python esmc_main.py --annot_file <annotations> --query_file <queries> --train_sequences <training> --graph <ontology> --output_baseline <output>
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_esmc_analysis(query_file, train_sequences, esmc_results, work_dir=None,
                      model_name='esmc_300m', top_k=3, metric='euclidean',
                      keep_self_hits=False, overwrite=False):
    """Run embedding + nearest-neighbor search via run_esmc.sh."""

    esmc_script = Path(__file__).resolve().parent / "run_esmc.sh"
    if not esmc_script.exists():
        raise FileNotFoundError(f"ESM C script not found: {esmc_script}")

    print("Running ESM C analysis...")
    print(f"Query: {query_file}")
    print(f"Database: {train_sequences}")
    print(f"Output: {esmc_results}")

    cmd = [
        "bash", str(esmc_script),
        "--query", query_file,
        "--database", train_sequences,
        "--output", esmc_results,
        "--model_name", model_name,
        "--top_k", str(top_k),
        "--metric", metric,
    ]
    if work_dir:
        cmd.extend(["--work_dir", work_dir])
    if keep_self_hits:
        cmd.append("--keep_self_hits")
    if overwrite:
        cmd.append("--overwrite")

    sys.stdout.flush()  # keep our log lines ahead of the subprocess output
    # stream output so embedding progress is visible while it runs;
    # PYTHON makes the shell script use this same interpreter / environment
    subprocess.run(cmd, check=True, env={**os.environ, 'PYTHON': sys.executable})


def main():
    parser = argparse.ArgumentParser(description='ESM C-based protein function prediction container')

    # Required arguments
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Training annotations: 3-column .tsv, .gaf, .dat, or sparse .npz')
    parser.add_argument('--query_file', '-q', required=True,
                        help='FASTA file containing query sequences')
    parser.add_argument('--graph', required=True,
                        help='Path to GO ontology file (.obo)')
    parser.add_argument('--output_baseline', '-o', required=True,
                        help='Path to output predictions file')

    # Optional arguments
    parser.add_argument('--train_sequences', default=None,
                        help='FASTA file containing training sequences (required unless --esmc_results already exists)')
    parser.add_argument('--esmc_results', default=None,
                        help='Path to an existing normalized similarity table (*_norm.tsv); skips step 1')
    parser.add_argument('--indices', '-i', default=None,
                        help='Path to term indices file (required for .npz annotation files)')
    parser.add_argument('--add_graph', default=None,
                        help='Path to additional OBO file for temporal filtering')
    parser.add_argument('--work_dir', default=None,
                        help='Directory for cached embeddings (default: <output dir>/embeddings)')
    parser.add_argument('--model_name', default='esmc_300m', choices=['esmc_300m', 'esmc_600m', 'prot_t5'],
                        help='Embedding model (default: esmc_300m); prot_t5 is for comparison')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of nearest neighbors kept per query (default: 3)')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cosine'],
                        help='Distance used for neighbor search and scoring (default: euclidean)')
    parser.add_argument('--keep_self_hits', action='store_true',
                        help='Keep self-hits (a query matching itself in the training set)')
    parser.add_argument('--overwrite_embeddings', action='store_true',
                        help='Recompute embeddings even if cached files exist')
    parser.add_argument('--n_terms', type=int, default=None,
                        help='Maximum number of terms to produce predictions per target (default: None)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of CPU threads for annotation transfer (default: 4)')

    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error('--num_threads must be a positive integer')
    if args.top_k < 1:
        parser.error('--top_k must be a positive integer')

    # Validate input files
    for label, path in [('Annotation', args.annot_file), ('Query', args.query_file),
                        ('Ontology', args.graph), ('Training sequences', args.train_sequences)]:
        if path and not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            sys.exit(1)

    output_dir = os.path.dirname(args.output_baseline) or '.'
    need_esmc_analysis = not (args.esmc_results and os.path.exists(args.esmc_results))
    if need_esmc_analysis and not args.train_sequences:
        print("Error: --train_sequences is required unless --esmc_results points to an existing similarity table")
        sys.exit(1)

    try:
        if need_esmc_analysis:
            esmc_results_raw = os.path.join(output_dir, f'esmc_results_{args.model_name}_{args.metric}_k{args.top_k}.tsv')
            esmc_results_norm = esmc_results_raw.replace('.tsv', '_norm.tsv')
            print("Step 1: Running ESM C embedding and neighbor search...")
            run_esmc_analysis(
                query_file=args.query_file,
                train_sequences=args.train_sequences,
                esmc_results=esmc_results_raw,
                work_dir=args.work_dir,
                model_name=args.model_name,
                top_k=args.top_k,
                metric=args.metric,
                keep_self_hits=args.keep_self_hits,
                overwrite=args.overwrite_embeddings,
            )
            if not os.path.exists(esmc_results_norm):
                print(f"Error: Normalized ESM C results file not found: {esmc_results_norm}")
                sys.exit(1)
        else:
            esmc_results_norm = args.esmc_results
            print(f"Step 1: Using provided ESM C results: {esmc_results_norm}")

        print("Step 2: Generating predictions from ESM C neighbors...")
        from esmc_predict import esmc_predict
        esmc_predict(
            annot_file=args.annot_file,
            query_file=args.query_file,
            indices=args.indices,
            graph=args.graph,
            add_graph=args.add_graph,
            esmc_results=esmc_results_norm,
            output_baseline=args.output_baseline,
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
            keep_self_hits=args.keep_self_hits,
            n_terms=args.n_terms,
            num_threads=args.num_threads,
        )

        print("ESM C prediction pipeline completed successfully!")
        print(f"Results saved to: {args.output_baseline}")

    except Exception as e:
        print(f"Error in ESM C prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
