#!/usr/bin/env python3
"""Transfer GO terms from ESM C nearest neighbors to query proteins.

Adapted from blast_container/blast_chunks_optimized.py. Each neighbor's
annotations are transferred with weight Similarity / 100; when several
neighbors share a term, the query keeps the maximum weight for that term.

Input similarity table (from process_embeddings.py, with header):
    Query ID, DB ID, e-val, Length, Similarity, N-ident
Output (no header):
    <query_id>  <go_term>  <score>
"""

import os
import sys
import csv
import gzip
import shutil
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse
import pickle as cp
from tqdm import tqdm

from retrieve_terms import wrapper_retrieve_terms
from ontology import sparse_matrix_and_indices

try:
    import cupy as cp_gpu
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def accession(ids):
    """'sp|Q7L5A3|ATOSB_HUMAN' -> 'Q7L5A3'; IDs without '|' are returned unchanged."""
    ids = ids.astype(str)
    parts = ids.str.split('|')
    return parts.str[1].where(parts.str.len() >= 2, ids)


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def output_dir_for(path):
    return os.path.dirname(path) or '.'


def read_query_ids(query_file):
    """Query IDs from a FASTA (plain or .gz) or a text file with one ID per line."""
    opener = gzip.open if query_file.endswith('.gz') else open
    with opener(query_file, 'rt') as handle:
        first = handle.readline()
        handle.seek(0)
        if first.startswith('>'):
            ids = [line[1:].split()[0] for line in handle if line.startswith('>')]
        else:
            ids = [line.strip() for line in handle if line.strip()]
    return list(accession(pd.Series(ids, dtype=str)))


def partition_hits_single_pass(sim_results, query_set, query_to_bucket,
                               keep_self_hits, chunksize, num_buckets):
    """Stream the similarity table once, bucketizing hits per query."""

    temp_dir = tempfile.mkdtemp(prefix='esmc_hits_')
    bucket_paths = [os.path.join(temp_dir, f'bucket_{i}.tsv') for i in range(num_buckets)]
    bucket_has_header = [False] * num_buckets
    bucket_query_sets = [set() for _ in range(num_buckets)]
    queries_with_hits = set()

    for chunk in pd.read_csv(
        sim_results,
        sep='\t',
        header=0,
        usecols=[0, 1, 4],  # Query ID, DB ID, Similarity
        names=['qseqid', 'sseqid', 'similarity'],
        dtype={'qseqid': str, 'sseqid': str, 'similarity': np.float32},
        chunksize=chunksize,
    ):
        chunk['qseqid_acc'] = accession(chunk['qseqid'])
        chunk = chunk[chunk['qseqid_acc'].isin(query_set)]
        if chunk.empty:
            continue

        chunk['sseqid_acc'] = accession(chunk['sseqid'])
        if not keep_self_hits:
            chunk = chunk[chunk['sseqid_acc'] != chunk['qseqid_acc']]
        if chunk.empty:
            continue

        chunk['bucket'] = chunk['qseqid_acc'].map(query_to_bucket)

        for bucket_idx, bucket_df in chunk.groupby('bucket'):
            bucket_df = bucket_df[['qseqid_acc', 'sseqid_acc', 'similarity']]
            mode = 'w' if not bucket_has_header[bucket_idx] else 'a'
            bucket_df.to_csv(
                bucket_paths[bucket_idx],
                sep='\t',
                header=not bucket_has_header[bucket_idx],
                index=False,
                mode=mode,
            )
            bucket_has_header[bucket_idx] = True
            qids = bucket_df['qseqid_acc'].unique()
            bucket_query_sets[bucket_idx].update(qids)
            queries_with_hits.update(qids)

    return temp_dir, bucket_paths, bucket_query_sets, queries_with_hits


def prepare_hit_arrays(batch_hits_df, proteins, batch_query_to_local):
    """Return per-hit arrays after deduplicating repeated query-subject pairs."""

    if batch_hits_df.empty:
        return None

    batch_hits_df['protein_idx'] = batch_hits_df['sseqid_acc'].map(proteins)
    batch_hits_df = batch_hits_df.dropna(subset=['protein_idx'])
    if batch_hits_df.empty:
        return None

    batch_hits_df['weight'] = batch_hits_df['similarity'].to_numpy(dtype=np.float32) / 100.0
    batch_hits_df = batch_hits_df[batch_hits_df['weight'] > 0]
    if batch_hits_df.empty:
        return None

    batch_hits_df = batch_hits_df.sort_values(
        by=['qseqid_acc', 'sseqid_acc', 'weight'],
        ascending=[True, True, False],
        kind='mergesort',
    )
    batch_hits_df = batch_hits_df.drop_duplicates(subset=['qseqid_acc', 'sseqid_acc'], keep='first')

    batch_hits_df['query_local_idx'] = batch_hits_df['qseqid_acc'].map(batch_query_to_local)
    batch_hits_df = batch_hits_df.dropna(subset=['query_local_idx'])
    if batch_hits_df.empty:
        return None

    query_idx = batch_hits_df['query_local_idx'].to_numpy(dtype=np.int32)
    protein_idx = batch_hits_df['protein_idx'].to_numpy(dtype=np.int32)
    weights = batch_hits_df['weight'].to_numpy(dtype=np.float32)
    return query_idx, protein_idx, weights


def collect_batch_results(batch_query_names, batch_scores, term_names, n_terms):
    """Convert dense batch scores into prediction rows."""

    results = []
    for local_idx, qid in enumerate(batch_query_names):
        row = batch_scores[local_idx]
        nz_terms = np.flatnonzero(row)
        if nz_terms.size == 0:
            continue
        if n_terms is not None and nz_terms.size > n_terms:
            top_idx = nz_terms[np.argsort(row[nz_terms])[-n_terms:]]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        else:
            top_idx = nz_terms
        for term_idx in top_idx:
            results.append([qid, term_names[term_idx], round(float(row[term_idx]), 4)])
    return results


def process_query_batch_cpu(batch_query_names, batch_hits_df, annotation_mat,
                            proteins, term_names, n_terms):
    batch_query_to_local = {qid: idx for idx, qid in enumerate(batch_query_names)}
    prepared = prepare_hit_arrays(batch_hits_df, proteins, batch_query_to_local)
    if prepared is None:
        return []
    query_idx, protein_idx, weights = prepared

    hit_annots = annotation_mat[protein_idx, :].tocoo()
    if hit_annots.nnz == 0:
        return []

    weighted_data = hit_annots.data * weights[hit_annots.row]
    term_indices = hit_annots.col
    query_assignments = query_idx[hit_annots.row]

    batch_scores = np.zeros((len(batch_query_names), annotation_mat.shape[1]), dtype=np.float32)
    # ufunc.at handles repeated (query, term) pairs; plain fancy assignment would keep only the last
    np.maximum.at(batch_scores, (query_assignments, term_indices), weighted_data)

    return collect_batch_results(batch_query_names, batch_scores, term_names, n_terms)


def process_query_batch_gpu(batch_query_names, batch_hits_df, annotation_mat_gpu,
                            proteins, term_names, n_terms):
    batch_query_to_local = {qid: idx for idx, qid in enumerate(batch_query_names)}
    prepared = prepare_hit_arrays(batch_hits_df, proteins, batch_query_to_local)
    if prepared is None:
        return []
    query_idx, protein_idx, weights = prepared

    query_idx_gpu = cp_gpu.asarray(query_idx)
    protein_idx_gpu = cp_gpu.asarray(protein_idx)
    weights_gpu = cp_gpu.asarray(weights)

    hit_annots = annotation_mat_gpu[protein_idx_gpu, :]
    if hit_annots.nnz == 0:
        return []
    hit_annots = hit_annots.tocoo()

    weighted_data = hit_annots.data * weights_gpu[hit_annots.row]
    term_indices = hit_annots.col
    query_assignments = query_idx_gpu[hit_annots.row]

    batch_scores = cp_gpu.zeros((len(batch_query_names), annotation_mat_gpu.shape[1]), dtype=cp_gpu.float32)
    cp_gpu.maximum.at(batch_scores, (query_assignments, term_indices), weighted_data)

    return collect_batch_results(batch_query_names, cp_gpu.asnumpy(batch_scores), term_names, n_terms)


def process_batch_cpu_task(task):
    return process_query_batch_cpu(*task)


def load_annotations(annot_file, indices, graph, config_path, output_dir):
    """Return (annotation matrix as CSR, {protein: row}, {term: col})."""
    if '.gaf' in annot_file or '.dat' in annot_file:
        if not graph:
            sys.exit("Please provide a graph file for GAF or DAT input")
        import yaml
        with open(config_path, 'r') as f:
            go_codes = yaml.safe_load(f).get('go_codes', {})
        temp_terms_path = os.path.join(output_dir, 'esmc_terms.tsv')
        wrapper_retrieve_terms(
            annot_file=annot_file,
            go_codes=go_codes,
            selected_go_codes='Experimental,IC,TAS',
            graph=graph,
            output_tsv=temp_terms_path,
        )
        terms_df = pd.read_csv(temp_terms_path, sep='\t', header=0, names=['EntryID', 'term', 'aspect'])
        os.remove(temp_terms_path)
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
    elif '.tsv' in annot_file:
        terms_df = pd.read_csv(annot_file, sep='\t', header=0)
        if terms_df.shape[1] != 3:
            sys.exit("Invalid TSV format. Expected 3 columns: EntryID, term, aspect.")
        annotation_mat, proteins, terms, _ = sparse_matrix_and_indices(terms_df)
    elif annot_file.endswith('.npz'):
        if not indices:
            sys.exit("Please provide a term indices file for matrix input")
        annotation_mat = sparse.load_npz(annot_file)
        with open(indices, 'rb') as f:
            proteins, terms = cp.load(f)
    else:
        sys.exit("Invalid annotation file format")
    # CSR makes the per-batch row selection fast
    return sparse.csr_matrix(annotation_mat, dtype=np.float32), proteins, terms


def esmc_predict(annot_file, query_file, indices, graph, add_graph,
                 esmc_results, output_baseline, config_path=None, keep_self_hits=False,
                 n_terms=None, batch_size=1000, num_threads=4):
    """Make predictions for query sequences based on ESM C nearest neighbors."""

    output_dir = output_dir_for(output_baseline)
    os.makedirs(output_dir, exist_ok=True)

    annotation_mat, proteins, terms = load_annotations(annot_file, indices, graph, config_path, output_dir)
    print(f"Annotation matrix shape: {annotation_mat.shape}")
    print(f"Number of proteins: {len(proteins)} | Number of terms: {len(terms)}")

    query_ids = read_query_ids(query_file)
    print(f"Loaded {len(query_ids)} query IDs")

    query_set = set(query_ids)
    query_order = {qid: idx for idx, qid in enumerate(query_ids)}
    estimated_buckets = max(1, len(query_ids) // max(batch_size, 1)) * 4
    num_buckets = min(4096, max(1, estimated_buckets))
    query_to_bucket = {qid: hash(qid) % num_buckets for qid in query_ids}

    print(f"Partitioning hits from {esmc_results} into {num_buckets} buckets (single pass)...")
    temp_dir, bucket_paths, bucket_query_sets, queries_with_hits = partition_hits_single_pass(
        sim_results=esmc_results,
        query_set=query_set,
        query_to_bucket=query_to_bucket,
        keep_self_hits=keep_self_hits,
        chunksize=1_000_000,
        num_buckets=num_buckets,
    )

    print(f"Processing {len(queries_with_hits)} query sequences with neighbors")
    if not queries_with_hits:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("No neighbors available for the provided queries.")
        return

    term_names = [None] * len(terms)
    for term, idx in terms.items():
        term_names[idx] = term

    annotation_mat_gpu = None
    if GPU_AVAILABLE:
        print("Transferring annotation matrix to GPU...")
        annotation_mat_gpu = cp_sparse.csr_matrix(annotation_mat)
    else:
        print("GPU not available, proceeding with CPU computation.")

    open_func = gzip.open if output_baseline.endswith('.gz') else open
    total_batches = sum((len(s) + batch_size - 1) // batch_size for s in bucket_query_sets)
    progress = tqdm(total=total_batches, desc="Processing batches") if total_batches else None
    n_rows = 0

    try:
        with open_func(output_baseline, 'wt', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')

            for bucket_idx, bucket_path in enumerate(bucket_paths):
                bucket_queries = bucket_query_sets[bucket_idx]
                if not bucket_queries or not os.path.exists(bucket_path):
                    continue
                bucket_df = pd.read_csv(bucket_path, sep='\t', dtype={'similarity': np.float32})
                sorted_queries = sorted(bucket_queries, key=lambda q: query_order.get(q, sys.maxsize))

                batch_tasks = []
                for batch_query_names in chunk_list(sorted_queries, batch_size):
                    batch_hits_df = bucket_df[bucket_df['qseqid_acc'].isin(batch_query_names)].copy()
                    if batch_hits_df.empty:
                        if progress:
                            progress.update(1)
                        continue
                    batch_tasks.append((batch_query_names, batch_hits_df, annotation_mat,
                                        proteins, term_names, n_terms))

                if annotation_mat_gpu is not None:
                    results_iter = (
                        process_query_batch_gpu(names, hits, annotation_mat_gpu, proteins, term_names, n_terms)
                        for names, hits, *_ in batch_tasks
                    )
                elif num_threads > 1 and len(batch_tasks) > 1:
                    executor = ThreadPoolExecutor(max_workers=min(num_threads, len(batch_tasks)))
                    results_iter = executor.map(process_batch_cpu_task, batch_tasks)
                else:
                    results_iter = map(process_batch_cpu_task, batch_tasks)

                for batch_results in results_iter:
                    if batch_results:
                        writer.writerows(batch_results)
                        n_rows += len(batch_results)
                    if progress:
                        progress.update(1)
                del bucket_df
    finally:
        if progress:
            progress.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Wrote {n_rows} predictions to {output_baseline}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='ESM C nearest-neighbor GO term transfer')
    parser.add_argument('--annot_file', '-a', required=True,
                        help='Training annotations: 3-column .tsv, .gaf, .dat, or sparse .npz')
    parser.add_argument('--indices', '-i', default=None,
                        help='Protein & term indices pickle (required with .npz annotations)')
    parser.add_argument('--graph', default=None, help='GO OBO file (required with .gaf/.dat annotations)')
    parser.add_argument('--add_graph', default=None, help='Additional OBO file (reserved)')
    parser.add_argument('--query_file', '-q', required=True,
                        help='Query FASTA (plain or .gz) or text file with one ID per line')
    parser.add_argument('--esmc_results', '-r', required=True,
                        help='Normalized similarity table from process_embeddings.py (*_norm.tsv)')
    parser.add_argument('--output_baseline', '-o', required=True,
                        help='Output predictions file (.gz to compress)')
    parser.add_argument('--keep_self_hits', action='store_true', help='Keep self-hits')
    parser.add_argument('--n_terms', '-n', type=int, default=None,
                        help='Upper limit for number of terms per target')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Number of queries to process per batch (default: 1000)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of CPU threads for batch processing (default: 4)')
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    if args.num_threads < 1:
        raise SystemExit('--num_threads must be a positive integer')
    esmc_predict(
        annot_file=args.annot_file,
        query_file=args.query_file,
        indices=args.indices,
        graph=args.graph,
        add_graph=args.add_graph,
        esmc_results=args.esmc_results,
        output_baseline=args.output_baseline,
        config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
        keep_self_hits=args.keep_self_hits,
        n_terms=args.n_terms,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == '__main__':
    main()
