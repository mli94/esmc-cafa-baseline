#!/usr/bin/env python3
"""Nearest-neighbor search between query and database ESM C embeddings.

For each query, keep the top-k closest database proteins and convert their
distances into a 0-100 similarity score. Self-hits (same accession in query
and database) are excluded *before* truncating to top-k, so every query that
also sits in the training set still gets k real neighbors.

Writes two TSVs in the same layout as the ProtT5 container:
    <output>.tsv       raw distances in the 'Similarity' column
    <output>_norm.tsv  normalized similarity (0-100) in the 'Similarity' column
Columns: Query ID, DB ID, e-val, Length, Similarity, N-ident
(e-val, Length and N-ident are placeholders kept for format compatibility.)
"""
import argparse
import sys

import h5py
import numpy as np
import pandas as pd

COLUMNS = ['Query ID', 'DB ID', 'e-val', 'Length', 'Similarity', 'N-ident']


def load_h5(path):
    with h5py.File(path, 'r') as hf:
        ids = list(hf.keys())
        mat = np.stack([hf[i][:] for i in ids]).astype(np.float32)
    return ids, mat


def knn(query_mat, db_mat, n_neighbors, metric):
    """Return (distances, indices), each of shape (n_query, n_neighbors), nearest first."""
    try:
        from cuml.neighbors import NearestNeighbors  # GPU, if available
        backend = 'cuml'
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        backend = 'sklearn'
    print(f"k-NN backend: {backend} | metric: {metric} | k: {n_neighbors}")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')
    nn.fit(db_mat)
    dist, idx = nn.kneighbors(query_mat)
    return np.asarray(dist), np.asarray(idx)


def top_k_neighbors(query_ids, query_mat, db_ids, db_mat, top_k, metric, keep_self_hits):
    # ask for one extra neighbor so a removed self-hit can be replaced
    n_neighbors = min(top_k + (0 if keep_self_hits else 1), len(db_ids))
    if n_neighbors < top_k:
        print(f"Warning: database has only {len(db_ids)} proteins; using top_k={n_neighbors}")
    dist, idx = knn(query_mat, db_mat, n_neighbors, metric)

    rows = []
    for qi, qid in enumerate(query_ids):
        kept = 0
        for d, di in zip(dist[qi], idx[qi]):
            sid = db_ids[di]
            if not keep_self_hits and sid == qid:
                continue
            rows.append((qid, sid, float(d)))
            kept += 1
            if kept == top_k:
                break
    df = pd.DataFrame(rows, columns=['Query ID', 'DB ID', 'distance'])
    print(f"Retained {len(df)} neighbor rows for {df['Query ID'].nunique()} queries")
    return df


def normalize(df, metric):
    """Convert distances into 0-100 similarity scores.

    euclidean: (max_d - d) / max_d * 100, where max_d is the largest retained
               distance (same rule as the ProtT5 container).
    cosine:    (1 - cosine_distance) * 100, clipped at 0; does not depend on
               the rest of the dataset.
    """
    d = df['distance'].to_numpy()
    if metric == 'cosine':
        sim = np.clip(1.0 - d, 0.0, 1.0) * 100
    else:
        max_d = d.max()
        sim = np.full_like(d, 100.0) if max_d == 0 else (max_d - d) / max_d * 100
    return np.round(sim, 2)


def to_table(df, score):
    out = pd.DataFrame({
        'Query ID': df['Query ID'],
        'DB ID': df['DB ID'],
        'e-val': 0,
        'Length': 0,
        'Similarity': score,
        'N-ident': 0,
    })
    return out[COLUMNS]


def process_embeddings(query_h5, db_h5, output_file, top_k=3, metric='euclidean',
                       keep_self_hits=False):
    query_ids, query_mat = load_h5(query_h5)
    db_ids, db_mat = load_h5(db_h5)
    print(f"Query: {query_mat.shape} | Database: {db_mat.shape}")
    if query_mat.shape[1] != db_mat.shape[1]:
        raise ValueError("Query and database embeddings have different dimensions "
                         "(were they made with the same model?)")

    df = top_k_neighbors(query_ids, query_mat, db_ids, db_mat, top_k, metric, keep_self_hits)

    to_table(df, df['distance'].round(6)).to_csv(output_file, sep='\t', index=False)
    norm_file = output_file.replace('.tsv', '_norm.tsv')
    to_table(df, normalize(df, metric)).to_csv(norm_file, sep='\t', index=False)
    print(f"Raw distances -> {output_file}")
    print(f"Normalized similarities -> {norm_file}")
    return norm_file


def parse_args(argv):
    p = argparse.ArgumentParser(description='Top-k nearest neighbors between ESM C embedding files.')
    p.add_argument('query_h5', help='HDF5 file with query embeddings')
    p.add_argument('db_h5', help='HDF5 file with database (training) embeddings')
    p.add_argument('output_file', help='Raw output TSV (must end with .tsv)')
    p.add_argument('--top_k', type=int, default=3, help='Neighbors kept per query (default: 3)')
    p.add_argument('--metric', default='euclidean', choices=['euclidean', 'cosine'])
    p.add_argument('--keep_self_hits', action='store_true',
                   help='Allow a query to be its own neighbor (default: excluded)')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args.top_k < 1:
        raise SystemExit('--top_k must be a positive integer')
    if not args.output_file.endswith('.tsv'):
        raise SystemExit('output_file must end with .tsv')
    process_embeddings(args.query_h5, args.db_h5, args.output_file,
                       args.top_k, args.metric, args.keep_self_hits)
