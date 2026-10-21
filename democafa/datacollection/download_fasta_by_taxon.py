#! /usr/bin/env python3
"""Download reviewed UniProt records for one or more taxonomy IDs.

The primary CLI downloads reviewed UniProtKB FASTA records for taxon IDs listed
in the first column of a TSV file. The TSV helper is kept for ad hoc annotation
audits and does not run from the command-line entry point.
"""

import sys
import os
import requests
import re
import pandas as pd
import gzip
import argparse
from requests.adapters import HTTPAdapter, Retry

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)


def download_fasta_by_taxon(taxon, output_file):
    """
    Download FASTA sequences from UniProt for a given taxon ID and save to output_file
    """
    taxon_id = get_taxon_id_from_file(taxon)
    taxon_query = f'%28taxonomy_id%3A{"+OR+taxonomy_id%3A".join(taxon_id)}%29'
    # Get plain text response, then write to gzip file
    url = f"https://rest.uniprot.org/uniprotkb/search?format=fasta&query={taxon_query}+AND+%28reviewed%3Atrue%29&size=500"
    print(f"Downloading FASTA sequences for taxon ID {taxon_id}...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Bug: 2 sequences merged into 1 header, total number does not match expected total
    # progress = 0
    # with gzip.open(output_file, 'wt') as f:
    #     for batch, total in get_batch(url):
    #         lines = batch.text.splitlines()
    #         if not progress:
    #             print(lines[0], file=f)
    #         for line in lines[1:]:
    #             if line.startswith('>'):
    #                 progress += 1
    #             print(line, file=f)
    #         print(f'{progress} / {total}')
    # Fix: Make progress cumulative
    total_progress = 0
    expected_total = None

    with gzip.open(output_file, 'wt') as f:
        for batch_num, (batch, total) in enumerate(get_batch(url), 1):
            if expected_total is None:
                expected_total = int(total)

            lines = batch.text.splitlines()
            batch_sequences = 0

            for line in lines:
                if line.startswith('>'):
                    batch_sequences += 1
                    total_progress += 1
                print(line, file=f)

            print(f'Batch {batch_num}: +{batch_sequences} sequences | Total: {total_progress:,} / {expected_total:,}')

def download_tsv_by_taxon(taxon_id, output_file, release_test_set=None, train_terms=None, missing_output=None):
    """
    Download a reviewed UniProtKB TSV for taxon IDs and optionally audit Function comments.

    Args:
        taxon_id: Iterable of taxonomy IDs.
        output_file: Gzipped TSV output path.
        release_test_set: Optional test-superset ID file. When provided with
            ``train_terms``, newly-added proteins are excluded from the audit.
        train_terms: Optional training terms TSV used to count annotated entries
            with missing Function comments.
        missing_output: Optional path for proteins missing Function comments.
    """
    # Chunk taxon IDs into batches of 200 to avoid URL length limits
    chunk_size = 100  # Safe chunk size; tune as needed
    taxon_chunks = [taxon_id[i:i+chunk_size] for i in range(0, len(taxon_id), chunk_size)]

    total_progress = 0
    expected_total = None
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    wrote_header = False
    with gzip.open(output_file, 'wt') as f:
        for chunk_idx, chunk in enumerate(taxon_chunks, 1):
            taxon_query = f'%28taxonomy_id%3A{"+OR+taxonomy_id%3A".join(chunk)}%29'
            # Get plain text response, then write to gzip file
            url = f"https://rest.uniprot.org/uniprotkb/search?&fields=accession%2Corganism_name%2Corganism_id%2Ccc_function&format=tsv&query={taxon_query}+AND+%28reviewed%3Atrue%29&size=500"
            print(f"Downloading chunk {chunk_idx}/{len(taxon_chunks)} ({len(chunk)} taxon IDs)...")

            for batch_num, (batch, total) in enumerate(get_batch(url), 1):
                if expected_total is None:
                    expected_total = int(total)

                lines = batch.text.splitlines()
                if not lines:
                    continue
                if not wrote_header:
                    print(lines[0], file=f)
                    wrote_header = True

                for line in lines[1:]:  # Skip duplicate batch headers
                    # if line.startswith('>'):
                    #     progress += 1
                    print(line, file=f)
                total_progress += len(lines) - 1  # Subtract 1 to account for header line
                # print(f'Batch {batch_num} | Total: {total_progress:,} / {expected_total:,}')
            print(f'Chunk {chunk_idx} | Total so far: {total_progress:,}')

    if not release_test_set or not train_terms:
        return

    # get number of proteins with a Function summary
    data = pd.read_csv(output_file, sep='\t', compression='gzip')
    data.columns = ['Entry', 'Organism', 'Organism ID', 'Function [CC]'] # based on the fields specified in the URL
    data = data.drop_duplicates()
    test_set = pd.read_csv(release_test_set, sep="\t", header=None)
    train_set = pd.read_csv(train_terms, sep="\t", header=0)
    new_proteins = set(data['Entry']) - set(test_set.iloc[:,0]) # 661 new proteins added to SwissProt; some proteins also changed their taxonomy ID (can't download by taxon anymore?, missing 333 entries with taxon change, 7 deleted)
    data = data[~data['Entry'].isin(new_proteins)]
    print(f"Number of proteins with a Function summary: {data['Function [CC]'].notna().sum()} / {len(data)}")
    missing = set.intersection(set(data[data['Function [CC]'].isna()]['Entry']), set(train_set.iloc[:,0]))
    print(f"Number of proteins without a Function summary: {data['Function [CC]'].isna().sum()} / {len(data)}, with {len(missing)} experimentally annotated")
    if missing_output:
        data_missing = data[data['Entry'].isin(missing)]
        data_missing.to_csv(missing_output, sep='\t', index=False, header=True)

def get_taxon_id_from_file(taxon_path):
    """
    Read taxon IDs from the first column of a TSV file.
    """
    taxon_file =  pd.read_csv(taxon_path, header=0, sep='\t', encoding='ISO-8859-1')
    selected_taxon = [str(taxon_id) for taxon_id in set(taxon_file.iloc[:,0].tolist())]

    return selected_taxon


def parse_args(args):
    parser = argparse.ArgumentParser(description='Download FASTA sequences from UniProt for a given taxon ID')
    parser.add_argument('--taxon', '-t', required=True,
                        help='Path to the file with taxon IDs')

    parser.add_argument('--output', '-o', required=True,
                        help='Output file path for the downloaded FASTA sequences')

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    download_fasta_by_taxon(args.taxon, args.output)

if __name__ == "__main__":
    main()
