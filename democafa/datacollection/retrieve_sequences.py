#!/usr/bin/env python3

"""Extract selected protein sequences from a FASTA file.

The input FASTA may be plain text or gzipped. The selection file is a plain-text
list of UniProt accessions, one per line. FASTA identifiers in UniProt pipe
format (for example ``sp|P12345|NAME``) are matched by accession.
"""

import gzip
import os
from Bio import SeqIO
import sys
import argparse


def retrieve_sequences(fasta, input, out_fasta):
    """
    Retrieve sequences from a FASTA file based on selected entries in a .txt file.

    Args:
        fasta: Path to the input FASTA file (can be gzipped).
        input: Path to a text file containing protein IDs (one per line).
        out_fasta: Path to the output FASTA file with selected proteins.
    """
    print(f"Retrieving sequences from {fasta} based on entries in {input}...")

    # Read protein IDs from input file
    with open(input, 'r') as f:
        selected_ids = set(line.strip() for line in f if line.strip())

    # Open output FASTA file
    output_dir = os.path.dirname(out_fasta)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(out_fasta, 'w') as out_handle:
        # Process the FASTA file
        with gzip.open(fasta, 'rt') if fasta.endswith('.gz') else open(fasta, 'r') as fasta_handle:
            for record in SeqIO.parse(fasta_handle, 'fasta'):
                entry_id = record.id.split("|")[1] if "|" in record.id else record.id
                if entry_id in selected_ids:
                    SeqIO.write(record, out_handle, 'fasta')

    print(f"Sequences written to {out_fasta}")


def parse_inputs(argv):
    parser = argparse.ArgumentParser(
        description='Retrieve sequences from UniProt FASTA file based on selected entries in a .txt file')

    parser.add_argument('--fasta', '-f', required=True,
                        help='Path to fasta file (can be gzipped)')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to a text file containing protein IDs (one per line)')
    parser.add_argument('--out_fasta', '-of', required=True,
                        help='File name for output FASTA file (with selected proteins)')
    return parser.parse_args(argv)

def main():
    args = parse_inputs(sys.argv[1:])
    retrieve_sequences(
        fasta = args.fasta,
        input = args.input,
        out_fasta = args.out_fasta
    )

if __name__ == "__main__":
    main()
