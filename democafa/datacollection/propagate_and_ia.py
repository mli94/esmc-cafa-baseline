#!/usr/bin/env python3
"""Propagate GO annotations and compute information accretion (IA).

This is the terminal-facing wrapper around :mod:`democafa.utils.ontology`.
Inputs are TSV files with ``EntryID``, ``term``, and optionally ``aspect``.
The propagated TSV keeps the same three-column schema and the IA output is a
two-column term/IA table compatible with CAFA evaluator workflows.
"""

import sys
import argparse
# from democafa.utils.ontology import propagate_and_ia_optimized
from democafa.utils.ia import run

def parse_args(args):
    parser = argparse.ArgumentParser(description='Propagate and Compute Information Accretion of GO annotations')
    parser.add_argument('--terms', '-t', required=True,
                        help='Path to annotation file')

    parser.add_argument('--graph', '-g', required=True,
                        help='Path to the OBO ontology graph used for propagation and IA.')

    parser.add_argument('--tsv_propagated', '-tp', default=None,
                        help='Path to save propagated term counts in 3-column TSV format.')

    parser.add_argument('--matrix_propagated', '-mp', default=None,
                        help='Path to save propagated term counts')

    parser.add_argument('--matrix_indices', '-mi', default=None,
                        help='Path to save protein and term indices')

    parser.add_argument('--output_tsv', '-ot', default='IA.txt',
                        help='Path to save computed IA for each term in the GO (default: IA.txt)')

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    # propagate_and_ia_optimized(
    #     terms_file=args.terms,
    #     graph=args.graph,
    #     tsv_propagated=args.tsv_propagated,
    #     matrix_propagated=args.matrix_propagated,
    #     matrix_indices=args.matrix_indices,
    #     output_tsv=args.output_tsv
    # )
    
    # Using official IA implementation
    run(
        annotation_path=args.terms,
        output_file_path=args.output_tsv,
        propagated_file_path=args.tsv_propagated,
        obo_path=args.graph,
        propagate_annotations=True
    )


if __name__ == "__main__":
    main()
