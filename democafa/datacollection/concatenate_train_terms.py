#!/usr/bin/env python3

"""Concatenate the terms from the different GO annotation sources (UniProt-GOA, SwissProt DAT, etc)
and de-duplicate them to create a single terms file with the same format as the original GOA terms file
"""

import pandas as pd

def concatenate_terms_files(input_files, output_file):
    """
    Concatenate the terms from the different GO annotation sources and de-duplicate them.

    Parameters:
    - input_files: List of input file paths containing GO terms.
    - output_file: Path to the output file where the concatenated and de-duplicated terms will be saved.
    """
    # Read all input files into a list of DataFrames
    dataframes = [pd.read_csv(file, sep='\t', header=0) for file in input_files]

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # De-duplicate the DataFrame based on all columns
    deduplicated_df = concatenated_df.drop_duplicates()

    # Save the de-duplicated DataFrame to the output file
    deduplicated_df.to_csv(output_file, sep='\t', header=True, index=False)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - Namespace containing the parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate and de-duplicate GO terms from multiple sources.")
    parser.add_argument('--input_files', nargs='+', help='List of input files containing GO terms.')
    parser.add_argument('--output_file', required=True, help='Path to the output file for concatenated and de-duplicated terms.')

    return parser.parse_args()

def main():
    args = parse_arguments()
    concatenate_terms_files(args.input_files, args.output_file)

if __name__ == "__main__":
    main()