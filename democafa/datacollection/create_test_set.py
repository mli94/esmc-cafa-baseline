#!/usr/bin/env python3

"""Create training sequence files and a CAFA test-superset FASTA.

The module starts from a three-column terms TSV (``EntryID``, ``term``,
``aspect``) and a UniProt FASTA file. SwissProt entries present in the terms
file become training sequences. Annotated proteins absent from the FASTA are
treated as TrEMBL accessions and downloaded from UniProt in batches.

When ``include_all`` is true, the test superset contains all selected FASTA
proteins plus downloaded TrEMBL sequences. Otherwise, it contains only proteins
that are missing one or more GO aspects.
"""

import time
import os
import sys
import requests
import argparse
import re
import gzip
import logging
from datetime import datetime
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Create a specific logger for this module (not the root logger)
logger = logging.getLogger('create_test_set')
logger.setLevel(logging.INFO)

# Prevent messages from propagating to the root logger (so multiple loggers can coexist)
logger.propagate = False

# Create file handler
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
log_filename = os.path.join(log_dir, datetime.now().strftime('create_test_set_%Y%m%d_%H%M%S.log'))
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_proteins_with_all_aspects(terms_file):
    """
    Get proteins that have all three GO aspects (molecular_function,
    biological_process, cellular_component).

    Args:
        terms_file: TSV with columns ``EntryID``, ``term``, and ``aspect``.

    Returns:
        Set of protein IDs with all three aspects
    """
    df = pd.read_csv(terms_file, sep='\t')
    required_columns = {'EntryID', 'aspect'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{terms_file} is missing required columns: {sorted(missing_columns)}")
    # Group by protein ID and get unique aspects for each protein
    protein_aspects = df.groupby('EntryID')['aspect'].unique()
    all_proteins = set(df['EntryID'].unique())

    # Find proteins with all three aspects
    complete_proteins = {
        protein for protein, aspects in protein_aspects.items() if len(aspects) == 3
    }

    return all_proteins, complete_proteins


def read_fasta_proteins(fasta_file: str, taxon):
    """
    Read proteins from FASTA file using Biopython.
    Optionally filter by one taxon ID or a TSV of taxon IDs.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        Dictionary mapping protein IDs to their sequences
    """
    # Read in taxon file
    if taxon is None:
        selected_taxon = None
    elif taxon.isdigit():
        # If taxon is a single ID, convert it to a list
        selected_taxon = [str(taxon)]
    elif isinstance(taxon, str) and os.path.exists(taxon):
        taxon_file = pd.read_csv(taxon, header=0, sep='\t', encoding='ISO-8859-1')
        selected_taxon = [str(taxon_id) for taxon_id in set(taxon_file.iloc[:,0].tolist())]
    else:
        raise ValueError(f"Taxon must be a numeric ID, an existing TSV file, or None: {taxon}")
    logger.info(f"Selected taxon: {selected_taxon}")

    tax_pattern = re.compile(r"OX=(\d+)")
    species_pattern = re.compile(r"OS=([^=]+?)\s+OX=")  # Match species name between OS= and OX=
    fasta_proteins = {}
    # species = {}

    seq_count = 0
    tax_dict = {}
    # Process gzipped fasta file to get all SwissProt proteins
    with gzip.open(fasta_file, "rt") as gz_file:
        for record in SeqIO.parse(gz_file, 'fasta'):
            # Extract accession (EntryID)
            entry_id = record.id.split("|")[1] if "|" in record.id else record.id

            # Extract taxonomy ID using regex
            tax_match = tax_pattern.search(record.description)
            tax_id = tax_match.group(1) if tax_match else "N/A"
            species_name = "N/A"
            if tax_id != "N/A":
                species_match = species_pattern.search(record.description)
                species_name = species_match.group(1) if species_match else "N/A"
                tax_dict[tax_id] = species_name

            if selected_taxon is None or tax_id in selected_taxon:
                seq_count += 1
                fasta_proteins[entry_id] = [str(record.id), str(record.description), str(record.seq), tax_id, species_name]
            # if species_name != "N/A":
                # species[species_name] = tax_id
    logger.info(f"Total sequences selected: {seq_count}")

    return fasta_proteins


def write_missing_aspects_fasta(fasta_proteins, complete_proteins, trembl_sequences, output_file):
    """
    Write FASTA file containing proteins missing GO aspects.

    Args:
        fasta_proteins: Dictionary of protein sequences
        complete_proteins: Set of proteins with all aspects
        trembl_sequences: Dictionary returned by ``batch_download_trembl_sequences_post``.
        output_file: Path to output FASTA file
    """
    # Get proteins missing aspects
    missing_aspects = set(fasta_proteins.keys()) - complete_proteins
    trembl_missing_aspects = set(trembl_sequences.keys()) - complete_proteins
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as fasta_out:
        for protein_id in missing_aspects:
            seq_record = SeqRecord(Seq(fasta_proteins[protein_id][2]), id=fasta_proteins[protein_id][0], description=fasta_proteins[protein_id][1])
            # values = fasta_proteins[protein_id]
            # f.write(f'>{protein_id}\t{values[1]}\n{values[0]}\n')
            SeqIO.write(seq_record, fasta_out, "fasta")

    with open(output_file, 'a') as fasta_out:
        for protein_id in trembl_missing_aspects:
            (header, sequence) = trembl_sequences[protein_id]
            seq_record = SeqRecord(Seq(sequence), id=header, description="")
            SeqIO.write(seq_record, fasta_out, "fasta")
    # for protein_id in trembl_proteins:
    #     header, sequence = get_fasta_from_API(protein_id, uniprot_api_version)

    #     if header and sequence:
    #         seq_record = SeqRecord(Seq(sequence), id=header, description="")
    #         with open(output_file, "a") as handle:
    #             SeqIO.write(seq_record, handle, "fasta")
    #     else:
    #         print(f"Release version {uniprot_api_version} not found.")


def write_fasta(fasta_proteins, trembl_sequences, annotated_proteins_to_include, output_file, num_queries, out_taxonomy, description=True, uniprot_api_version=None):
    """
    Write FASTA file all proteins with or without missing GO aspects (for partial knowledge evaluation).

    Args:
        fasta_proteins: Dictionary of protein sequences
        trembl_sequences: Dictionary of downloaded TrEMBL sequences
        uniprot_api_version: UniProt release version to retrieve TrEMBL sequences
        output_file: Path to output FASTA file
        num_queries: Number of queries to include in the test superset (for test data)
        out_taxonomy: Optional path to write selected taxonomy IDs and species.
        description: Whether to include full FASTA descriptions or just accession and taxonomy ID
    """
    selected_taxon = dict()
    for output_path in (output_file, out_taxonomy):
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
    if description:
        logger.info("Including full FASTA descriptions.")

    def write_swissprot_record(fasta_out, protein_id):
        if description:
            seq_record = SeqRecord(
                Seq(fasta_proteins[protein_id][2]),
                id=fasta_proteins[protein_id][0],
                description=fasta_proteins[protein_id][1]
            )
        else:
            seq_record = SeqRecord(
                Seq(fasta_proteins[protein_id][2]),
                id=f'{fasta_proteins[protein_id][0].split("|")[1]} {fasta_proteins[protein_id][3]}',
                description=""
            )
        SeqIO.write(seq_record, fasta_out, "fasta")
        selected_taxon[fasta_proteins[protein_id][3]] = fasta_proteins[protein_id][4]

    if num_queries is None:
        with open(output_file, 'w') as fasta_out:
            for protein_id in fasta_proteins:
                write_swissprot_record(fasta_out, protein_id)
        num_selected = len(fasta_proteins)
    else:
        num_queries = int(num_queries)
        count=0
        with open(output_file, 'w') as fasta_out:
            for protein_id in annotated_proteins_to_include:
                if protein_id not in fasta_proteins:
                    continue
                if count >= num_queries:
                    break
                write_swissprot_record(fasta_out, protein_id)
                count += 1
            for protein_id in fasta_proteins:
                if count >= num_queries:
                    break
                if protein_id in annotated_proteins_to_include:
                    continue
                write_swissprot_record(fasta_out, protein_id)
                count += 1
        num_selected = count
    # For fetching through UniProt website
    # with open(output_file.replace('.fasta', '_trembl.txt'), 'w') as trembl_out:
    #     for protein_id in trembl_proteins:
    #         trembl_out.write(f'{protein_id}\n')

    with open(output_file, 'a') as fasta_out:
        for protein_id, (header, sequence) in trembl_sequences.items():
            seq_record = SeqRecord(Seq(sequence), id=header, description="")
            SeqIO.write(seq_record, fasta_out, "fasta")
    logger.info(f"Total sequences in test superset: {num_selected + len(trembl_sequences)}")

    if out_taxonomy:
        with open(out_taxonomy, 'w') as tax_out:
            tax_out.write('ID\tSpecies\n')
            for tax_id, species_name in selected_taxon.items():
                tax_out.write(f'{tax_id}\t{species_name}\n')
        logger.info(f"Total taxa in test superset: {len(selected_taxon)}")
    # print(f"Appending {len(trembl_proteins)} TrEMBL proteins to {output_file}...")
    # if uniprot_api_version is None:
    #     for protein_id in trembl_proteins:
    #         # header, sequence = get_fasta_from_API(protein_id, uniprot_api_version)
    #         header, sequence = get_latest_fasta_from_API(protein_id)

    #         if header and sequence:
    #             seq_record = SeqRecord(Seq(sequence), id=header, description="")
    #             with open(output_file, "a") as handle:
    #                 SeqIO.write(seq_record, handle, "fasta")

# s = timeit.timeit(lambda: get_latest_fasta_from_API('Q04681'), number=1) # 0.44s
# s2 = timeit.timeit(lambda: get_fasta_from_API('Q04681', '2025_02'), number=1) # 1.2s

def batch_download_trembl_sequences_post(protein_ids, batch_size=50000):
    """
    Download UniProt FASTA records for accessions that are not in SwissProt.
    """
    sequences = {}

    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num} ({len(batch)} proteins)...")

        # Use POST for larger batches
        url = "https://rest.uniprot.org/idmapping/run"

        data = {
            'from': 'UniProtKB_AC-ID',
            'to': 'UniProtKB',
            'ids': f"{','.join(batch)}"
        }

        # Submit job
        response = requests.post(url, data=data)
        response.raise_for_status()
        job_id = response.json()['jobId']

        # Poll for results
        results_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"

        while True:
            status_response = requests.get(results_url)
            status_response.raise_for_status()
            status = status_response.json()

            if 'results' in status:
                # Download results
                fasta_url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job_id}?format=fasta"
                fasta_response = requests.get(fasta_url)
                fasta_response.raise_for_status()

                batch_sequences = parse_fasta_content(fasta_response.text)
                sequences.update(batch_sequences)
                break

            if status.get('jobStatus') in {'FAILED', 'ERROR'}:
                raise RuntimeError(f"UniProt ID mapping failed for batch {batch_num}: {status}")

            time.sleep(1)  # Wait before polling again

    return sequences


def parse_fasta_content(fasta_content):
    """Parse FASTA content and return dictionary of sequences."""
    sequences = {}
    current_header = None
    current_seq = []

    for line in fasta_content.strip().split('\n'):
        if line.startswith('>'):
            if current_header:
                # Extract accession from header
                accession = current_header.split('|')[1] if '|' in current_header else current_header.split()[0]
                sequences[accession] = (current_header, ''.join(current_seq))
            current_header = line[1:]  # Remove '>'
            current_seq = []
        else:
            current_seq.append(line)

    # Don't forget the last sequence
    if current_header:
        accession = current_header.split('|')[1] if '|' in current_header else current_header.split()[0]
        sequences[accession] = (current_header, ''.join(current_seq))

    return sequences


def get_fasta_from_API(accession, uniprot_api_version):
    # fetching API too slow, download from UniProt website very fast but point-and-click
    # This function retrieves the FASTA sequence for a given UniProt accession (but doesn't work for the latest version)
    url = f"https://rest.uniprot.org/unisave/{accession}?format=fasta"

    response = requests.get(url)
    response.raise_for_status()
    fasta_content = response.text

    header = fasta_content.splitlines()[0][1:]
    sequence = ""
    found_release = False

    for line in fasta_content.splitlines():
        if line.startswith(">"):
            if re.search(uniprot_api_version, line):
                found_release = True
            elif found_release:
                break
        elif found_release:
            sequence += line

    if sequence:
        return header, sequence
    else:
        return None, None


def get_latest_fasta_from_API(accession):
    """
    Retrieve the latest FASTA sequence for a given UniProt accession.

    Args:
        accession: UniProt accession ID

    Returns:
        Tuple of header and sequence
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    response = requests.get(url)
    response.raise_for_status()
    fasta_content = response.text

    header = fasta_content.splitlines()[0][1:]
    sequence = "".join(fasta_content.splitlines()[1:])

    return header, sequence


def write_species(species, output_file):
    """
    Write text file containing species names and taxonomy IDs.

    Args:
        species: Dictionary of species names and taxonomy IDs
        output_file: Path to output text file
    """
    with open(output_file, 'w') as f:
        f.write('ID\tSpecies\n')
        for name, tax_id in species.items():
            f.write(f'{tax_id}\t{name}\n')


def create_train_sequences(proteins_with_terms, sequences_gzfile, trembl_sequences, train_out_fasta, train_out_taxonomy):
    """
    Process UniProt FASTA file and extract sequences and taxonomy information.
    This file only contains sequences that are in train_terms.tsv (proteins labeled with GO terms).
    """

    # Compile regex pattern for taxonomy ID extraction
    tax_pattern = re.compile(r"OX=(\d+)")

    # Counter for progress tracking
    seq_count = 0
    trembl_seq_count = 0
    all_taxid = set()
    for output_path in (train_out_fasta, train_out_taxonomy):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Process gzipped file and write both outputs simultaneously
    with gzip.open(sequences_gzfile, "rt") as gz_file, \
            open(train_out_fasta, "w") as fasta_out, \
            open(train_out_taxonomy, "w") as mapping_out:

        # Process sequences in SwissProt
        for record in SeqIO.parse(gz_file, "fasta"):
            # Extract accession (EntryID)
            entry_id = record.id.split("|")[1] if "|" in record.id else record.id
            if entry_id not in proteins_with_terms:
                continue
            seq_count += 1

            # Extract taxonomy ID using regex
            tax_match = tax_pattern.search(record.description)
            tax_id = tax_match.group(1) if tax_match else "N/A"
            if tax_id != "N/A":
                all_taxid.add(tax_id)
            mapping_out.write(f"{entry_id}\t{tax_id}\n")

            SeqIO.write(record, fasta_out, "fasta")

            # Print progress every 10 sequences
            if seq_count % 10000 == 0:
                logger.debug(f"Processed {seq_count} sequences in SwissProt...")

        # Process TrEMBL sequences
        if trembl_sequences:
            for protein_id, (header, sequence) in trembl_sequences.items():
                trembl_seq_count += 1

                tax_match = tax_pattern.search(header)
                tax_id = tax_match.group(1) if tax_match else "N/A"
                if tax_id != "N/A":
                    all_taxid.add(tax_id)
                mapping_out.write(f"{protein_id}\t{tax_id}\n")

                seq_record = SeqRecord(Seq(sequence), id=header, description="")
                SeqIO.write(seq_record, fasta_out, "fasta")
            # Print progress every 10 sequences
            if trembl_seq_count % 10000 == 0:
                logger.debug(f"Processed {trembl_seq_count} sequences in TrEMBL...")

    logger.info(f"Total SwissProt sequences in training data: {seq_count}")
    logger.info(f"Total sequences in training data: {trembl_seq_count + seq_count} proteins in {len(all_taxid)} taxa")


def create_test_set(terms_file, sequences_gzfile, train_out_fasta, train_out_taxonomy, include_all, out_fasta=None, test_out_taxonomy=None, num_queries=None, uniprot_api_version=None, in_taxonomy=None):
    """Create train FASTA/taxonomy files and optionally a test-superset FASTA."""
    # Read GO terms data
    logger.info("Reading GO terms data...")
    all_proteins, complete_proteins = get_proteins_with_all_aspects(terms_file)

    # Read FASTA file
    logger.info("Reading FASTA sequences...")
    fasta_proteins = read_fasta_proteins(sequences_gzfile, in_taxonomy)

    # Get TrEMBL proteins with missing aspects
    trembl_proteins = list(all_proteins - set(fasta_proteins.keys()))
    logger.info(f"Found {len(trembl_proteins)} TrEMBL proteins with missing aspects.")

    if trembl_proteins:
        logger.info(f"Batch downloading {len(trembl_proteins)} TrEMBL proteins...")
        trembl_sequences = batch_download_trembl_sequences_post(trembl_proteins)
    else:
        trembl_sequences = {}

    # Create training sequences and taxonomy files
    create_train_sequences(
        proteins_with_terms=all_proteins,
        sequences_gzfile=sequences_gzfile,
        trembl_sequences=trembl_sequences,
        train_out_fasta=train_out_fasta,
        train_out_taxonomy=train_out_taxonomy
    )
    # For test data only (to make sure that there are proteins with non-experimental terms in the test superset):
    if num_queries is not None:
        annotated_proteins_to_include = set.intersection(all_proteins, set(fasta_proteins.keys()))
    else:
        annotated_proteins_to_include = set()

    if out_fasta is None:
        return
    # Append TrEMBL proteins to fasta file of test_superset_all.fasta and train_sequences.fasta
    if include_all:  # including partial knowledge proteins
        write_fasta(fasta_proteins, trembl_sequences, annotated_proteins_to_include, out_fasta, num_queries, test_out_taxonomy, description=False)
    else:  # only include proteins missing aspects (no knowledge and limited knowledge)
        write_missing_aspects_fasta(fasta_proteins, complete_proteins, trembl_sequences, out_fasta)
    # write_species(species, in_taxonomy)


def parse_inputs(args):
    parser = argparse.ArgumentParser(
        description='Create test set from train_terms and train_sequences.'
    )
    parser.add_argument('--terms', '-t', required=True,
                        help='Tab-separated file with UniProtKB accessions and GO terms and GO aspects with header')
    parser.add_argument('--fasta_gz', '-f', required=True,
                        help='Path to gzipped SwissProt FASTA file')
    parser.add_argument('--out_fasta', '-o', required=False,
                        help='Path to test superset FASTA file')
    parser.add_argument('--train_out_fasta', '-tf', required=True,
                        help='Path to training FASTA file with annotated proteins')
    parser.add_argument('--train_out_taxonomy', '-tt', required=True,
                        help='Path to training taxonomy mapping file')
    parser.add_argument('--include_all', action='store_true', default=False,
                        help='Include all proteins in the test superset')
    parser.add_argument('--num_queries', '-n', default=None,
                        help='Number of queries to include in the test superset')
    parser.add_argument('--test_out_taxonomy', required=False,
                        help='Path to test superset taxonomy mapping file')
    parser.add_argument('--uniprot_api', '-u', required=False,
                        help='UniProt API version to retrieve TrEMBL sequences')
    parser.add_argument('--in_taxonomy', required=False, default=None)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set the logging level (default: INFO)')

    args = parser.parse_args(args)

    # Configure logging level based on argument
    logger.setLevel(getattr(logging, args.log_level))
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, args.log_level))

    return args

    # python3 -m democafa.datacollection.create_test_set --terms data/processed/train_terms.tsv -f data/raw/uniprot_sprot.fasta.gz
    # -o data/processed/test_superset_all.fasta --include_all

def main():
    args = parse_inputs(sys.argv[1:])

    logger.info(f"Arguments: {vars(args)}")

    create_test_set(
        terms_file=args.terms,
        sequences_gzfile=args.fasta_gz,
        train_out_fasta=args.train_out_fasta,
        train_out_taxonomy=args.train_out_taxonomy,
        include_all=args.include_all,
        out_fasta=args.out_fasta,
        test_out_taxonomy=args.test_out_taxonomy,
        num_queries=args.num_queries,
        uniprot_api_version=args.uniprot_api,
        in_taxonomy=args.in_taxonomy
    )

if __name__ == "__main__":
    main()
