#!/usr/bin/env python3
"""Normalize raw holdout ground-truth annotations into democafa terms TSVs.

The expected input is a TSV with ``ENTITY_ID``, ``GO_ID``, and ``GO_EVIDENCE``
columns. The output contains ``EntryID``, ``term``, and ``aspect`` after
evidence-code filtering, isoform collapsing, obsolete-term removal, and removal
of molecular-function annotations for proteins that only gained protein-binding
terms.
"""

import sys
import argparse
import os
import gzip
import pandas as pd
import re
from Bio import SeqIO
import requests
import obonet
import networkx as nx
import time
from collections import Counter
from democafa.utils.ontology import add_aspect_column, clean_ontology_edges, fetch_aspect
from democafa.datacollection.retrieve_terms import filter_evidence_codes
from democafa.utils.constants import GO_CODES
from Bio import Entrez
Entrez.email = "ahphan@iastate.edu"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

def process_tsv(input_file, ontology_graph):
    """
    Process a TSV file to add an 'aspect' column based on GO terms.
    Specifically designed for the CAFA holdout ground truth, which is provided by UniProt in a TSV format 

    Args:
        input_file: Path to the input TSV file.
        ontology_graph: Path to the OBO file used to assign GO aspects.
    """
    ontology_graph = clean_ontology_edges(obonet.read_obo(ontology_graph))
    roots = {'P': 'GO:0008150', 'C': 'GO:0005575', 'F': 'GO:0003674'}
    subontologies = {aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots}

    annotation_df = pd.read_csv(input_file, sep='\t', header=0)
    required_columns = {'ENTITY_ID', 'GO_ID', 'GO_EVIDENCE'}
    missing_columns = required_columns - set(annotation_df.columns)
    if missing_columns:
        raise ValueError(f"{input_file} is missing required columns: {sorted(missing_columns)}")
    annotation_df = annotation_df[['ENTITY_ID', 'GO_ID', 'GO_EVIDENCE']]
    annotation_df.columns = ['EntryID', 'term', 'evidence']
    annotation_df = add_aspect_column(annotation_df, subontologies)

    # Filter for EXPERIMENTAL,IC,TAS
    selected_codes = filter_evidence_codes(GO_CODES, selected='Experimental,IC,TAS').get('Evidence')
    annotation_df = annotation_df[annotation_df['evidence'].isin(selected_codes)]

    obsolete_terms = set(annotation_df['term']) - set(ontology_graph.nodes())
    if obsolete_terms:
        print(f"Warning: {len(obsolete_terms)} obsolete (or new) terms ({obsolete_terms}) found in annotation file but not in graph.")
        print("These terms will not appear in terms file.")
        annotation_df = annotation_df[~annotation_df['term'].isin(obsolete_terms)]

    # Union-ize isoform annotations
    annotation_df['EntryID'] = annotation_df['EntryID'].str.split('-').str[0]
    annotation_df = annotation_df.drop_duplicates(subset=['EntryID', 'term', 'aspect'])

    print(annotation_df.describe())

    # Remove 3 binding terms of proteins even though they are annotated in other aspects
    binding_terms = set(nx.descendants(subontologies['F'],'GO:0005515'))
    binding_terms.add('GO:0005515')

    annotated_F_proteins = set(annotation_df[annotation_df['aspect']=='F']['EntryID'])
    not_binding_only_proteins = set(annotation_df[(annotation_df['EntryID'].isin(annotated_F_proteins)) &
                                                  (~annotation_df['term'].isin(binding_terms)) &
                                                  (annotation_df['aspect']=='F')]['EntryID']) # these proteins have other MFO annotations
    binding_df = annotation_df[(annotation_df['EntryID'].isin(annotated_F_proteins - not_binding_only_proteins)) &
                               (annotation_df['aspect']=='F')] # these annotations are protein-binding only
    # Remove rows in binding_df from annotation_df
    annotation_df = annotation_df[~annotation_df.index.isin(binding_df.index)]

    # # Remove TrEMBL proteins
    # fasta_proteins = read_fasta_proteins(swissprot_fasta)
    # trembl_gain_proteins = set.difference(set(annotation_df['EntryID']), set(fasta_proteins.keys()))
    # print(f"Removed {len(trembl_gain_proteins)} TrEMBL proteins: {trembl_gain_proteins}.")
    # annotation_df = annotation_df[~annotation_df['EntryID'].isin(trembl_gain_proteins)]

    print(annotation_df['aspect'].value_counts())
    print(f"Number of unique proteins: {len(annotation_df['EntryID'].unique())}")

    batch = list(annotation_df['EntryID'].unique())
    return annotation_df


def fetch_taxonomy(batch):
    """Fetch UniProt review status and taxonomy metadata for accessions."""
    url = "https://rest.uniprot.org/idmapping/run"
    taxon = dict()
    data = {
            'from': 'UniProtKB_AC-ID',
            'to': 'UniProtKB',
            'ids': f"{','.join(batch)}"
        }
    review_status = dict()
    # Submit job
    response = requests.post(url, data=data)
    job_id = response.json()['jobId']
    results_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        status_response = requests.get(results_url)
        status = status_response.json()
        if 'results' in status:
            # Download results
            stream_url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job_id}?fields=accession%2Creviewed%2Corganism_name%2Corganism_id&format=tsv"
            tsv_response = requests.get(stream_url)
            tsv_content = tsv_response.text

            for line in tsv_content.strip().split('\n')[1:]:
                accession, _, reviewed, organism_name, taxid = line.split('\t')
                review_status[accession] = reviewed
                taxon[accession] = (organism_name, taxid)
            break
        time.sleep(0.5)

    return taxon


def fetch_taxon_lineage(taxon_id, output_tsv=None, api_key=None, timeout=60):
    """Fetch NCBI taxonomy lineage using ncbi-datasets-cli.

    This runs the equivalent of:

    datasets summary taxonomy taxon <taxon_id> --api-key <api_key> --as-json-lines \
        | dataformat tsv taxonomy --template tax-summary

    Args:
        taxon_id: NCBI taxonomy identifier.
        output_tsv: Optional path to write the ``tax-summary`` TSV.
        api_key: Optional NCBI API key. Defaults to the ``NCBI_API_KEY``
            environment variable when it is set.
        timeout: Maximum seconds to wait for each CLI command.

    Returns:
        A dictionary with the parsed tax-summary row, ranked lineage IDs, and a
        UniProt-style ``Taxonomic lineage (Ids)`` string.
    """
    import subprocess
    import shutil
    import csv
    import io

    taxon_id = str(taxon_id)
    selected_api_key = api_key if api_key is not None else NCBI_API_KEY
    python_bin_dir = os.path.dirname(sys.executable)
    datasets_exe = shutil.which('datasets') or os.path.join(python_bin_dir, 'datasets')
    dataformat_exe = shutil.which('dataformat') or os.path.join(python_bin_dir, 'dataformat')

    def build_datasets_cmd(key=None):
        cmd = [
            datasets_exe,
            'summary',
            'taxonomy',
            'taxon',
            taxon_id,
        ]
        if key:
            cmd.extend(['--api-key', key])
        cmd.append('--as-json-lines')
        return cmd

    def datasets_env(use_api_key):
        if use_api_key:
            return None
        env = os.environ.copy()
        env.pop('NCBI_API_KEY', None)
        return env

    datasets_cmd = build_datasets_cmd(selected_api_key)
    dataformat_cmd = [
        dataformat_exe,
        'tsv',
        'taxonomy',
        '--template',
        'tax-summary',
    ]

    try:
        datasets_result = subprocess.run(
            datasets_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
            env=datasets_env(bool(selected_api_key)),
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not find the NCBI datasets CLI. Activate the micromamba "
            "environment that provides both 'datasets' and 'dataformat'."
        ) from None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        retry_without_key = (
            selected_api_key
            and api_key is None
            and any(status in stderr for status in ("400", "401", "403"))
        )
        if not retry_without_key:
            raise RuntimeError(
                f"Error fetching lineage for taxon ID {taxon_id}: {stderr}"
            ) from None

        datasets_cmd = build_datasets_cmd()
        try:
            datasets_result = subprocess.run(
                datasets_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout,
                env=datasets_env(False),
            )
        except subprocess.CalledProcessError as retry_error:
            retry_stderr = retry_error.stderr.strip() if retry_error.stderr else ""
            raise RuntimeError(
                "Error fetching lineage for taxon ID "
                f"{taxon_id} with NCBI_API_KEY ({stderr}) and without an API "
                f"key ({retry_stderr})."
            ) from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Timed out fetching lineage for taxon ID {taxon_id} without "
                f"an API key after {timeout} seconds."
            ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timed out fetching lineage for taxon ID {taxon_id} after "
            f"{timeout} seconds."
        ) from None

    try:
        dataformat_result = subprocess.run(
            dataformat_cmd,
            input=datasets_result.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not find 'dataformat'. Activate the micromamba environment "
            "that provides the NCBI datasets CLI."
        ) from None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(
            f"Error formatting lineage for taxon ID {taxon_id}: {stderr}"
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timed out formatting lineage for taxon ID {taxon_id} after "
            f"{timeout} seconds."
        ) from None

    tsv_output = dataformat_result.stdout
    if output_tsv:
        output_dir = os.path.dirname(output_tsv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_tsv, 'w') as f:
            f.write(tsv_output)

    rows = list(csv.DictReader(io.StringIO(tsv_output), delimiter='\t'))
    if not rows:
        raise RuntimeError(f"No taxonomy lineage returned for taxon ID {taxon_id}.")

    tax_summary = rows[0]
    rank_columns = [
        ('Domain/Realm taxid', 'domain'),
        ('Kingdom taxid', 'kingdom'),
        ('Phylum taxid', 'phylum'),
        ('Class taxid', 'class'),
        ('Order taxid', 'order'),
        ('Family taxid', 'family'),
        ('Genus taxid', 'genus'),
        ('Species taxid', 'species'),
    ]
    lineage = []
    seen_taxids = set()
    for column, rank in rank_columns:
        lineage_taxid = tax_summary.get(column, '').strip()
        if lineage_taxid and lineage_taxid not in seen_taxids:
            lineage.append((lineage_taxid, rank))
            seen_taxids.add(lineage_taxid)

    query_taxid = tax_summary.get('Taxid', '').strip()
    query_rank = tax_summary.get('Rank', '').strip().lower()
    if query_taxid and query_taxid not in seen_taxids:
        lineage.append((query_taxid, query_rank or 'no rank'))

    return {
        'tax_summary': tax_summary,
        'lineage_ids': [lineage_taxid for lineage_taxid, _ in lineage],
        'Taxonomic lineage (Ids)': ', '.join(
            f"{lineage_taxid} ({rank})" for lineage_taxid, rank in lineage
        ),
    }


def read_fasta_proteins(fasta_file: str, entries):
    """
    Read proteins from FASTA file using Biopython.
    Return taxonomy IDs only for requested accessions.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        Dictionary mapping protein IDs to their sequences
    """
    tax_pattern = re.compile(r"OX=(\d+)")
    species_pattern = re.compile(r"OS=([^O]+)")
    fasta_proteins = {}
    # species = {}


    # Process gzipped fasta file to get all SwissProt proteins
    with gzip.open(fasta_file, "rt") as gz_file:
        for record in SeqIO.parse(gz_file, 'fasta'):
            # Extract accession (EntryID)
            entry_id = record.id.split("|")[1] if "|" in record.id else record.id
            if entry_id not in entries:
                continue
            # Extract taxonomy ID using regex
            tax_match = tax_pattern.search(record.description)
            tax_id = tax_match.group(1) if tax_match else "N/A"

            # species_match = species_pattern.search(record.description)
            # species_name = species_match.group(1) if species_match else "N/A"
            # if selected_taxon is None or tax_id in selected_taxon:
                # seq_count += 1
            fasta_proteins[entry_id] = tax_id
            # elif tax_id not in selected_taxon:
                # missing_taxon[entry_id] = tax_id
            # if species_name != "N/A":
                # species[species_name] = tax_id

    return fasta_proteins


def fetch_eukaryote_taxa():
    """Fetch eukaryote taxonomy data from UniProt"""
    url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Corganism_name%2Corganism_id&format=tsv&query=%28%28taxonomy_id%3A2759%29+AND+%28reviewed%3Atrue%29%29"

    response = requests.get(url)
    response.raise_for_status()

    # Decompress the content
    decompressed_data = gzip.decompress(response.content).decode('utf-8')

    # Parse into lines
    lines = decompressed_data.strip().split('\n')

    # Extract taxonomy IDs
    eukaryote_taxa_ids = set()
    for line in lines[1:]:  # Skip header
        if line.strip():
            fields = line.split('\t')
            if len(fields) >= 3:
                tax_id = fields[2]  # Organism ID column
                eukaryote_taxa_ids.add(tax_id)

    return eukaryote_taxa_ids


def add_additional_taxon(old_taxon_path, new_taxon_path, holdout_groundtruth, fasta_file):
    """
    Compare old taxon list with new taxon list and add any additional taxon found in holdout ground truth.

    Args:
        old_taxon_path: Path to the old taxon list file (CAFA5).
        new_taxon_path: Path to the new taxon list file (CAFA6).
        holdout_groundtruth: Path to the processed holdout ground truth file.
        fasta_file: Gzipped FASTA containing holdout proteins.
    """
    groundtruth_df = pd.read_csv(holdout_groundtruth, sep='\t', header=0)
    gained_entries = set(groundtruth_df.iloc[:,0].tolist())
    print(f"Holdout ground truth has {len(gained_entries)} entries.")
    # gained_taxon = fetch_taxonomy(gained_entries)  # API lookup is intentionally skipped for large batches.
    gained_protein_taxon = read_fasta_proteins(fasta_file, gained_entries)
    gained_taxon_counts = {id:count for id,count in Counter(gained_protein_taxon.values()).items()}
    print(gained_taxon_counts)
    gained_taxon_ids = set(gained_taxon_counts.keys())

    # species of interest in CAFA5: 90 species
    taxon_file =  pd.read_csv(old_taxon_path, header=0, sep='\t', encoding='ISO-8859-1')
    old_taxon = [str(taxon_id) for taxon_id in set(taxon_file.iloc[:,0].tolist())]
    print(f"Old taxon list has {len(old_taxon)} entries.")

    # CAFA6: all Eukaryotes (id=2759)
    taxon_new_set = set()
    eukaryote_taxa = fetch_eukaryote_taxa()

    old_prokaryote_taxa = set(old_taxon) - eukaryote_taxa
    if old_prokaryote_taxa:
        print(f"Adding back {len(old_prokaryote_taxa)} prokaryote taxa from old taxon list.")
        taxon_new_set.update(old_prokaryote_taxa)

    new_prokaryote_taxa = (set(gained_taxon_ids) - eukaryote_taxa) - old_prokaryote_taxa
    if new_prokaryote_taxa:
        print(f"Adding {len(new_prokaryote_taxa)} prokaryote taxa from holdout ground truth.")
        taxon_new_set.update(new_prokaryote_taxa)

    # write to new taxon file
    with open(new_taxon_path,'w') as f:
        f.write("ID\tSpecies\n")
        f.write("2759\tEukaryota\n") # write Eukaryota first
        for taxid in taxon_new_set:
            s = Entrez.esummary(db="taxonomy", id=taxid, retmode="xml")
            result = Entrez.read(s)
            print(taxid, result[0]['ScientificName'])
            f.write(f"{taxid}\t{result[0]['ScientificName']}\n")
            taxon_new_set.add(taxid)
    print(f"New taxon list has {len(taxon_new_set)} entries.")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Process ground truth data for CAFA challenge.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input TSV file with GO annotations.')
    parser.add_argument('--graph', type=str, required=True, help='Path to the ontology OBO file.')
    parser.add_argument('--output_terms', type=str, required=True, help='Path to the output TSV file with processed terms.')
    parser.add_argument('--fasta_file', type=str, required=False, help='Optional FASTA file for taxon-list expansion workflows.')

    return parser.parse_args(argv)


def process_ground_truth(input_file, graph, output_terms):
    """Process raw ground truth and write ``EntryID``, ``term``, ``aspect`` TSV."""
    annotation_df = process_tsv(input_file, graph)
    output_dir = os.path.dirname(output_terms)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    annotation_df[['EntryID', 'term', 'aspect']].to_csv(output_terms, sep='\t', index=False, header=True)


def main():
    args = parse_args(sys.argv[1:])
    process_ground_truth(args.input_file, args.graph, args.output_terms)
    # add_additional_taxon('data/raw/testsuperset-taxon-list.tsv',
    #                      'data/cafa6/raw/selected-taxon-list.tsv',
    #                      args.output_terms,
    #                      args.fasta_file)

if __name__ == "__main__":
    main()
