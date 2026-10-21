#!/usr/bin/env python3
"""Find source-derived annotations (recently experimentally supported) from a GAF file, and remove
them from the t1 annotations

Input: Path to a GAF file (plain text or .gz) and an annotation DataFrame (in retrieve_terms.py)
Output: TSV file with source-derived duplicate annotations removed from the annotation dataframe, 
to be used as the new ground truth set for evaluation.

Finds annotations from sources such as UniProtKB-SubCell and RHEA that were
originally IEA annotations and were added as experimental annotations in the
latest UniProtKB release (Apr/May 2026). Removes them from the annotation
dataframe, as they are not really new annotations.
"""
import sys
import gzip
import argparse
from pathlib import Path
import pandas as pd
from Bio.UniProt import GOA
from democafa.utils.constants import GO_CODES

DEFAULT_GAIN_PAIRS_FILE = 'data/cafa6/for_pub/raw_gained_leaves_Mar2026_May2026.tsv'
DEFAULT_DUPLICATE_SOURCE_RULES = (
    {
        'source': 'UniProtKB-SubCell',
        'aspect': 'C',
        'duplicate_class': 'uniprotkb_subcell_iea',
    },
    {
        'source': 'RHEA',
        'aspect': 'F',
        'duplicate_class': 'rhea_iea',
    },
)


def parse_inputs(argv):
    parser = argparse.ArgumentParser(description="Find source-derived annotations (recently experimentally supported) from a GAF file, and remove them from the t1 propagated annotations")
    parser.add_argument("gaf_file", help="Path to the input GAF file (plain text or .gz)")
    parser.add_argument("annotation_df", help="Path to the input annotation DataFrame (after retrieve_terms.py)")
    return parser.parse_args(argv)

def _gaf_field_contains(field, text):
    if isinstance(field, str):
        return text in field
    return any(text in str(value) for value in field)


def _format_gaf_field(field):
    if isinstance(field, str):
        return field
    return '|'.join(str(value) for value in field)


def find_subcell_experimental_duplicate_annotations(
    gaf_file,
    duplicate_source_rules=DEFAULT_DUPLICATE_SOURCE_RULES,
    gain_pairs_file=DEFAULT_GAIN_PAIRS_FILE,
):
    """
    Find protein-GO pairs annotated both by source-derived IEA and experiment.
    
    Args:
        gaf_file (str): Path to a GAF file. Plain text and .gz files are supported.
        duplicate_source_rules (tuple): Source/aspect rules for annotations that
            were originally IEA and later gained experimental evidence.
        gain_pairs_file (str): TSV containing comparison rows with EntryID, term,
            and difference_type columns. Only file2_only pairs are checked.
        experimental_codes (set, optional): Experimental evidence codes. Defaults
            to GO_CODES['EXPERIMENTAL'] (case-insensitive key lookup).
    
    Returns:
        pandas.DataFrame: Matching rows for pairs that have both annotation
        classes. Columns include EntryID, term, aspect, evidence, with_from, and
        duplicate_class.
    """
    
    experimental_codes = set(GO_CODES['EXPERIMENTAL'])
    rules_by_aspect = {
        rule['aspect']: rule
        for rule in duplicate_source_rules
    }
    target_aspects = set(rules_by_aspect)
    
    gain_pairs = pd.read_csv(
        gain_pairs_file,
        sep="\t",
        header=0,
        usecols=['difference_type', 'EntryID', 'term', 'aspect'],
        dtype=str,
    )
    gain_pairs = gain_pairs.apply(lambda col: col.str.strip())
    gain_pairs = gain_pairs[
        (gain_pairs['difference_type'] == "file2_only")
        & (gain_pairs['aspect'].isin(target_aspects))
    ][['EntryID', 'term']]
    gain_tuples = set(zip(gain_pairs['EntryID'], gain_pairs['term']))
    if not gain_tuples:
        print("No gained protein-GO pairs found to check for duplicate source annotations.")
        return pd.DataFrame()
    
    pair_annotations = {}
    is_gzipped = gaf_file.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    
    with open_func(gaf_file, mode) as handle:
        for rec in GOA.gafiterator(handle):
            if 'NOT' in rec['Qualifier']:
                continue
            if rec['DB'] != 'UniProtKB':
                continue
            aspect = rec['Aspect']
            rule = rules_by_aspect.get(aspect)
            if rule is None:
                continue
                
            key = (rec['DB_Object_ID'], rec['GO_ID'])
            if key not in gain_tuples:
                continue
            evidence = rec['Evidence']
            raw_with_from = rec.get('With', '')
            with_from = _format_gaf_field(raw_with_from)
            has_source = _gaf_field_contains(raw_with_from, rule['source'])
            is_source_iea = has_source and evidence == 'IEA'
            is_experimental = evidence in experimental_codes
            
            if not is_source_iea and not is_experimental:
                continue
                
            pair_annotations.setdefault(key, {'source_iea': [], 'experimental': []})
            row = {
                'EntryID': rec['DB_Object_ID'],
                'term': rec['GO_ID'],
                'aspect': aspect,
                'evidence': evidence,
                'with_from': with_from,
                'duplicate_source': rule['source'],
                'date': rec['Date'],
            }
            if is_source_iea:
                pair_annotations[key]['source_iea'].append({
                    **row,
                    'duplicate_class': rule['duplicate_class'],
                })
            if is_experimental:
                pair_annotations[key]['experimental'].append({
                    **row,
                    'duplicate_class': 'experimental',
                })
    
    duplicated_rows = []
    for annotations in pair_annotations.values():
        if annotations['source_iea'] and annotations['experimental']:
            duplicated_rows.extend(annotations['source_iea'])
            duplicated_rows.extend(annotations['experimental'])
    
    df = pd.DataFrame(duplicated_rows)
    if not df.empty:
        df = df.drop_duplicates()
    pair_count = len(set(zip(df['EntryID'], df['term']))) if not df.empty else 0
    sources = ', '.join(rule['source'] for rule in duplicate_source_rules)
    print(f"Found {pair_count} protein-GO pairs with both source-derived IEA ({sources}) and experimental annotations.")
    return df


def remove_subcell_annt_from_annotation_df(
    annotation_df,
    gaf_file,
    gain_pairs_file=DEFAULT_GAIN_PAIRS_FILE,
):
    duplicates_df = find_subcell_experimental_duplicate_annotations(
        gaf_file,
        gain_pairs_file=gain_pairs_file,
    )
    
    if duplicates_df.empty:
        print("No duplicate annotations found. No changes made to the propagated annotations.")
        filtered_df = annotation_df.copy()
    else:
        df = annotation_df.copy()
        # Create a set of (EntryID, term) pairs to remove
        pairs_to_remove = set(zip(duplicates_df['EntryID'], duplicates_df['term']))
        # Filter out rows from prop_df that match any of the pairs to remove
        filtered_df = df[
            ~pd.MultiIndex.from_frame(df[['EntryID', 'term']]).isin(pairs_to_remove)
        ]
        # filtered_prop_df.to_csv(output_tsv_file, sep='\t', index=False)
    return filtered_df
    
    
if __name__ == "__main__":
    args = parse_inputs(sys.argv[1:])
    remove_subcell_annt_from_annotation_df(args.annotation_df, args.gaf_file)
    
