#!/usr/bin/env python3

"""Classify ground-truth gains into CAFA NK, LK, and PK subsets.

Inputs are known annotations at the public/submission timepoint, annotations at
the later ground-truth timepoint, a FASTA/TXT/TSV query set, and the pivot and
later OBO graphs. Outputs are ``*_NK.tsv``, ``*_LK.tsv``, ``*_PK.tsv``,
``*_PK_known.tsv``, ``*_targets.tsv``, and ``*_terms_of_interest.txt``.
"""

import sys
import os
import obonet
import argparse
import pandas as pd
from Bio import SeqIO
import gzip
import logging
from datetime import datetime
from democafa.utils.ontology import clean_ontology_edges, fetch_aspect

# Create a specific logger for this module (not the root logger)
logger = logging.getLogger('classify_ground_truth')
logger.setLevel(logging.INFO)

# Prevent messages from propagating to the root logger (so multiple loggers can coexist)
logger.propagate = False

def setup_logging(use_file_handler=True, log_level='INFO'):
    """
    Set up logging configuration.

    Args:
        use_file_handler (bool): If True, log to file. If False, log to console.
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set logging level
    logger.setLevel(getattr(logging, log_level))

    if use_file_handler:
        # Create file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
        log_filename = os.path.join(log_dir, datetime.now().strftime('classify_ground_truth_%Y%m%d_%H%M%S.log'))
        handler = logging.FileHandler(log_filename)
    else:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# When imported as a module, set up console logging by default
setup_logging(use_file_handler=False)


def wrapper_ground_truth(annot_known, annot2, query_file, graph, graph2, out_prefix):
    """Classify query proteins by whether new terms appear in known aspects."""
    logger.info("Starting ground truth classification")
    logger.info(f"Output prefix: {out_prefix}")

    # Collect all target proteins
    query_ids = []
    is_gzipped = query_file.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    if '.fasta' in query_file:
        logger.info("Reading query IDs from FASTA file")
        with open_func(query_file, mode) as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                entry_id = record.id.split("|")[1] if "|" in record.id else record.id
                query_ids.append(entry_id)
    elif query_file.endswith(('.txt', '.tsv')):
        logger.info("Reading query IDs from text/TSV file")
        with open(query_file, 'r') as handle:
            query_ids = [line.strip().split('\t')[0] for line in handle if line.strip()]
    else:
        logger.error("Please provide a fasta file or a text file with query IDs")
        sys.exit(1)

    logger.info(f"Number of proteins in query file: {len(query_ids)}")
    dfk = pd.read_csv(annot_known, sep='\t', header=0)
    df2 = pd.read_csv(annot2, sep='\t', header=0)
    required_columns = {'EntryID', 'term', 'aspect'}
    for path, df in ((annot_known, dfk), (annot2, df2)):
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    # During re-propagation with frozen graph, some obsolete terms (at t1) reappeared, so remove them
    # Creating "terms-of-interest" set (use this for GOslim too)
    toi = set.intersection(set(obonet.read_obo(graph).nodes()), set(obonet.read_obo(graph2).nodes()))
    # Roots are excluded using -no_orphans during cafaevaluator
    # remove_terms = {"GO:0003674","GO:0008150","GO:0005575"}
    # logger.info(f"Removing terms: {remove_terms}")
    # toi = toi - remove_terms
    output_dir = os.path.dirname(out_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(f'{out_prefix.replace(".tsv","_terms_of_interest.txt")}', 'w') as f:
        f.write("\n".join(sorted(toi)))
    # dfk_df2_toi = dfk_df2[dfk_df2['term'].isin(toi)]
    dfk = dfk[(dfk['term'].isin(toi)) & (dfk['EntryID'].isin(query_ids))]
    df2 = df2[(df2['term'].isin(toi)) & (df2['EntryID'].isin(query_ids))]
    logger.info(f"Terms of interest: {len(toi)} terms")
    logger.info(f"Known annotations after filtering: {len(dfk)} annotations")
    logger.info(f"New annotations after filtering: {len(df2)} annotations")

    # (added Mar 2026) Remove binding-only proteins from MFO ground truth
    import networkx as nx
    subontologies_F = fetch_aspect(clean_ontology_edges(obonet.read_obo(graph)), root='GO:0003674')
    binding_terms = set(nx.descendants(subontologies_F,'GO:0005515'))
    binding_terms.add('GO:0005515')

    annotated_F_proteins = set(df2[df2['aspect']=='F']['EntryID'])
    not_binding_only_proteins = set(df2[(df2['EntryID'].isin(annotated_F_proteins)) &
                                                  (~df2['term'].isin(binding_terms)) &
                                                  (df2['aspect']=='F')]['EntryID']) # these proteins have other MFO annotations
    binding_df = df2[(df2['EntryID'].isin(annotated_F_proteins - not_binding_only_proteins)) &
                               (df2['aspect']=='F')] # these annotations are protein-binding only
    logger.debug(f"Binding-only rows removed from MFO ground truth: {binding_df.head().to_dict(orient='records')}")
    # Remove rows of binding-only proteins from df2, but proteins with binding + other MFO annotations can stay (because they are not binding-only)
    df2_old = df2.copy()
    df2 = df2[~df2.index.isin(binding_df.index)]
    logger.info(f"Removed {len(df2_old['EntryID'].unique())-len(df2['EntryID'].unique())} binding-only proteins from MFO aspect")

    # Compare dfk and df2 to find proteins that gained new terms
    dfk_df2_toi = dfk.merge(df2, on=df2.columns.to_list(), how='outer', indicator=True)
    logger.info(f"Merge results: {dict(dfk_df2_toi['_merge'].value_counts())}")
    df2_gain = dfk_df2_toi.loc[dfk_df2_toi._merge=='right_only',dfk_df2_toi.columns!='_merge'] # new terms
    logger.info(f"Number of terms gained in each aspect: {dict(df2_gain['aspect'].value_counts())}")

    p_asp_pairs = df2_gain[['EntryID', 'aspect']].drop_duplicates().reset_index(drop=True)
    logger.info(f"Number of proteins gaining terms: {len(df2_gain['EntryID'].unique())}, with {len(p_asp_pairs)} protein-aspect pairs")

    # # Remove already existing annotations in df2 (biocuration redundancy or partial knowledge proteins)
    # redundant_proteins = set.difference(set(df2['EntryID']), set(df2_gain['EntryID']))
    # if len(redundant_proteins) > 0:
    #     logger.warning(f"There were {len(redundant_proteins)} proteins whose new annotations already existed: {list(redundant_proteins)[:10]}")

    # NK: proteins in query_ids that are in df2 but not in dfk
    logger.info("Processing No Knowledge proteins...")
    NK = set.intersection(set.difference(set(df2['EntryID']), set(dfk['EntryID'])), set(query_ids))
    NK_df = df2[df2['EntryID'].isin(NK)].reset_index(drop=True) # propagated already
    NK_asp_pairs = NK_df[['EntryID', 'aspect']].drop_duplicates().reset_index(drop=True)
    # NK_t1 = filter_terms_given_obo(NK_df, current_graph=graph2, pivot_graph=graph)
    if len(NK_df) == 0:
        logger.warning("No proteins found in No Knowledge subset")
    NK_df.to_csv(f'{out_prefix.replace(".tsv","_NK.tsv")}', sep="\t", index=False, header=True)
    logger.info(f"Number of NK proteins: {len(NK)}, by aspect: {dict(NK_asp_pairs['aspect'].value_counts())}")

    # LK: proteins in query_ids that gained new aspect in df2
    logger.info("Processing Limited Knowledge and Partial Knowledge proteins...")
    remaining_ids = set.intersection(set(query_ids), set(df2_gain['EntryID'])) - set(NK) # proteins in query_ids that gained new terms
    if len(remaining_ids) == 0:
        logger.warning("No remaining proteins for LK/PK classification")
        # Create empty dataframes and save
        empty_df = pd.DataFrame(columns=['EntryID','term','aspect'])
        empty_df.to_csv(f'{out_prefix.replace(".tsv","_LK.tsv")}', sep="\t", index=False, header=True)
        empty_df.to_csv(f'{out_prefix.replace(".tsv","_PK.tsv")}', sep="\t", index=False, header=True)
        empty_df.to_csv(f'{out_prefix.replace(".tsv","_PK_known.tsv")}', sep="\t", index=False, header=True)
        with open(f'{out_prefix.replace(".tsv","_targets.tsv")}', 'w') as f:
            for target in sorted(NK):
                f.write(f"{target}\n")
        return

    dfk_remaining = dfk[dfk['EntryID'].isin(remaining_ids)]
    known_asp_pairs = dfk_remaining.groupby('EntryID')['aspect'].apply(set).to_dict() # protein-aspect pairs in t0
    logger.debug(f"Remaining IDs for LK/PK classification: {len(remaining_ids)} proteins")

    # Check each remaining protein-aspect pair in gained dataframe (p_asp_pairs)
    p_asp_pairs_filtered = p_asp_pairs[p_asp_pairs['EntryID'].isin(remaining_ids)].copy()
    # Vectorized classification: add a column indicating if aspect is known
    p_asp_pairs_filtered['aspect_known'] = p_asp_pairs_filtered.apply(
        lambda row: row['aspect'] in known_asp_pairs.get(row['EntryID'], set()),
        axis=1
    )
    # If aspect is known at t0 (known_asp_pairs) then it is Partial Knowledge, otherwise Limited Knowledge
    LK_pairs = p_asp_pairs_filtered[~p_asp_pairs_filtered['aspect_known']]
    PK_pairs = p_asp_pairs_filtered[p_asp_pairs_filtered['aspect_known']]

    # Get gained terms by matching gained aspect to df2_gain
    if len(LK_pairs) > 0:
        LK_df = df2_gain.merge(
            LK_pairs[['EntryID', 'aspect']],
            on=['EntryID', 'aspect'],
            how='inner'
        )
        LK_asp_pairs = LK_df[['EntryID', 'aspect']].drop_duplicates()
    else:
        LK_df = pd.DataFrame(columns=['EntryID','term','aspect'])
        LK_asp_pairs = pd.DataFrame(columns=['EntryID', 'aspect'])
        logger.warning("No proteins found for Limited Knowledge classification.")
    LK = set(LK_pairs['EntryID'].unique()) if len(LK_pairs) > 0 else set()
    LK_df.to_csv(f'{out_prefix.replace(".tsv","_LK.tsv")}', sep="\t", index=False, header=True)
    logger.info(f"Number of LK proteins: {len(LK)} proteins, by aspect: {dict(LK_asp_pairs['aspect'].value_counts()) if len(LK_asp_pairs) > 0 else {}}")

    # Create PK DataFrame using merge instead of loops
    if len(PK_pairs) > 0:
        PK_df = df2_gain.merge(
            PK_pairs[['EntryID', 'aspect']],
            on=['EntryID', 'aspect'],
            how='inner'
        )
        PK_asp_pairs = PK_df[['EntryID', 'aspect']].drop_duplicates().reset_index(drop=True)

        # Create PK_known DataFrame using merge instead of loops
        # Note: some proteins gained a new ancestral term (due to graph2 having new edges to terms in TOI), but leaf term does not change -> should not be gaining?
        dfk_PK = dfk.merge(
            PK_pairs[['EntryID', 'aspect']],
            on=['EntryID', 'aspect'],
            how='inner'
        )
    else:
        PK_df = pd.DataFrame(columns=['EntryID','term','aspect'])
        PK_asp_pairs = pd.DataFrame(columns=['EntryID', 'aspect'])
        dfk_PK = pd.DataFrame(columns=['EntryID','term','aspect'])
        logger.warning("No proteins found for Partial Knowledge classification.")
    PK = set(PK_pairs['EntryID'].unique()) if len(PK_pairs) > 0 else set()
    PK_df.to_csv(f'{out_prefix.replace(".tsv","_PK.tsv")}', sep="\t", index=False, header=True)
    dfk_PK.to_csv(f'{out_prefix.replace(".tsv","_PK_known.tsv")}', sep="\t", index=False, header=True)
    logger.info(f"Number of PK proteins: {len(PK)} proteins, by aspect: {dict(PK_asp_pairs['aspect'].value_counts()) if len(PK_asp_pairs) > 0 else {}}")

    # LK_dict = {}
    # PK_dict = {}
    # for _, row in p_asp_pairs.iterrows():
    #     if row['EntryID'] not in remaining_ids:
    #         continue # already classified as NK
    #     dfk_asp = dfk_asp_pairs[dfk_asp_pairs['EntryID'] == row['EntryID']]['aspect'].values[0]
    #     if row['aspect'] in dfk_asp: # if gained aspect is in known aspect, it is PK
    #         if row['EntryID'] not in PK_dict:
    #             PK_dict[row['EntryID']] = set(row['aspect'])
    #         else:
    #             PK_dict[row['EntryID']].add(row['aspect'])
    #     else: # otherwise, it is LK in that aspect
    #         if row['EntryID'] not in LK_dict:
    #             LK_dict[row['EntryID']] = set(row['aspect'])
    #         else:
    #             LK_dict[row['EntryID']].add(row['aspect'])

    # LK_df = pd.DataFrame(columns=['EntryID','term','aspect'])
    # for protein, aspects in LK_dict.items():
    #     df = df2_gain[(df2_gain['EntryID'] == protein) & (df2_gain['aspect'].isin(aspects))]
    #     LK_df = pd.concat([LK_df, df], ignore_index=True) # leaf and ancestor (because df2 is already propagated)
    # if len(LK_df) == 0:
    #     LK_asp_pairs = pd.DataFrame(columns=['EntryID', 'aspect'])
    #     logger.warning("No proteins found in df2_gain for Limited Knowledge classification.")
    # else:
    #     LK_asp_pairs = LK_df[['EntryID', 'aspect']].drop_duplicates()
    # LK_df.to_csv(f'{out_prefix.replace(".tsv","_LK.tsv")}', sep="\t", index=False, header=True) # new terms propagated
    # logger.info(f"Number of LK proteins: {len(LK_dict)}, by aspect: {dict(LK_asp_pairs['aspect'].value_counts())}")

    # PK_df = pd.DataFrame(columns=['EntryID','term','aspect'])
    # for protein, aspects in PK_dict.items():
    #     df = df2_gain[(df2_gain['EntryID'] == protein) & (df2_gain['aspect'].isin(aspects))]
    #     PK_df = pd.concat([PK_df, df], ignore_index=True)
    # if len(PK_df) == 0:
    #     PK_asp_pairs = pd.DataFrame(columns=['EntryID', 'aspect'])
    #     logger.warning("No proteins found in df2_gain for Partial Knowledge classification.")
    # else:
    #     PK_asp_pairs = PK_df[['EntryID', 'aspect']].drop_duplicates().reset_index(drop=True)
    # PK_df.to_csv(f'{out_prefix.replace(".tsv","_PK.tsv")}', sep="\t", index=False, header=True) # new terms only

    assert len(NK_asp_pairs) + len(LK_asp_pairs) + len(PK_asp_pairs) == len(p_asp_pairs), "Sum of aspect pairs in NK, LK, PK should equal total gained aspect pairs"
    logger.debug("Assertion passed: sum of aspect pairs equals total gained aspect pairs")
    # dfk_PK = []
    # for protein, aspect in zip(PK_asp_pairs['EntryID'], PK_asp_pairs['aspect']):
    #     add_PK = dfk[(dfk['EntryID'] == protein) & (dfk['aspect'] == aspect)]
    #     dfk_PK.append(add_PK)
    # dfk_PK = pd.concat(dfk_PK, ignore_index=True)
    # dfk_PK.to_csv(f'{out_prefix.replace(".tsv","_PK_known.tsv")}', sep="\t", index=False, header=True)

    logger.info(f"- No Knowledge (NK): {len(NK)} proteins, {len(NK_df)} terms")
    logger.info(f"- Limited Knowledge (LK): {len(LK)} proteins, {len(LK_df)} terms")
    logger.info(f"- Partial Knowledge (PK): {len(PK)} proteins, {len(PK_df)} new child terms gained")

    all_targets = set.union(set(NK), set(LK), set(PK))
    logger.info(f"Total number of target proteins: {len(all_targets)}")
    with open(f'{out_prefix.replace(".tsv","_targets.tsv")}', 'w') as f:
        for target in sorted(all_targets):
            f.write(f"{target}\n")

def parse_inputs(argv):
    parser = argparse.ArgumentParser(
        description='Compare two annotation files and classify proteins into No Knowledge, Limited Knowledge, and Partial Knowledge')


    parser.add_argument('--annot_known', '-ak', required=True,
                        help='Path to first annotation file (can be gzipped). This file is t0.')
    parser.add_argument('--annot2', '-a2', required=True,
                        help='Path to second annotation file (can be gzipped). This file is t1.')
    parser.add_argument('--query_file', '-q', required=True,
                        help='Path to target set of proteins, either in .fasta format or .tsv with protein IDs')
    parser.add_argument('--graph', '-g', default=None,
                        help='Path to OBO ontology graph at pivot timepoint (frozen graph).')
    parser.add_argument('--graph2', '-g2', default=None,
                        help='Path to OBO ontology graph at a ground truth timepoint.')
    parser.add_argument('--out_prefix', default='groundtruth.tsv',
                        help='Prefix for 3 output files')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level (default: INFO)')
    return parser.parse_args(argv)

    # python3 -m democafa.groundtruth.classify_ground_truth -ak data/processed/cafa5/t0_terms.tsv -a2 data/processed/cafa5/t1_terms.tsv
    # -q data/processed/cafa5/test_superset_all.fasta -g data/raw/go-basic.obo -g2 data/raw/t1_go-basic.obo --out_prefix data/processed/cafa5/t1_groundtruth.tsv

    # python3 -m democafa.groundtruth.classify_ground_truth -ak data/processed/cafa5/t0_terms.tsv -a2 data/processed/cafa5/t1_terms.tsv
    # -q data/processed/cafa5/test_superset_all.fasta -g data/raw/go-basic.obo -g2 ../robot_obo/results/t1_iba-slim-roots.obo --out_prefix data/processed/cafa5/goslim_t1_groundtruth.tsv

def main():
    args = parse_inputs(sys.argv[1:])

    # Set up file logging when running as a script
    setup_logging(use_file_handler=True, log_level=args.log_level)

    logger.info(f"Arguments: {vars(args)}")

    try:
        wrapper_ground_truth(
            annot_known=args.annot_known,
            annot2=args.annot2,
            query_file=args.query_file,
            graph=args.graph,
            graph2=args.graph2,
            out_prefix=args.out_prefix
        )
    except Exception as e:
        logger.error(f"Error in classify_ground_truth script: {str(e)}")
        raise

if __name__ == "__main__":
    main()
