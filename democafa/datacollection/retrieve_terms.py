#!/usr/bin/env python3

"""Extract protein-to-GO annotations from GOA GAF or UniProt DAT files based on 
selected evidence codes.

The output is a de-duplicated TSV with columns ``EntryID``, ``term``, and
``aspect``. ``NOT`` annotations are excluded, evidence codes are selected by
GO-code group or individual code, alternate GO IDs are normalized through the
provided ontology, and obsolete terms absent from the ontology are removed.

When ``add_graph`` is provided, annotations from a later ontology are filtered
back to the pivot ontology so downstream ground-truth and prediction files use
the same term universe.
"""


import os
import sys
import argparse
import gzip
import logging
from datetime import datetime
import obonet
import pandas as pd
from Bio.UniProt import GOA
from Bio import SwissProt as sp
from democafa.utils.ontology import clean_ontology_edges, filter_terms_given_obo, replace_alternate_GO_terms, replace_obsolete_GO_terms_replacedby
from democafa.utils.constants import GO_CODES
from democafa.datacollection.remove_subcell_annt import remove_subcell_annt_from_annotation_df

# Create a specific logger for this module (not the root logger)
logger = logging.getLogger('retrieve_terms')
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
        log_filename = os.path.join(log_dir, datetime.now().strftime('retrieve_terms_%Y%m%d_%H%M%S.log'))
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


def process_gaf_file(gaf_file):
    '''
    function : given a file handle, find the !gaf-version line and return all content from there
    input    : file path
    output   : content starting from !gaf-version line
    '''
    # Check if file is gzipped
    is_gzipped = gaf_file.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'

    with open_func(gaf_file, mode) as f:
        content = f.read()

    # Find the position of !gaf-version line
    gaf_version_pos = content.find('!gaf-version')
    if gaf_version_pos == -1:
        return content  # If not found, return all content
    return content[gaf_version_pos:]  # Return content from !gaf-version onwards


def filter_evidence_codes(go_codes, selected='Experimental,IC,TAS'):
    """Resolve GO evidence-code group names and individual codes."""
    ALL_LISTS = go_codes
    ALL_CODES = set(code for codes in ALL_LISTS.values() for code in codes)
    input_codes = [code.strip().upper() for code in selected.split(',')]
    accepted_codes = set()
    for code in input_codes:
        if code in ALL_LISTS:
            accepted_codes.update(ALL_LISTS[code])
        elif code in ALL_CODES:
            accepted_codes.add(code)
        else:
            logger.warning(f"'{code}' is not a recognized evidence code.")
    logger.debug(f"Selected evidence codes: {accepted_codes}")
    return {'Evidence': set(accepted_codes)}


def read_gaf(file_path, selected_codes, record_db='UniProtKB'):
    """
    Read and process a GAF file (gzipped or plain text)
    Takes 1h45m for 22Gb uniprot goa (if not pre-filtered)
    """
    #selected_codes = filter_evidence_codes(go_codes, selected) # handle this before passing to this function
    data = []
    
    # Process annotations
    is_gzipped = file_path.endswith('.gz')
    processed_count = 0
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    with open_func(file_path, mode) as handle:
        for rec in GOA.gafiterator(handle):
            processed_count += 1
            if processed_count % 100000 == 0:
                logger.debug(f"Processed {processed_count} records, kept {len(data)} annotations")

            if 'NOT' in rec['Qualifier']:
                continue
            if GOA.record_has(rec, selected_codes) and rec['DB'] == record_db:
                data.append({'EntryID': rec['DB_Object_ID'], 'term': rec['GO_ID'], 'aspect': rec['Aspect'], 'date': rec['Date']})

    logger.info(f"Processed {processed_count} total records, kept {len(data)} annotations")
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} unique annotations")
    return df


def process_chunk_file(chunk_file, selected_codes, record_db='UniProtKB'):
    chunk_data = []
    processed_count = 0
    with open(chunk_file, 'r') as f:
        for rec in GOA.gafiterator(f):
            processed_count += 1
            if 'NOT' in rec['Qualifier']:
                continue
            if GOA.record_has(rec, selected_codes) and rec['DB'] == record_db:
                chunk_data.append({
                    'EntryID': rec['DB_Object_ID'],
                    'term': rec['GO_ID'],
                    'aspect': rec['Aspect']
                })

    logger.debug(f"Chunk {os.path.basename(chunk_file)}: processed {processed_count} records, kept {len(chunk_data)} annotations")
    return chunk_data


def read_gaf_mp(file_path, selected_codes, record_db='UniProtKB', use_mp=False, num_processes=None, chunk_size=10000000):
    """
    Read and process a GAF file (gzipped or plain text) with multiprocessing.
    Takes 30m for 22Gb uniprot goa for 15 processes.
    """
    logger.info(f"Starting GAF reading process. Using multiprocessing: {use_mp}")

    if not use_mp:
        return read_gaf(file_path, selected_codes, record_db=record_db)
    else:
        import multiprocessing
        from functools import partial
        import tempfile
        if num_processes is None:
            num_processes = min(int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) - 1, 16)
            num_processes = max(1, num_processes)
            logger.info(f"Using {num_processes} processes for parallel computation")



        temp_dir = tempfile.mkdtemp(dir=os.path.dirname(file_path) or None)
        logger.debug(f"Created temporary directory: {temp_dir}")
        chunk_files = []
        try:
            # Read and chunk the file
            logger.info(f"Splitting GAF file into chunks of {chunk_size} records...")
            chunk_count = 0
            record_count = 0
            is_gzipped = file_path.endswith('.gz')
            open_func = gzip.open if is_gzipped else open
            mode = 'rt' if is_gzipped else 'r'

            with open_func(file_path, mode) as handle:
                current_chunk_lines = []
                headers = []
                line = handle.readline()
                while line.startswith('!'):
                    headers.append(line)
                    line = handle.readline()

                logger.debug(f"Found {len(headers)} header lines")

                # Process first non-header line
                if line:
                    current_chunk_lines.append(line)

                # Process remaining lines
                for line in handle:
                    current_chunk_lines.append(line)
                    record_count += 1

                    if len(current_chunk_lines) >= chunk_size:
                        # Write chunk to temporary file
                        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_count}.gaf")
                        with open(chunk_file, 'w') as f:
                            # Write headers to each chunk
                            f.writelines(headers)
                            f.writelines(current_chunk_lines)

                        chunk_files.append(chunk_file)
                        chunk_count += 1
                        current_chunk_lines = []
                        if chunk_count % 10 == 0:
                            logger.info(f"Created chunk {chunk_count} with {chunk_size} records")
                # Write any remaining lines to the final chunk
                if current_chunk_lines:
                    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_count}.gaf")
                    with open(chunk_file, 'w') as f:
                        # Write headers to each chunk
                        f.writelines(headers)
                        f.writelines(current_chunk_lines)

                    chunk_files.append(chunk_file)
                    chunk_count += 1
                    logger.info(f"Created final chunk {chunk_count} with {len(current_chunk_lines)} records")

            # Process chunks in parallel
            logger.info(f"Processing {len(chunk_files)} chunks with {num_processes} processes")
            process_func = partial(process_chunk_file, selected_codes=selected_codes, record_db=record_db)
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(process_func, chunk_files)

            all_data = [item for sublist in results for item in sublist]
            logger.info(f"Retrieved {len(all_data)} annotations from all chunks")
            df = pd.DataFrame(all_data)
            df = df.drop_duplicates()
            logger.info(f"After removing duplicates: {len(df)} unique annotations")
            return df
        except Exception as e:
            logger.error(f"Error during GAF reading: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            logger.debug("Cleaning up temporary files")
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)


def process_go_from_dat(file_path, selected_codes):
    """Extract selected GO cross-references from a UniProt DAT file."""
    logger.info(f"Processing GO annotations from DAT file: {file_path}")
    entries = []
    # selected_codes = filter_evidence_codes(GO_CODES, selected).get('Evidence') # handle this before passing to this function
    is_gzipped = file_path.endswith('.gz')
    processed_records = 0
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    with open_func(file_path, mode) as handle:
        for record in sp.parse(handle):
            processed_records += 1
            if processed_records % 10000 == 0:
                logger.debug(f"Processed {processed_records} records, found {len(entries)} annotations")

            if not record.taxonomy_id:
                continue
            # if selected_taxon is not None: # only filter if selected_taxon is not None
            #     if f'taxon:{record.taxonomy_id[0]}' not in selected_taxon:
            #         continue
            current_id = record.accessions[0]
            for dr in record.cross_references:    #dr -> db cross refernce
                if dr[0] == 'GO' and len(dr) >= 4:
                    go_id = dr[1]
                    aspect = dr[2][0]  # Getting only the first letter (its either P/ C/ F)
                    # aspect_description = dr[2][2:]  # Getting the rest of the description
                    evidence = dr[3] if len(dr) >= 4 else ''
                    evidence_code = evidence[:3]  # First 3 letters of evidence
                    if evidence_code not in selected_codes:
                        continue
                    # evidence_source = evidence[4:] if len(evidence) > 4 else ''
                    entries.append({
                        "EntryID": current_id,
                        "term": go_id,
                        "aspect": aspect
                    })

    logger.info(f"Processed {processed_records} total records, found {len(entries)} annotations")
    df = pd.DataFrame(entries)
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} unique annotations")
    return df

def map_uniprotkb_from_recorddb(file_path, record_db):
    """Extract selected GO cross-references from a UniProt DAT file."""
    logger.info(f"Processing GO annotations from DAT file: {file_path}")
    entries = dict()
    is_gzipped = file_path.endswith('.gz')
    processed_records = 0
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    with open_func(file_path, mode) as handle:
        for record in sp.parse(handle):
            processed_records += 1
            if processed_records % 10000 == 0:
                logger.debug(f"Processed {processed_records} records, found {len(entries)} annotations")
                
            if not record.taxonomy_id:
                continue
            current_id = record.accessions[0]
            for dr in record.cross_references:    #dr -> db cross refernce
                if dr[0] == record_db:
                    # Getting UniProtKB accession mapping to record_db (e.g., RGD, MGI, etc.)
                    db_accession = dr[1]
                    entries[db_accession] = current_id
                    break  # Assuming only one mapping per record for the specified record_db
    return entries


def wrapper_retrieve_terms(annot_file, selected_go_codes, graph, add_graph=None, output_tsv='train_terms.tsv', remove_subcell_annt=False, record_db='UniProtKB', dat_mapping=None):
    """Read annotations, normalize ontology terms, and write a terms TSV."""
    logger.info(f"Annotation file: {annot_file}")
    logger.info(f"Selected GO codes: {selected_go_codes}")
    logger.info(f"Graph file: {graph}")
    logger.info(f"Additional graph file: {add_graph}")
    logger.info(f"Output file: {output_tsv}")
    output_dir = os.path.dirname(output_tsv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load annotations from GOA or DAT file
    if 'gaf' in annot_file:
        filetype = 'gaf'
    elif 'dat' in annot_file:
        filetype = 'dat'
    else:
        raise ValueError("Unsupported annotation file type. Please provide a GAF or DAT file.")

    logger.info(f"Detected file type: {filetype}")

    if filetype == 'gaf':
        selected_codes = filter_evidence_codes(GO_CODES, selected_go_codes)
        annotation_df = read_gaf_mp(annot_file, selected_codes, record_db=record_db, use_mp=False, chunk_size=10000000)
    elif filetype == 'dat':
        selected_codes = filter_evidence_codes(GO_CODES, selected_go_codes).get('Evidence')
        annotation_df = process_go_from_dat(annot_file, selected_codes)
    if annotation_df.empty:
        logger.warning(f"No annotations found for the given evidence codes {selected_go_codes}.")
        return

    logger.info(f"Initial annotations: {len(annotation_df)} for {len(set(annotation_df['EntryID']))} proteins")

    # load ontology graph and GO terms. obonet doesn't store OBSOLETE terms
    if add_graph is None:
        ontology_graph = clean_ontology_edges(obonet.read_obo(graph, ignore_obsolete=False))
    else:
        ontology_graph = clean_ontology_edges(obonet.read_obo(add_graph, ignore_obsolete=False))
    logger.info(f"Loaded ontology with {len(ontology_graph.nodes())} terms")

    annotation_df = replace_alternate_GO_terms(annotation_df, ontology_graph)
    annotation_df = replace_obsolete_GO_terms_replacedby(annotation_df, ontology_graph)

    ontology_graph_valid = clean_ontology_edges(obonet.read_obo(graph))
    obsolete_terms = set(annotation_df['term']) - set(ontology_graph_valid.nodes())
    if obsolete_terms:
        # These could be obsolete without replaced_by direct replacement, or new terms that are not in the ontology graph yet
        logger.warning(f"{len(obsolete_terms)} obsolete terms (or new terms) found in the annotation file: {list(obsolete_terms)[:10]}. These terms will not appear in terms file.")
        annotation_df = annotation_df[~annotation_df['term'].isin(obsolete_terms)]
        
    # Added May 2026: remove subcellular localization or RHEA annotations that are experimental (for CAFA6), 
    # these were pre-existing UniProtKB-SubCell and RHEA annotations with IEA evidence code, that were recently 
    # added as EXP annotations, they are not really new annotations so remove them from t1 annotations
    if remove_subcell_annt:
        annotation_df = remove_subcell_annt_from_annotation_df(annotation_df, annot_file)
        
    # Added Jun 2026: map UniProtKB to record_db (e.g., RGD, MGI, etc.) if record_db is provided
    if record_db and record_db != 'UniProtKB':
        if not dat_mapping:
            raise ValueError("dat_mapping file must be provided when using a record_db other than UniProtKB.")
        logger.info(f"Mapping UniProtKB accessions to {record_db} accessions...")
        uniprot_to_record_db = map_uniprotkb_from_recorddb(dat_mapping, record_db)
        annotation_df['map_UniProtKB'] = annotation_df['EntryID'].map(uniprot_to_record_db)
        missing_mappings = set(annotation_df[annotation_df['map_UniProtKB'].isnull()]['EntryID'])
        annotation_df['EntryID'] = annotation_df['EntryID'].map(uniprot_to_record_db)
        if missing_mappings:
            logger.warning(f"{len(missing_mappings)} entries could not be mapped to {record_db}. They will be excluded from the output.")
            annotation_df = annotation_df.dropna(subset=['EntryID'])
        annotation_df = annotation_df.drop(columns=['map_UniProtKB'])
    # Remove terms that are not in the frozen graph in 3 steps
    # (propagate using graph2, intersect with graph terms, propagate again with graph)
    if add_graph is not None:
        logger.info("Filtering terms using additional graph...")
        annotation_df_filtered = filter_terms_given_obo(annotation_df, current_graph=add_graph, pivot_graph=graph)
        if 'date' in annotation_df_filtered.columns:
            annotation_df_filtered = annotation_df_filtered.sort_values(by=['date'], ascending=True)  # Sort by date to keep the earliest annotation
        else:
            annotation_df_filtered['date'] = [0]*len(annotation_df_filtered)  # Add a date column with NaT if it doesn't exist
        annotation_df_filtered = annotation_df_filtered.drop_duplicates(subset=['EntryID', 'term', 'aspect'], keep='first')
        annotation_df_filtered.to_csv(output_tsv, sep='\t', index=False)
    else:
        if 'date' in annotation_df.columns:
            annotation_df_filtered = annotation_df.sort_values(by=['date'], ascending=True)  # Sort by date to keep the earliest annotation
        else:
            annotation_df_filtered = annotation_df.copy()
            annotation_df_filtered['date'] = [0]*len(annotation_df_filtered)  # Add a date column with NaT if it doesn't exist
        annotation_df_filtered = annotation_df_filtered.drop_duplicates(subset=['EntryID', 'term', 'aspect'], keep='first')
        annotation_df_filtered.to_csv(output_tsv, sep='\t', index=False)

    logger.info(f"Annotations saved to {output_tsv}")
    logger.info(f"Total: {len(set(annotation_df_filtered['EntryID']))} proteins, {len(set(annotation_df_filtered['term']))} unique terms, {len(annotation_df_filtered)} annotations.")


def parse_inputs(argv):
    parser = argparse.ArgumentParser(
        description='Retrieve terms annotated to UniProtKB proteins from GOA file and save to TSV file')

    parser.add_argument('--annot', '-a', required=True,
                        help='Path to first annotation file (can be gzipped)')
    parser.add_argument('--dat-mapping', '-dm', required=False, default=None,
                        help='Path to UniProt DAT file for mapping UniProtKB accessions to other databases (e.g., RGD, MGI, etc.)')
    parser.add_argument('--record_db', '-db', required=False, default='UniProtKB',
                        help='Record DB column to filter annotations by (default: UniProtKB). Example: Change to MGI if using MGI GAF file.')
    parser.add_argument('--selected_go_codes', '-sgc', required=False, default='Experimental,IC,TAS',
                        help='Comma-separated list of evidence codes to include in the analysis')
    parser.add_argument('--graph', '-g', required=True, default=None,
                        help='Path to OBO ontology graph file if local. If empty (default) current OBO structure at run-time will be downloaded from http://purl.obolibrary.org/obo/go/go-basic.obo')
    parser.add_argument('--add_graph', '-ag', default=None,
                        help='Path to OBO ontology graph of a later timepoint. Provide this graph to remove terms that are not in frozen graph.')
    parser.add_argument('--tsv', default='train_terms.tsv',
                        help='Path to save annotations in TSV format')
    parser.add_argument('--remove_subcell_annt', action='store_true',
                        help='Whether to remove subcellular localization and RHEA annotations that are experimental (for CAFA6)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set the logging level (default: INFO)')

    args = parser.parse_args(argv)

    return args

    # python3 -m democafa.datacollection.retrieve_terms --annot data/processed/cafa5/goa_uniprot_filtered_mp.gaf.213.gz
    # --taxon data/raw/testsuperset-taxon-list.tsv -sgc 'Experimental,IC,TAS' -g data/raw/go-basic.obo --tsv data/processed/cafa5/train_terms.tsv

    # python3 -m democafa.datacollection.retrieve_terms --annot data/processed/cafa5/goa_uniprot_filtered_mp.gaf.216.gz
    # --taxon data/raw/testsuperset-taxon-list.tsv -sgc 'Experimental,IC,TAS' -g data/raw/go-basic.obo -ag data/raw/t0_go-basic.obo --tsv data/processed/cafa5/t0_terms.tsv

    # python3 -m democafa.datacollection.retrieve_terms --annot data/processed/cafa5/goa_uniprot_filtered_mp.gaf.224.gz
    # --taxon data/raw/testsuperset-taxon-list.tsv -sgc 'Experimental,IC,TAS' -g data/raw/go-basic.obo -ag data/raw/t1_go-basic.obo --tsv data/processed/cafa5/t1_terms.tsv

def main():
    args = parse_inputs(sys.argv[1:])

    # Set up file logging when running as a script
    setup_logging(use_file_handler=True, log_level=args.log_level)

    logger.info(f"Arguments: {vars(args)}")

    try:
        wrapper_retrieve_terms(
            annot_file=args.annot,
            # taxon=args.taxon,
            selected_go_codes=args.selected_go_codes,
            graph=args.graph,
            add_graph=args.add_graph,
            output_tsv=args.tsv,
            remove_subcell_annt=args.remove_subcell_annt,
            record_db=args.record_db,
            dat_mapping=args.dat_mapping
        )
    except Exception as e:
        logger.error(f"Error in retrieve_terms script: {str(e)}")
        raise

if __name__ == "__main__":
    main()
