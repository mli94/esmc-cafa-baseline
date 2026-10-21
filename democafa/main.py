#!/usr/bin/env python3

"""IMPORTANT: main.py file is outdated, refer to the instruction in README.md to run each module separately.
Most functions listed here have not been updated for correct arguments
"""


import os
from democafa.config import GO_CODES, RAW_FILE_PATHS, PROCESSED_FILE_PATHS, BASELINE_PATHS
from democafa.datacollection.filter_gaf import filter_gaf
from democafa.datacollection.retrieve_terms import wrapper_retrieve_terms
from democafa.datacollection.create_test_set import create_test_set
from democafa.datacollection.propagate_and_ia import propagate_and_ia

from democafa.baselines.blast import blast_predict
from democafa.baselines.naive import naive_predict
from democafa.baselines.goa_nonexp import goa_nonexp_predict
from democafa.baselines.prott5 import prott5_predict

from democafa.groundtruth.classify_ground_truth import wrapper_ground_truth

def main():
    # 0. Load configuration
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    # 1. Collecting data for release
    filter_gaf(
        file_path=RAW_FILE_PATHS['uniprot_goa'],
        entries=RAW_FILE_PATHS['swissprot_fasta'],
        output_file=PROCESSED_FILE_PATHS['goa_uniprot_filtered']
        # taxon_file=RAW_FILE_PATHS['testsuperset_taxon']
    )
    if not os.path.exists(PROCESSED_FILE_PATHS['train_terms']):
        wrapper_retrieve_terms(
            annot_file=RAW_FILE_PATHS['goa_uniprot_filtered'],
            # taxon=RAW_FILE_PATHS['testsuperset_taxon'],
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS',
            graph=RAW_FILE_PATHS['obo'],
            output_tsv=PROCESSED_FILE_PATHS['train_terms']
        )
    else:
        print(f"{PROCESSED_FILE_PATHS['train_terms']} already exists, moving to sequences retrieval")

    if not os.path.exists(PROCESSED_FILE_PATHS['test_sequences']):
        create_test_set(
            terms_file=PROCESSED_FILE_PATHS['train_terms'],
            sequences_gzfile=RAW_FILE_PATHS['swissprot_fasta'], # full SwissProt fasta file
            out_fasta=PROCESSED_FILE_PATHS['test_sequences'],
            train_out_fasta=PROCESSED_FILE_PATHS['train_sequences'],
            train_out_taxonomy=PROCESSED_FILE_PATHS['train_taxonomy'],
            include_all=True # include all proteins in test superset (not just proteins with missing aspects)
        )
    else:
        print(f"{PROCESSED_FILE_PATHS['test_sequences']} already exists, all data collected")

    if not os.path.exists(PROCESSED_FILE_PATHS['train_matrix']):    
        propagate_and_ia(
            terms_file=PROCESSED_FILE_PATHS['train_terms'],
            tsv_propagated=PROCESSED_FILE_PATHS['train_terms_propagated'],
            graph=RAW_FILE_PATHS['obo'],
            output_tsv=PROCESSED_FILE_PATHS['train_ia'] # change to None if don't need IA calculation
        )
    # Use os module to copy files from processed to release folder
    # os.system('cp data/processed/* data/release/')
    # os.system(f'cp {RAW_FILE_PATHS["obo"]} data/release/')
    # print(f"Data for release ready with {len(os.listdir('data/release'))} files")
    
    # 2. Create baseline predictors
    # Collect known terms by end of submission (t0)
    filter_gaf(
        file_path=RAW_FILE_PATHS['t0_uniprot_goa'],
        entries=RAW_FILE_PATHS['t0_swissprot_fasta'],
        output_file=PROCESSED_FILE_PATHS['t0_goa_uniprot_filtered']
        # taxon_file=RAW_FILE_PATHS['testsuperset_taxon']
    )
    if not os.path.exists(PROCESSED_FILE_PATHS['t0_terms']):
        wrapper_retrieve_terms(
            annot_file=RAW_FILE_PATHS['t0_uniprot_goa_filtered'],
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS',
            graph=RAW_FILE_PATHS['obo'],
            add_graph=RAW_FILE_PATHS['t0_obo'],
            output_tsv=PROCESSED_FILE_PATHS['t0_terms']
        )
    else:
        print(f"{PROCESSED_FILE_PATHS['t0_terms']} already exists")
        
    if not os.path.exists(BASELINE_PATHS['baseline_goa_nonexp']):
        goa_nonexp_predict(
            annot_file=RAW_FILE_PATHS['t0_uniprot_goa_filtered'],
            graph=RAW_FILE_PATHS['obo'],
            add_graph=RAW_FILE_PATHS['t0_obo'],
            selected_go='Computational,Phylogenetical,Electronic,ND,NAS',
            query_file=PROCESSED_FILE_PATHS['test_sequences_all'],
            output_baseline=BASELINE_PATHS['baseline_goa_nonexp']
        )
    if not os.path.exists(BASELINE_PATHS['baseline_naive']):
        naive_predict(
            # annotations=PROCESSED_PATHS['train_matrix'],
            # indices=PROCESSED_PATHS['matrix_indices'],
            annotations=RAW_FILE_PATHS['t0_uniprot_goa_filtered'],
            graph=RAW_FILE_PATHS['obo'],
            add_graph=RAW_FILE_PATHS['t0_obo'],
            query_file=PROCESSED_FILE_PATHS['test_sequences_all'],
            output_baseline=BASELINE_PATHS['baseline_naive'] 
        )
    
    # os.system(f'mkdir -p {DATA_DIR}/processed/blast_db && cp {PROCESSED_PATHS["train_sequences"]} {DATA_DIR}/processed/blast_db/')
    # os.system(f'sbatch {EXTERNAL_TOOLS["blast"]}') 
    if not os.path.exists(BASELINE_PATHS['baseline_blast']):
        blast_predict(
            annotations=PROCESSED_FILE_PATHS['t0_goa_uniprot_filtered'],
            query_file=PROCESSED_FILE_PATHS['test_sequences'],
            # indices=PROCESSED_FILE_PATHS['matrix_indices'],
            blast_results=BASELINE_PATHS['output_blast'],
            output_baseline=BASELINE_PATHS['baseline_blast'],
            keep_self_hits=False,
            use_rscore=False
        )
    
    # # os.system(f'mkdir -p {DATA_DIR}/processed/prott5')
    # # os.system(f'sbatch {EXTERNAL_TOOLS["prott5"]} -q {RELEASE_PATHS["test_sequences"]} -d {PROCESSED_PATHS["train_sequences"]} -m {VERSIONS["prott5_model"]} -o {PROCESSED_PATHS["output_prott5_raw"]}')
    if not os.path.exists(BASELINE_PATHS['baseline_prott5']):
        prott5_predict(
            annotations=PROCESSED_FILE_PATHS['t0_goa_uniprot_filtered'],
            query_file=PROCESSED_FILE_PATHS['test_sequences'],
            # indices=PROCESSED_FILE_PATHS['matrix_indices'],
            prott5_results=BASELINE_PATHS['output_prott5'],
            output_baseline=BASELINE_PATHS['baseline_prott5'],
            keep_self_hits=False
        )
        
    # # 3. Evaluation
    # os.system(f'mkdir -p {DATA_DIR}/processed/predictions')
    
    # Collect t1 ground truth from a later release
    if not os.path.exists(PROCESSED_FILE_PATHS['t1_terms']):
        wrapper_retrieve_terms(
            annot_file=RAW_FILE_PATHS['t1_uniprot_goa_filtered'],
            go_codes=GO_CODES,
            selected_go_codes='Experimental,IC,TAS',
            graph=RAW_FILE_PATHS['obo'],
            add_graph=RAW_FILE_PATHS['t1_obo'],
            output_tsv=PROCESSED_FILE_PATHS['t1_terms']
        )
    else:
        print(f"{PROCESSED_FILE_PATHS['t1_terms']} already exists")

    if not os.path.exists(f"{PROCESSED_FILE_PATHS['t1_ground_truth']}_NK.tsv"):
        wrapper_ground_truth(
            annot_known=PROCESSED_FILE_PATHS['t0_terms'], # t0 annotations
            annot2=PROCESSED_FILE_PATHS['t1_terms'],
            query_file=PROCESSED_FILE_PATHS['test_sequences'],
            graph=RAW_FILE_PATHS['obo'], # frozen graph or GO slim graph
            graph2=RAW_FILE_PATHS['t1_obo'],
            out_prefix=PROCESSED_FILE_PATHS['t1_ground_truth']
        )
    
    
if __name__ == "__main__":
    main()
