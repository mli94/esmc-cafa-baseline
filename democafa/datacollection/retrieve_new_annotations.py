"""Filter GOA annotations to only include those with a date after YYYYMMDD, and
retrieve the publish year of the reference for each annotation. This is used to create an annotation
dataset cutoff at a specific date
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
from Bio import Entrez, SwissProt, Medline
from democafa.utils.ontology import clean_ontology_edges, filter_terms_given_obo, replace_alternate_GO_terms
from democafa.utils.constants import GO_CODES
from democafa.datacollection.retrieve_terms import filter_evidence_codes

NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
Entrez.email = os.getenv("ENTREZ_EMAIL", "example@example.com")  # NCBI requires an email address for API access
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
    RATE_LIMIT_DELAY = 0.11  # 10 requests per second with API key
else:
    RATE_LIMIT_DELAY = 0.34  # 3 requests per second without API key

# Configuration
BATCH_SIZE = 200  # Number of PMIDs to fetch per request

def read_gaf(file_path, selected_codes, cutoff_date):
    """Read a GAF file and filter annotations based on evidence codes and date."""
    data = []
    
    # Process annotations
    is_gzipped = file_path.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    with open_func(file_path, mode) as handle:
        for rec in GOA.gafiterator(handle):
            if 'NOT' in rec['Qualifier']:
                continue
            if GOA.record_has(rec, selected_codes) and rec['DB'] == 'UniProtKB':
                if int(rec['Date']) >= cutoff_date:
                    data.append({'EntryID': rec['DB_Object_ID'], 'term': rec['GO_ID'], 'aspect': rec['Aspect'], 'reference': rec['DB:Reference'][0]})
    
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    return df

def retrieve_reference_years(pmid_list):
    """Retrieve the publication year for a list of PMIDs."""
    pmid_to_date_edate = {}
    for i in range(0, len(pmid_list), BATCH_SIZE):
        batch_pmids = ','.join(pmid_list[i:i+BATCH_SIZE])
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            for record in records:
                pmid = record.get('PMID', '?')
                pub_date = record.get('DP', '1900')
                epub_date = record.get('DEP', None)
                pmid_to_date_edate[f'PMID:{pmid}'] = [str(pub_date), str(epub_date) if epub_date else None]
            handle.close()
        except Exception as e:
            logging.error(f"Error fetching PMIDs {batch_pmids}: {e}")
    pmid_to_date = clean_reference_dates(pmid_to_date_edate)
    return pmid_to_date, pmid_to_date_edate

def clean_reference_dates(pmid_to_year):
    """Clean the publication years, extracting just the year from the date string."""
    cleaned_pmid_to_year = {}
    from datetime import datetime
    for pmid, date_edate in pmid_to_year.items():
        try:
            date_str = date_edate[0]
            # Turn YYYY mm dd into YYYYMMDD for comparison
            if len(date_str) == 4:  # Only year is present
                date_str = date_str + " Jan 01"  # Default to January 1st
            elif len(date_str) == 8:  # Year and month are present
                date_str = date_str + " 01"  # Default to the first of the month
            elif '-' in date_str:  # Date is in YYYY MM-MM format or YYYY MM DD-DD format
                date_str = date_str.split('-')[0]
                if len(date_str) == 8:
                    date_str = date_str + " 01"  # Default to the first of the month
                else:
                    date_str = date_str 
            date_str = datetime.strptime(date_str, "%Y %b %d").strftime("%Y%m%d")
            cleaned_pmid_to_year[pmid] = int(date_str)
        except Exception as e:
            logging.error(f"Error parsing date '{date_str}' for PMID {pmid}: {e}")
            cleaned_pmid_to_year[pmid] = date_str[:4]  # Fallback to just the year if parsing fails
    return cleaned_pmid_to_year

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter GOA annotations by date and retrieve reference years")
    parser.add_argument("--gaf_file", type=str, required=True, help="Path to the GOA GAF file (can be gzipped)")
    parser.add_argument("--cutoff_date", type=int, required=True, help="Cutoff date in YYYYMMDD format")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the filtered annotations with reference years")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logging.basicConfig(level=logging.INFO)
    
    # Step 1: Read and filter GAF annotations
    logging.info("Reading and filtering GAF annotations...")
    selected_codes = filter_evidence_codes(GO_CODES)
    annotation_df = read_gaf(args.gaf_file, selected_codes, args.cutoff_date)
    
    # Step 2: Retrieve publication years for references
    logging.info("Retrieving publication years for references...")
    unique_pmids = annotation_df['reference'].unique().tolist()
    pmid_to_date, pmid_to_date_edate = retrieve_reference_years(unique_pmids)
    
    # Step 3: Add publication years to the DataFrame
    annotation_df['publication_date'] = annotation_df['reference'].map(pmid_to_date)
    annotation_df['publication_edate'] = annotation_df['reference'].apply(lambda ref: pmid_to_date_edate.get(ref, [None, None])[1])  # Get the e-publication date if available
    
    # Step 4: Filter annotations to only include those with publication dates after the cutoff
    annotation_df['publication_date'] = pd.to_numeric(annotation_df['publication_date'], errors='coerce')
    annotation_df['publication_edate'] = pd.to_numeric(annotation_df['publication_edate'], errors='coerce')
    filtered_df = annotation_df[annotation_df['publication_date'] >= args.cutoff_date]
    print(f"Number of annotations after filtering by publication date: {len(filtered_df)}")
    filtered_df2 = filtered_df[filtered_df['publication_edate'] >= args.cutoff_date]
    print(f"Number of annotations after filtering by e-publication date: {len(filtered_df2)}")
    
    # Step 5: Save the results
    logging.info(f"Saving results to {args.output_file} with less strict cutoff...")
    filtered_df = filtered_df.sort_values(by=['reference'])
    filtered_df.to_csv(args.output_file, index=False,sep="\t")
    
if __name__ == "__main__":
    main()