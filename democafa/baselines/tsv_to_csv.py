#!/usr/bin/env python3
import sys
import csv
import multiprocessing as mp
from tqdm import tqdm


def process_chunk(chunk_data):
    """Process a chunk of TSV lines and convert to CSV format."""
    lines,output_delim = chunk_data
    csv_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            csv_lines.append(output_delim.join(parts[:3]) + '\n')
    
    return csv_lines


def tsv_to_csv(input_file, output_file, num_processes=None, chunk_size=100000):
    """Convert TSV to CSV using multiprocessing."""
    if num_processes is None:
        num_processes = min(mp.cpu_count() - 1, 8)
    
    print(f"Converting {input_file} to {output_file}")
    print(f"Using {num_processes} processes, chunk size {chunk_size:,}")
    
    # Count lines
    print("Counting lines...")
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines:,}")
    
    # Read and chunk
    chunks = []
    current_chunk = []
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Reading file"):
            current_chunk.append(line)
            
            if len(current_chunk) >= chunk_size:
                chunks.append((current_chunk, ','))
                current_chunk = []
        
        if current_chunk:
            chunks.append((current_chunk, ','))
    
    print(f"Created {len(chunks)} chunks")
    
    # Process in parallel
    print("Processing chunks...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), 
                           total=len(chunks), 
                           desc="Converting"))
    
    # Write output
    print(f"Writing to {output_file}")
    with open(output_file, 'w') as f:
        for chunk_result in tqdm(results, desc="Writing"):
            f.writelines(chunk_result)
    
    print("Completed!")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python tsv_to_csv.py input.tsv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    tsv_to_csv(input_file, output_file)
