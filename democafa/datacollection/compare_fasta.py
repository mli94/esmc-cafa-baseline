#!/usr/bin/env python3
"""
Compare two FASTA files and identify sequences that differ between them.
Optimized for large files (up to 300MB) with hundreds of thousands of sequences.

Usage:
    python compare_fasta.py file1.fasta file2.fasta output_diff.fasta [--chunk-size 10000]

Features:
- Handles gzipped files automatically
- Memory-efficient chunked processing
- Progress tracking
- Fast sequence comparison using dictionaries
- Outputs differences with source file information
"""

import argparse
import gzip
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Tuple, TextIO, Union
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def open_file(filepath: Union[str, Path]) -> TextIO:
    """
    Open a file, automatically detecting if it's gzipped.

    Args:
        filepath: Path to the file (can be .gz or plain text)

    Returns:
        File handle (text mode)
    """
    filepath = Path(filepath)
    if filepath.suffix == '.gz':
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')


def count_sequences(filepath: Union[str, Path]) -> int:
    """
    Count the number of sequences in a FASTA file for progress tracking.

    Args:
        filepath: Path to the FASTA file

    Returns:
        Number of sequences in the file
    """
    count = 0
    with open_file(filepath) as handle:
        for _ in SeqIO.parse(handle, "fasta"):
            count += 1
    return count


def load_sequences_chunked(filepath: Union[str, Path], chunk_size: int = 10000) -> Iterator[Dict[str, str]]:
    """
    Load sequences from FASTA file in chunks to manage memory usage.

    Args:
        filepath: Path to the FASTA file
        chunk_size: Number of sequences to load per chunk

    Yields:
        Dictionary mapping sequence IDs to sequences
    """
    chunk = {}
    with open_file(filepath) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            chunk[record.id] = str(record.seq)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = {}

        # Yield remaining sequences
        if chunk:
            yield chunk


def load_all_sequences(filepath: Union[str, Path], show_progress: bool = True) -> Dict[str, str]:
    """
    Load all sequences from a FASTA file into memory.

    Args:
        filepath: Path to the FASTA file
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}

    if show_progress:
        total_seqs = count_sequences(filepath)
        print(f"Loading {total_seqs:,} sequences from {filepath}")

        with open_file(filepath) as handle:
            for record in tqdm(SeqIO.parse(handle, "fasta"), total=total_seqs, desc="Loading"):
                sequences[record.id] = str(record.seq)
    else:
        with open_file(filepath) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences


def compare_sequences_memory_efficient(file1: Union[str, Path], file2: Union[str, Path],
                                     output_file: Union[str, Path], chunk_size: int = 10000):
    """
    Memory-efficient comparison of two FASTA files using chunked processing.

    Args:
        file1: Path to first FASTA file
        file2: Path to second FASTA file
        output_file: Path to output file for differences
        chunk_size: Number of sequences to process per chunk
    """
    print("=== Memory-Efficient Comparison ===")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Chunk size: {chunk_size:,}")

    # Load file2 completely into memory (assuming it's the reference)
    file2_seqs = load_all_sequences(file2, show_progress=True)
    print(f"File 2 loaded: {len(file2_seqs):,} sequences")
    file2_uniprot_ids = {k.split("|")[1] if "|" in k else k: k for k in file2_seqs.keys()}

    # Process file1 in chunks
    total_chunks = (count_sequences(file1) + chunk_size - 1) // chunk_size
    differences = []

    print(f"Processing file 1 in {total_chunks} chunks...")

    for chunk_idx, chunk in enumerate(tqdm(load_sequences_chunked(file1, chunk_size),
                                          total=total_chunks, desc="Processing chunks")):
        # seq_id will be sp|G2X4G0|424Y_VERDV
        # we want to compare by uniprot_id (G2X4G0) only, but note the full id difference
        for seq_id, seq in chunk.items():
            uniprot_id = seq_id.split("|")[1] if "|" in seq_id else seq_id
            # full id is the same
            if seq_id in file2_seqs:
                if seq != file2_seqs[seq_id]:
                    # Sequences differ
                    differences.append((seq_id, seq, file2_seqs[seq_id], 'different', seq_id))
            # uniprot id match but full id different
            elif uniprot_id in file2_uniprot_ids:
                file2_full_id = file2_uniprot_ids[uniprot_id]
                if seq != file2_seqs[file2_full_id]:
                    differences.append((seq_id, seq, file2_seqs[file2_full_id], 'different', file2_full_id))
                else:
                    differences.append((seq_id, seq, file2_seqs[file2_full_id], 'same sequence different gene name', file2_full_id))
            # uniprot id and full id different, Sequence only in file1
            else:
                differences.append((seq_id, seq, None, 'file1_only', None))

    # Find sequences only in file2
    file1_ids = set()
    file1_uniprot_ids = set()
    for chunk in load_sequences_chunked(file1, chunk_size):
        file1_ids.update(chunk.keys())
        file1_uniprot_ids.update({k.split("|")[1] if "|" in k else k for k in chunk.keys()})

    for seq_id, seq in file2_seqs.items():
        uniprot_id = seq_id.split("|")[1] if "|" in seq_id else seq_id
        # Only add to differences if both full ID and UniProt ID are not in file1
        if seq_id not in file1_ids and uniprot_id not in file1_uniprot_ids:
            differences.append((seq_id, None, seq, 'file2_only', None))

    # Write differences to output file
    write_differences(differences, output_file, file1, file2)


def compare_sequences_fast(file1: Union[str, Path], file2: Union[str, Path],
                          output_file: Union[str, Path]):
    """
    Fast comparison by loading both files into memory (use when memory allows).

    Args:
        file1: Path to first FASTA file
        file2: Path to second FASTA file
        output_file: Path to output file for differences
    """
    print("=== Fast In-Memory Comparison ===")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")

    # Load both files
    file1_seqs = load_all_sequences(file1, show_progress=True)
    file2_seqs = load_all_sequences(file2, show_progress=True)

    print(f"File 1: {len(file1_seqs):,} sequences")
    print(f"File 2: {len(file2_seqs):,} sequences")

    # Find differences
    print("Finding differences...")
    differences = []
    common = []

    # Create uniprot ID mapping for file2
    file2_uniprot_ids = {k.split("|")[1] if "|" in k else k: k for k in file2_seqs.keys()}

    # Check sequences in file1
    for seq_id in tqdm(file1_seqs.keys(), desc="Comparing file1"):
        uniprot_id = seq_id.split("|")[1] if "|" in seq_id else seq_id

        # Full id is the same
        if seq_id in file2_seqs:
            if file1_seqs[seq_id] != file2_seqs[seq_id]:
                # Sequences differ
                differences.append((seq_id, file1_seqs[seq_id], file2_seqs[seq_id], 'different', seq_id))
            else:
                # Otherwise, sequence is common (same id and same sequence)
                common.append((seq_id, file1_seqs[seq_id]))
        # Uniprot id match but full id different
        elif uniprot_id in file2_uniprot_ids:
            file2_full_id = file2_uniprot_ids[uniprot_id]
            if file1_seqs[seq_id] != file2_seqs[file2_full_id]:
                differences.append((seq_id, file1_seqs[seq_id], file2_seqs[file2_full_id], 'different', file2_full_id))
            else:
                # Same sequence, different gene name, should be in common file with id from file1
                differences.append((seq_id, file1_seqs[seq_id], file2_seqs[file2_full_id], 'same sequence different gene name', file2_full_id))
                common.append((seq_id, file1_seqs[seq_id]))
        # Uniprot id and full id different, Sequence only in file1
        elif uniprot_id not in file2_uniprot_ids:
            differences.append((seq_id, file1_seqs[seq_id], None, 'file1_only', None))


    # Check sequences only in file2
    file1_uniprot_ids = {k.split("|")[1] if "|" in k else k for k in file1_seqs.keys()}
    for seq_id in file2_seqs.keys():
        uniprot_id = seq_id.split("|")[1] if "|" in seq_id else seq_id
        # Only add to differences if both full ID and UniProt ID are not in file1
        if seq_id not in file1_seqs and uniprot_id not in file1_uniprot_ids:
            differences.append((seq_id, None, file2_seqs[seq_id], 'file2_only', None))

    # Write common sequences (that are not in differences)
    output_path = Path(output_file)
    common_output_file = output_path.with_name(f"{output_path.stem}_common.fasta")
    if common_output_file.parent != Path('.'):
        common_output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(common):,} common sequences to {common_output_file}")
    with open(common_output_file, 'w') as handle:
        SeqIO.write([SeqRecord(seq=Seq(seq), id=seq_id, description="") for seq_id, seq in common], handle, "fasta")

    # Write differences
    write_differences(differences, output_file, file1, file2)


def write_differences(differences: list, output_file: Union[str, Path],
                     file1: Union[str, Path], file2: Union[str, Path]):
    """
    Write sequence differences to output FASTA file.

    Args:
        differences: List of tuples (seq_id, seq1, seq2, diff_type)
        output_file: Path to output file
        file1: Path to first input file (for headers)
        file2: Path to second input file (for headers)
    """
    print(f"Writing {len(differences):,} differences to {output_file}")

    records_to_write = []

    for seq_id, seq1, seq2, diff_type, file2_id in differences:
        if diff_type == 'different':
            # Write both versions
            record1 = SeqRecord(
                seq=Seq(seq1),
                id=f"{seq_id}_file1",
                description=f"from {Path(file1).name} - DIFFERENT from file2"
            )
            # Use the actual file2 ID if available, otherwise use seq_id
            f2_id = file2_id if file2_id else seq_id
            record2 = SeqRecord(
                seq=Seq(seq2),
                id=f"{f2_id}_file2",
                description=f"from {Path(file2).name} - DIFFERENT from file1"
            )
            records_to_write.extend([record1, record2])

        elif diff_type == 'same sequence different gene name':
            # Write both versions to show gene name differences
            record1 = SeqRecord(
                seq=Seq(seq1),
                id=f"{seq_id}_file1_{file2_id}_file2",
                description=f"from {Path(file1).name} - SAME sequence but DIFFERENT gene name from file2"
            )
            f2_id = file2_id if file2_id else seq_id
            record2 = SeqRecord(
                seq=Seq(seq2),
                id=f"{f2_id}_file2",
                description=f"from {Path(file2).name} - SAME sequence but DIFFERENT gene name from file1"
            )
            records_to_write.extend([record1, record2])

        elif diff_type == 'file1_only':
            record = SeqRecord(
                seq=Seq(seq1),
                id=f"{seq_id}_file1_only",
                description=f"from {Path(file1).name} - NOT in file2"
            )
            records_to_write.append(record)

        elif diff_type == 'file2_only':
            record = SeqRecord(
                seq=Seq(seq2),
                id=f"{seq_id}_file2_only",
                description=f"from {Path(file2).name} - NOT in file1"
            )
            records_to_write.append(record)

    # Write to output file
    output_path = Path(output_file)
    if output_path.parent != Path('.'):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as handle:
        SeqIO.write(records_to_write, handle, "fasta")

    print("Summary:")
    print(f"  - Same sequence, different gene name: {sum(1 for _, _, _, t, _ in differences if t == 'same sequence different gene name')}")
    print(f"  - Different sequences: {sum(1 for _, _, _, t, _ in differences if t == 'different')}")
    print(f"  - Only in file1: {sum(1 for _, _, _, t, _ in differences if t == 'file1_only')}")
    print(f"  - Only in file2: {sum(1 for _, _, _, t, _ in differences if t == 'file2_only')}")
    print(f"  - Total output records: {len(records_to_write)}")


def estimate_memory_usage(filepath: Union[str, Path]) -> Tuple[int, float]:
    """
    Estimate memory usage for loading a FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        Tuple of (sequence_count, estimated_memory_gb)
    """
    sample_size = 1000
    total_chars = 0
    seq_count = 0

    with open_file(filepath) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            total_chars += len(record.id) + len(record.seq)
            seq_count += 1
            if seq_count >= sample_size:
                break

    if seq_count == 0:
        return 0, 0.0

    # Estimate total sequences
    total_seqs = count_sequences(filepath)
    avg_chars_per_seq = total_chars / seq_count
    estimated_total_chars = total_seqs * avg_chars_per_seq

    # Rough estimate: each character takes ~1 byte, plus Python overhead (~3x)
    estimated_memory_gb = (estimated_total_chars * 3) / (1024**3)

    return total_seqs, estimated_memory_gb

def parse_inputs(args):
    parser = argparse.ArgumentParser(
        description="Compare two FASTA files and identify sequence differences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_fasta.py file1.fasta file2.fasta differences.fasta
  python compare_fasta.py file1.fasta.gz file2.fasta.gz diff.fasta --chunk-size 5000
  python compare_fasta.py file1.fasta file2.fasta diff.fasta --memory-efficient
        """
    )

    parser.add_argument("file1", help="First FASTA file (can be gzipped)")
    parser.add_argument("file2", help="Second FASTA file (can be gzipped)")
    parser.add_argument("output", help="Output FASTA file for differences")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Chunk size for memory-efficient mode (default: 10000)")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Force memory-efficient mode")
    parser.add_argument("--estimate-memory", action="store_true",
                       help="Only estimate memory usage and exit")

    args = parser.parse_args(args)
    return args

def main():
    args = parse_inputs(sys.argv[1:])

    # Check if files exist
    for filepath in [args.file1, args.file2]:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

    # Estimate memory usage
    print("Estimating memory requirements...")
    file1_seqs, file1_mem = estimate_memory_usage(args.file1)
    file2_seqs, file2_mem = estimate_memory_usage(args.file2)
    total_mem = file1_mem + file2_mem

    print(f"File 1: {file1_seqs:,} sequences, ~{file1_mem:.2f} GB memory")
    print(f"File 2: {file2_seqs:,} sequences, ~{file2_mem:.2f} GB memory")
    print(f"Total estimated memory: ~{total_mem:.2f} GB")

    if args.estimate_memory:
        sys.exit(0)

    # Choose comparison method
    if args.memory_efficient or total_mem > 8.0:  # Use memory-efficient mode if > 8GB
        if total_mem > 8.0 and not args.memory_efficient:
            print(f"Auto-selecting memory-efficient mode (estimated memory: {total_mem:.2f} GB)")

        start_time = time.time()
        compare_sequences_memory_efficient(args.file1, args.file2, args.output, args.chunk_size)
        end_time = time.time()
    else:
        print("Using fast in-memory comparison")
        start_time = time.time()
        compare_sequences_fast(args.file1, args.file2, args.output)
        end_time = time.time()

    print(f"Comparison completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
