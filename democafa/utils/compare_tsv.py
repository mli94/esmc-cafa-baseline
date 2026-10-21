#!/usr/bin/env python3

"""
Compare two 3-column TSV files and write row-level differences to an output TSV.

Each input file must have the header:
    EntryID    term    aspect

The output file contains unmatched rows from either file with an additional
`difference_type` column indicating whether the row appears only in `file1`
or only in `file2`.
"""

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, TextIO, Tuple, Union


EXPECTED_HEADER = ["EntryID", "term", "aspect"]
OUTPUT_HEADER = ["difference_type", "EntryID", "term", "aspect"]
Row = Tuple[str, str, str]


def open_text_file(path: Union[str, Path]) -> TextIO:
    """Open a plain text or gzipped TSV file in text mode."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def read_tsv_rows(path: Union[str, Path]) -> Iterator[Row]:
    """Yield rows from a TSV after validating its header and row shape."""
    with open_text_file(path) as handle:
        reader = csv.reader(handle, delimiter="\t")

        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc

        if header != EXPECTED_HEADER:
            raise ValueError(
                f"{path} has header {header}, expected {EXPECTED_HEADER}."
            )

        for line_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError(
                    f"{path} line {line_number} has {len(row)} columns, expected 3."
                )
            yield tuple(row)


def count_rows(path: Union[str, Path]) -> Counter[Row]:
    """Return row counts so duplicate rows are compared correctly."""
    return Counter(read_tsv_rows(path))


def expand_difference_rows(
    left_counts: Counter[Row],
    right_counts: Counter[Row],
    difference_type: str,
) -> Iterable[Tuple[str, str, str, str]]:
    """Yield unmatched rows, repeating rows when duplicate counts differ."""
    for row in sorted(left_counts):
        extra_count = left_counts[row] - right_counts.get(row, 0)
        for _ in range(max(0, extra_count)):
            yield (difference_type, *row)


def expand_common_rows(
    file1_counts: Counter[Row], file2_counts: Counter[Row]
) -> Iterable[Row]:
    """Yield shared rows, repeating rows by the smaller duplicate count."""
    for row in sorted(file1_counts):
        common_count = min(file1_counts[row], file2_counts.get(row, 0))
        for _ in range(common_count):
            yield row


def write_common(
    file1_counts: Counter[Row], file2_counts: Counter[Row], output_file: Union[str, Path]
) -> int:
    """Write shared rows between two TSVs and return the number written."""
    output_path = Path(output_file)
    common_rows = list(expand_common_rows(file1_counts, file2_counts))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(EXPECTED_HEADER)
        writer.writerows(common_rows)

    print(f"Rows shared by both files: {len(common_rows):,}")
    print(f"Wrote shared rows to {output_path}")
    return len(common_rows)


def write_differences(
    file1: Union[str, Path],
    file2: Union[str, Path],
    output_file: Union[str, Path],
    common_output_file: Union[str, Path, None] = None,
) -> None:
    """Compare two TSVs and write the unmatched rows to a diff TSV."""
    file1_counts = count_rows(file1)
    file2_counts = count_rows(file2)
    output_path = Path(output_file)

    file1_only_rows = list(
        expand_difference_rows(file1_counts, file2_counts, "file1_only")
    )
    file2_only_rows = list(
        expand_difference_rows(file2_counts, file1_counts, "file2_only")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(file1_only_rows)
        writer.writerows(file2_only_rows)

    print(f"File 1 total rows: {sum(file1_counts.values()):,}")
    print(f"File 1 unique rows: {len(file1_counts):,}")
    print(f"File 2 total rows: {sum(file2_counts.values()):,}")
    print(f"File 2 unique rows: {len(file2_counts):,}")
    print(f"Rows only in file 1: {len(file1_only_rows):,}")
    print(f"Rows only in file 2: {len(file2_only_rows):,}")
    print(f"Wrote differences to {output_path}")

    if common_output_file is not None:
        write_common(file1_counts, file2_counts, common_output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two 3-column TSV files and write row-level differences."
    )
    parser.add_argument("file1", help="Path to the first TSV file")
    parser.add_argument("file2", help="Path to the second TSV file")
    parser.add_argument("output_file", help="Path to the output TSV file")
    parser.add_argument(
        "--common",
        dest="common_output_file",
        help="Optional path to write rows shared by both files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_differences(
        args.file1, args.file2, args.output_file, args.common_output_file
    )


if __name__ == "__main__":
    main()
