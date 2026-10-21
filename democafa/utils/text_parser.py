#!/usr/bin/env python3
"""
Utility functions for parsing text entries from CAFA6/Kaggle submissions
Takes in a folder of TSV/CSV files and outputs two files with old name as prefix: *_text.tsv and *_go.tsv
The *_text.tsv file contains the text entries for each protein, and the *_go.tsv file contains the GO annotations for each protein
"""

import gzip
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import pandas as pd


def setup_logging(log_level: str = "INFO", log_file: str | None = "text_parser.log"):
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)

def parse_go_and_text(handle):
    for ln in handle:
        if isinstance(ln, bytes):
            ln = ln.decode("utf-8", errors="ignore")
        line = ln.rstrip("\n\r")
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t", 3)
        elif "," in line:
            parts = line.split(",", 3)
        else:
            parts = line.split(None, 3)
        if len(parts) < 3:
            continue

        protein = parts[0].strip()
        term = parts[1].strip()
        score_s = parts[2].strip()

        # Text records
        if not term.startswith("GO:"):
            if term.startswith("Text"):
                if len(parts) < 4:
                    continue
                text = parts[3].strip()
                yield True, protein, text, score_s
            continue

        # GO records
        if "," in score_s:
            score_s = score_s.replace(",", ".")
        try:
            score = float(score_s)
        except ValueError:
            continue
        if not (0.0 < score <= 1.0):
            continue
        yield False, protein, term, score
        
def build_two_tarballs_streaming(
    in_tar_gz: str,
    out_go_tar_gz: str,
    out_text_tar_gz: str,
    subdir_prefix: str = "selected_submissions/",
    spool_max_mb: int = 64,          # per-file buffering cap in memory
    log_every: int = 100,
    logger: logging.Logger | None = None,
):
    logger = logger or logging.getLogger(__name__)
    spool_max_bytes = spool_max_mb * 1024 * 1024
    files_seen = 0
    files_processed = 0
    files_with_go = 0
    files_with_text = 0
    files_failed = 0
    skipped_non_file = 0
    skipped_prefix = 0
    skipped_extension = 0
    skipped_unreadable = 0

    logger.info(
        "Starting tar stream parse: in=%s out_go=%s out_text=%s subdir_prefix=%s spool_max_mb=%s",
        in_tar_gz,
        out_go_tar_gz,
        out_text_tar_gz,
        subdir_prefix,
        spool_max_mb,
    )
    valid_exts = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".raw") # extra submissions with .raw extension (2026-03-26 update)
    # Read input as a true stream
    with tarfile.open(in_tar_gz, mode="r|gz") as tin, \
         tarfile.open(out_go_tar_gz, mode="w|gz") as tgo, \
         tarfile.open(out_text_tar_gz, mode="w|gz") as ttext:

        for m in tin:
            files_seen += 1
            if not m.isfile():
                skipped_non_file += 1
                continue
            if subdir_prefix and not m.name.startswith(subdir_prefix):
                skipped_prefix += 1
                continue

            base = os.path.basename(m.name)
            if not (base.endswith(valid_exts)):
                skipped_extension += 1
                continue

            f = tin.extractfile(m)
            if f is None:
                skipped_unreadable += 1
                logger.warning("Skipping unreadable tar member: %s", m.name)
                continue

            raw = None
            go_spool = None
            text_spool = None
            try:
                # If the file inside the tar is itself gzipped, decompress it
                raw = gzip.GzipFile(fileobj=f, mode="rb") if base.endswith(".gz") else f

                # Determine output member names
                base_no_gz = base[:-3] if base.endswith(".gz") else base
                stem = base_no_gz[:-4] if base_no_gz.endswith(valid_exts) else base_no_gz

                go_member_name = f"selected_submissions_go/{stem}_go.tsv"
                text_member_name = f"selected_submissions_text/{stem}_text.tsv"

                # Spool per-file outputs so we know size before adding to tar
                go_spool = tempfile.SpooledTemporaryFile(max_size=spool_max_bytes, mode="w+b")
                text_spool = tempfile.SpooledTemporaryFile(max_size=spool_max_bytes, mode="w+b")

                go_written = 0
                text_written = 0

                for is_text, protein, term_or_text, score in parse_go_and_text(raw):
                    if is_text:
                        line = f"{protein}\tText\t{score}\t{term_or_text}\n".encode("utf-8")
                        text_spool.write(line)
                        text_written += len(line)
                    else:
                        line = f"{protein}\t{term_or_text}\t{score}\n".encode("utf-8")
                        go_spool.write(line)
                        go_written += len(line)

                # Add non-empty outputs as tar members (plain TSV inside tar)
                if go_written:
                    go_spool.seek(0)
                    ti = tarfile.TarInfo(name=go_member_name)
                    ti.size = go_written
                    ti.mtime = m.mtime
                    tgo.addfile(ti, fileobj=go_spool)
                    files_with_go += 1

                if text_written:
                    text_spool.seek(0)
                    ti = tarfile.TarInfo(name=text_member_name)
                    ti.size = text_written
                    ti.mtime = m.mtime
                    ttext.addfile(ti, fileobj=text_spool)
                    files_with_text += 1

                files_processed += 1
                if files_processed % max(1, log_every) == 0:
                    logger.info(
                        "Progress: processed=%s files_with_go=%s files_with_text=%s failed=%s",
                        files_processed,
                        files_with_go,
                        files_with_text,
                        files_failed,
                    )
            except Exception:
                files_failed += 1
                logger.exception("Failed to process member: %s", m.name)
            finally:
                if go_spool is not None:
                    go_spool.close()
                if text_spool is not None:
                    text_spool.close()
                if raw is not None:
                    try:
                        raw.close()
                    except Exception:
                        pass

    logger.info(
        "Finished tar stream parse: seen=%s processed=%s files_with_go=%s files_with_text=%s failed=%s "
        "skipped_non_file=%s skipped_prefix=%s skipped_extension=%s skipped_unreadable=%s",
        files_seen,
        files_processed,
        files_with_go,
        files_with_text,
        files_failed,
        skipped_non_file,
        skipped_prefix,
        skipped_extension,
        skipped_unreadable,
    )

    return {
        "files_seen": files_seen,
        "files_processed": files_processed,
        "files_with_go": files_with_go,
        "files_with_text": files_with_text,
        "files_failed": files_failed,
        "skipped_non_file": skipped_non_file,
        "skipped_prefix": skipped_prefix,
        "skipped_extension": skipped_extension,
        "skipped_unreadable": skipped_unreadable,
    }


def _process_one_tar_zst_shard(
    shard_path: str,
    out_go_dir: str,
    out_text_dir: str,
    subdir_prefix: str = "selected_submissions/",
    overwrite: bool = False,
    zstd_bin: str = "zstd",
):
    shard = Path(shard_path)
    go_root = Path(out_go_dir)
    text_root = Path(out_text_dir)
    go_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "shard": str(shard),
        "files_seen": 0,
        "files_processed": 0,
        "files_with_go": 0,
        "files_with_text": 0,
        "files_failed": 0,
        "files_skipped_existing": 0,
        "skipped_non_file": 0,
        "skipped_prefix": 0,
        "skipped_extension": 0,
        "skipped_unreadable": 0,
    }

    proc = subprocess.Popen(
        [zstd_bin, "-dc", str(shard)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None:
        raise RuntimeError(f"Failed to read decompressed stream from shard: {shard}")

    valid_exts = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".raw") # extra submissions with .raw extension (2026-03-26 update)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tin:
            for m in tin:
                stats["files_seen"] += 1
                if not m.isfile():
                    stats["skipped_non_file"] += 1
                    continue
                if subdir_prefix and not m.name.startswith(subdir_prefix):
                    stats["skipped_prefix"] += 1
                    continue

                base = os.path.basename(m.name)
                if not (base.endswith(valid_exts)):
                    stats["skipped_extension"] += 1
                    continue

                in_member = tin.extractfile(m)
                if in_member is None:
                    stats["skipped_unreadable"] += 1
                    continue

                rel_name = m.name[len(subdir_prefix):] if subdir_prefix else m.name
                rel_path = Path(rel_name)
                out_name = rel_path.name[:-3] if rel_path.name.endswith(".gz") else rel_path.name

                go_path = go_root / rel_path.parent / out_name
                text_path = text_root / rel_path.parent / out_name
                go_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.parent.mkdir(parents=True, exist_ok=True)

                if not overwrite and go_path.exists() and text_path.exists():
                    stats["files_skipped_existing"] += 1
                    continue

                raw = None
                go_tmp = None
                text_tmp = None
                try:
                    raw = gzip.GzipFile(fileobj=in_member, mode="rb") if base.endswith(".gz") else in_member

                    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False, dir=str(go_path.parent)) as go_fh, \
                         tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False, dir=str(text_path.parent)) as text_fh:

                        go_tmp = Path(go_fh.name)
                        text_tmp = Path(text_fh.name)
                        go_writer = csv.writer(go_fh)
                        text_writer = csv.writer(text_fh)

                        go_rows = 0
                        text_rows = 0
                        for is_text, protein, term_or_text, score in parse_go_and_text(raw):
                            if is_text:
                                text_writer.writerow([protein, "Text", score, term_or_text])
                                text_rows += 1
                            else:
                                go_writer.writerow([protein, term_or_text, score])
                                go_rows += 1

                    os.replace(go_tmp, go_path)
                    if text_rows > 0:
                        os.replace(text_tmp, text_path)
                        stats["files_with_text"] += 1
                    else:
                        if text_tmp is not None and text_tmp.exists():
                            text_tmp.unlink()
                        if overwrite and text_path.exists():
                            text_path.unlink()

                    if go_rows > 0:
                        stats["files_with_go"] += 1
                    stats["files_processed"] += 1
                except Exception:
                    stats["files_failed"] += 1
                    for tmp in (go_tmp, text_tmp):
                        try:
                            if tmp is not None:
                                tmp.unlink()
                        except FileNotFoundError:
                            pass
                finally:
                    if raw is not None:
                        try:
                            raw.close()
                        except Exception:
                            pass
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="ignore")
            proc.stderr.close()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"zstd failed for shard {shard} with exit code {rc}. stderr: {stderr_text.strip()}"
            )

    return stats


def build_two_folders_from_tar_zst_shards(
    input_shards_dir: str,
    out_go_dir: str,
    out_text_dir: str,
    shard_glob: str = "shard_*.tar.zst",
    workers: int = 1,
    overwrite: bool = False,
    smoke_test: int | None = None,
    subdir_prefix: str = "selected_submissions/",
    log_every: int = 1,
    zstd_bin: str = "zstd",
    logger: logging.Logger | None = None,
):
    logger = logger or logging.getLogger(__name__)

    if shutil.which(zstd_bin) is None:
        raise RuntimeError(f"zstd executable not found: {zstd_bin}")

    shards_root = Path(input_shards_dir)
    if not shards_root.is_dir():
        raise ValueError(f"input_shards_dir is not a directory: {input_shards_dir}")

    shards = sorted(p for p in shards_root.glob(shard_glob) if p.is_file())
    if smoke_test is not None:
        if smoke_test < 1:
            raise ValueError("smoke_test must be >= 1 when provided")
        shards = shards[:smoke_test]
        logger.info("Smoke test enabled: limiting shard run to %s shard(s)", len(shards))

    if not shards:
        raise ValueError(f"No shard files matched pattern '{shard_glob}' in {input_shards_dir}")

    logger.info(
        "Starting shard parse: input_shards_dir=%s shard_glob=%s out_go_dir=%s out_text_dir=%s workers=%s shards=%s",
        input_shards_dir,
        shard_glob,
        out_go_dir,
        out_text_dir,
        workers,
        len(shards),
    )

    totals = {
        "shards_seen": len(shards),
        "shards_processed": 0,
        "shards_failed": 0,
        "files_seen": 0,
        "files_processed": 0,
        "files_with_go": 0,
        "files_with_text": 0,
        "files_failed": 0,
        "files_skipped_existing": 0,
        "skipped_non_file": 0,
        "skipped_prefix": 0,
        "skipped_extension": 0,
        "skipped_unreadable": 0,
    }

    def add_shard_stats(shard_stats):
        totals["shards_processed"] += 1
        for key in (
            "files_seen",
            "files_processed",
            "files_with_go",
            "files_with_text",
            "files_failed",
            "files_skipped_existing",
            "skipped_non_file",
            "skipped_prefix",
            "skipped_extension",
            "skipped_unreadable",
        ):
            totals[key] += shard_stats[key]

    if workers <= 1:
        for idx, shard in enumerate(shards, start=1):
            try:
                shard_stats = _process_one_tar_zst_shard(
                    shard_path=str(shard),
                    out_go_dir=out_go_dir,
                    out_text_dir=out_text_dir,
                    subdir_prefix=subdir_prefix,
                    overwrite=overwrite,
                    zstd_bin=zstd_bin,
                )
                add_shard_stats(shard_stats)
            except Exception:
                totals["shards_failed"] += 1
                logger.exception("Failed to process shard: %s", shard)

            if idx % max(1, log_every) == 0:
                logger.info(
                    "Shard progress: done=%s/%s shards_failed=%s files_processed=%s files_failed=%s",
                    idx,
                    len(shards),
                    totals["shards_failed"],
                    totals["files_processed"],
                    totals["files_failed"],
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_shard = {
                executor.submit(
                    _process_one_tar_zst_shard,
                    str(shard),
                    out_go_dir,
                    out_text_dir,
                    subdir_prefix,
                    overwrite,
                    zstd_bin,
                ): shard
                for shard in shards
            }

            for idx, fut in enumerate(as_completed(future_to_shard), start=1):
                shard = future_to_shard[fut]
                try:
                    shard_stats = fut.result()
                    add_shard_stats(shard_stats)
                except Exception:
                    totals["shards_failed"] += 1
                    logger.exception("Failed to process shard: %s", shard)

                if idx % max(1, log_every) == 0:
                    logger.info(
                        "Shard progress: done=%s/%s shards_failed=%s files_processed=%s files_failed=%s",
                        idx,
                        len(shards),
                        totals["shards_failed"],
                        totals["files_processed"],
                        totals["files_failed"],
                    )

    logger.info(
        "Finished shard parse: shards_seen=%s shards_processed=%s shards_failed=%s files_seen=%s files_processed=%s "
        "files_with_go=%s files_with_text=%s files_failed=%s files_skipped_existing=%s skipped_non_file=%s "
        "skipped_prefix=%s skipped_extension=%s skipped_unreadable=%s",
        totals["shards_seen"],
        totals["shards_processed"],
        totals["shards_failed"],
        totals["files_seen"],
        totals["files_processed"],
        totals["files_with_go"],
        totals["files_with_text"],
        totals["files_failed"],
        totals["files_skipped_existing"],
        totals["skipped_non_file"],
        totals["skipped_prefix"],
        totals["skipped_extension"],
        totals["skipped_unreadable"],
    )
    return totals


def _process_one_submission_file(
    src_path: str,
    input_root: str,
    out_go_dir: str,
    out_text_dir: str,
    overwrite: bool = False,
):
    src = Path(src_path)
    root = Path(input_root)
    rel = src.relative_to(root)

    out_name = rel.name[:-3] if rel.name.endswith(".gz") else rel.name
    go_path = Path(out_go_dir) / rel.parent / out_name
    text_path = Path(out_text_dir) / rel.parent / out_name

    go_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and go_path.exists() and text_path.exists():
        return {
            "status": "skipped_existing",
            "go_rows": 0,
            "text_rows": 0,
            "file": str(src),
        }

    go_tmp = go_path.with_name(go_path.name + ".tmp")
    text_tmp = text_path.with_name(text_path.name + ".tmp")

    go_rows = 0
    text_rows = 0
    open_in = gzip.open if src.name.endswith(".gz") else open

    try:
        with open_in(src, mode="rb") as in_fh, \
             open(go_tmp, "w", newline="", encoding="utf-8") as go_fh, \
             open(text_tmp, "w", newline="", encoding="utf-8") as text_fh:

            go_writer = csv.writer(go_fh)
            text_writer = csv.writer(text_fh)

            for is_text, protein, term_or_text, score in parse_go_and_text(in_fh):
                if is_text:
                    text_writer.writerow([protein, "Text", score, term_or_text])
                    text_rows += 1
                else:
                    go_writer.writerow([protein, term_or_text, score])
                    go_rows += 1

        os.replace(go_tmp, go_path)
        if text_rows > 0:
            os.replace(text_tmp, text_path)
        else:
            try:
                text_tmp.unlink()
            except FileNotFoundError:
                pass
            if overwrite and text_path.exists():
                text_path.unlink()
    except Exception:
        for tmp in (go_tmp, text_tmp):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        raise

    return {
        "status": "processed",
        "go_rows": go_rows,
        "text_rows": text_rows,
        "file": str(src),
    }


def build_two_folders_streaming(
    input_dir: str,
    out_go_dir: str,
    out_text_dir: str,
    recursive: bool = False,
    workers: int = 1,
    overwrite: bool = False,
    smoke_test: int | None = None,
    log_every: int = 100,
    logger: logging.Logger | None = None,
):
    logger = logger or logging.getLogger(__name__)

    input_root = Path(input_dir)
    if not input_root.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")

    go_root = Path(out_go_dir)
    text_root = Path(out_text_dir)
    go_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "files_seen": 0,
        "files_candidates": 0,
        "files_processed": 0,
        "files_failed": 0,
        "files_with_go": 0,
        "files_with_text": 0,
        "files_skipped_existing": 0,
        "skipped_non_file": 0,
        "skipped_extension": 0,
    }

    valid_exts = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".raw") # extra submissions with .raw extension (2026-03-26 update)
    iterator = input_root.rglob("*") if recursive else input_root.iterdir()
    candidates = []

    for p in iterator:
        stats["files_seen"] += 1
        if not p.is_file():
            stats["skipped_non_file"] += 1
            continue
        if not p.name.endswith(valid_exts):
            stats["skipped_extension"] += 1
            continue
        candidates.append(p)

    stats["files_candidates"] = len(candidates)

    if smoke_test is not None:
        if smoke_test < 1:
            raise ValueError("smoke_test must be >= 1 when provided")
        candidates = candidates[:smoke_test]
        logger.info("Smoke test enabled: limiting folder run to %s files", len(candidates))

    logger.info(
        "Starting folder parse: input_dir=%s out_go_dir=%s out_text_dir=%s recursive=%s workers=%s candidates=%s",
        input_dir,
        out_go_dir,
        out_text_dir,
        recursive,
        workers,
        len(candidates),
    )

    def apply_result(res):
        if res["status"] == "skipped_existing":
            stats["files_skipped_existing"] += 1
            return
        stats["files_processed"] += 1
        if res["go_rows"] > 0:
            stats["files_with_go"] += 1
        if res["text_rows"] > 0:
            stats["files_with_text"] += 1

    if workers <= 1:
        for idx, p in enumerate(candidates, start=1):
            try:
                res = _process_one_submission_file(
                    str(p),
                    str(input_root),
                    str(go_root),
                    str(text_root),
                    overwrite,
                )
                apply_result(res)
            except Exception:
                stats["files_failed"] += 1
                logger.exception("Failed to process file: %s", p)

            if idx % max(1, log_every) == 0:
                logger.info(
                    "Progress: done=%s/%s processed=%s failed=%s skipped_existing=%s",
                    idx,
                    len(candidates),
                    stats["files_processed"],
                    stats["files_failed"],
                    stats["files_skipped_existing"],
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(
                    _process_one_submission_file,
                    str(p),
                    str(input_root),
                    str(go_root),
                    str(text_root),
                    overwrite,
                ): p
                for p in candidates
            }

            for idx, fut in enumerate(as_completed(future_to_path), start=1):
                p = future_to_path[fut]
                try:
                    res = fut.result()
                    apply_result(res)
                except Exception:
                    stats["files_failed"] += 1
                    logger.exception("Failed to process file: %s", p)

                if idx % max(1, log_every) == 0:
                    logger.info(
                        "Progress: done=%s/%s processed=%s failed=%s skipped_existing=%s",
                        idx,
                        len(candidates),
                        stats["files_processed"],
                        stats["files_failed"],
                        stats["files_skipped_existing"],
                    )

    logger.info(
        "Finished folder parse: seen=%s candidates=%s processed=%s files_with_go=%s files_with_text=%s failed=%s "
        "skipped_existing=%s skipped_non_file=%s skipped_extension=%s",
        stats["files_seen"],
        stats["files_candidates"],
        stats["files_processed"],
        stats["files_with_go"],
        stats["files_with_text"],
        stats["files_failed"],
        stats["files_skipped_existing"],
        stats["skipped_non_file"],
        stats["skipped_extension"],
    )
    return stats

def create_list_selected_files(metadata_file, output_file):
    """
    Create a list of selected files for download based on the metadata file
    """
    df = pd.read_csv(metadata_file, sep=",",header=0)
    all_teams = df['TeamName'].unique().tolist()
    selected_teams = df[df['IsSelected']]['TeamName'].unique().tolist() # only the ones the teams selected for lb, not top 2/team
    print(f"Max files selected per team: {df[df['IsSelected']].groupby('TeamName').size().max()}")
    # For teams that hand-select their submissions (IsSelected=True), we take all their submissions
    selected_idx = df[(df['IsSelected'])].index.tolist()
    # If the teams only selected one submission, we take another non-selected highest scoring submission from the same team
    for team in all_teams:
        if team in selected_teams:
            team_selected_idx = df[(df['TeamName'] == team) & (df['IsSelected'])].index.tolist()
            if len(team_selected_idx) == 1:
                team_non_selected_idx = df[(df['TeamName'] == team) & (~df['IsSelected'])].sort_values('PrivateScore', ascending=False).index.tolist()
                if team_non_selected_idx:
                    best_non_selected_idx = team_non_selected_idx[0] # already sorted by PrivateScore
                    selected_idx.append(best_non_selected_idx)
        # If the teams did not hand-select any submission, we take their 2 highest scoring submissions
        elif team not in selected_teams:
            team_df = df[(df['TeamName'] == team) & (~df['IsSelected'])].sort_values('PrivateScore', ascending=False).index.tolist()
            if team_df:
                best_2 = team_df[:2] # already sorted by PrivateScore
                selected_idx.extend(best_2)
    selected_files = df.loc[selected_idx, 'SubmissionId'].tolist()
    with open(output_file, 'w') as f:
        for file in selected_files:
            f.write(f"selected_submissions/{file}.csv\n")


def compare_with_kaggle_selected(kaggle_file):
    kaggle = pd.read_csv(kaggle_file, sep=",",header=None)
    kaggle_files = [f.split('.')[0] for f in kaggle[0]] # remove .csv extension
    # Pending newer filtered data from Kaggle, which should match the new lb update
    
def rename_new_extension(input_file, old_ext, new_ext):
    """
    Rename files with a new extension (e.g. .raw to .csv)
    """
    input_path = Path(input_file)
    if input_path.suffix == old_ext:
        new_name = input_path.with_suffix(new_ext)
        input_path.rename(new_name)
        return str(new_name)
    else:
        raise ValueError(f"File {input_file} does not have expected extension {old_ext}")
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse CAFA6/Kaggle submissions into separate GO and text outputs from either tar.gz or folder input")
    parser.add_argument("input_tar_gz", nargs="?", help="Input tar.gz file containing submission TSV/CSV files")
    parser.add_argument("output_go_tar_gz", nargs="?", help="Output tar.gz file to write GO annotation TSVs")
    parser.add_argument("output_text_tar_gz", nargs="?", help="Output tar.gz file to write text annotation TSVs")
    parser.add_argument("--input-dir", help="Input folder containing submission CSV/TSV files")
    parser.add_argument("--input-shards-dir", help="Input folder containing shard_XX.tar.zst files")
    parser.add_argument("--output-go-dir", help="Output folder to write GO parsed files")
    parser.add_argument("--output-text-dir", help="Output folder to write text parsed files")
    parser.add_argument("--shard-glob", default="shard_*.tar.zst", help="Glob pattern for shard files inside --input-shards-dir")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir")
    parser.add_argument("--workers", type=int, default=1, help="Folder mode only: number of worker processes (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Folder mode only: overwrite existing parsed outputs")
    parser.add_argument("--smoke-test", type=int, default=None, help="Folder/shard mode only: process only first N files or shards for a quick smoke test")

    parser.add_argument("--subdir-prefix", default="selected_submissions/", help="Only process files in this subdirectory prefix inside the input tar (default: selected_submissions/)")
    parser.add_argument("--spool-max-mb", type=int, default=64, help="Maximum memory to spool per file before writing to disk (default: 64 MB)")
    parser.add_argument("--log-every", type=int, default=1000, help="Log progress every N files processed (default: 1000)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO)")
    parser.add_argument("--log-file", default="text_parser.log", help='Path to log file (default: text_parser.log). Use "none" to disable file logging')
    parser.add_argument("--zstd-bin", default="zstd", help="zstd executable name or absolute path for shard mode (default: zstd)")
    args = parser.parse_args()

    tar_mode = all([args.input_tar_gz, args.output_go_tar_gz, args.output_text_tar_gz])
    folder_mode = all([args.input_dir, args.output_go_dir, args.output_text_dir])
    shard_mode = all([args.input_shards_dir, args.output_go_dir, args.output_text_dir])

    mode_count = sum([tar_mode, folder_mode, shard_mode])

    if mode_count > 1:
        parser.error("Choose exactly one mode: tar.gz positional args, folder mode (--input-dir), or shard mode (--input-shards-dir)")
    if mode_count == 0:
        parser.error("Provide one mode: tar.gz positional args, folder mode args, or shard mode args")

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.smoke_test is not None and args.smoke_test < 1:
        parser.error("--smoke-test must be >= 1")
    if tar_mode and args.smoke_test is not None:
        parser.error("--smoke-test is only supported in folder or shard mode")
    if tar_mode and args.input_shards_dir:
        parser.error("--input-shards-dir cannot be used in tar mode")

    return args

def main():
    args = parse_args()
    log_file = None if str(args.log_file).lower() == "none" else args.log_file
    logger = setup_logging(log_level=args.log_level, log_file=log_file)

    if args.input_dir:
        build_two_folders_streaming(
            input_dir=args.input_dir,
            out_go_dir=args.output_go_dir,
            out_text_dir=args.output_text_dir,
            recursive=args.recursive,
            workers=args.workers,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            log_every=args.log_every,
            logger=logger,
        )
    elif args.input_shards_dir:
        build_two_folders_from_tar_zst_shards(
            input_shards_dir=args.input_shards_dir,
            out_go_dir=args.output_go_dir,
            out_text_dir=args.output_text_dir,
            shard_glob=args.shard_glob,
            workers=args.workers,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            subdir_prefix=args.subdir_prefix,
            log_every=args.log_every,
            zstd_bin=args.zstd_bin,
            logger=logger,
        )
    else:
        build_two_tarballs_streaming(
            in_tar_gz=args.input_tar_gz,
            out_go_tar_gz=args.output_go_tar_gz,
            out_text_tar_gz=args.output_text_tar_gz,
            subdir_prefix=args.subdir_prefix,
            spool_max_mb=args.spool_max_mb,
            log_every=args.log_every,
            logger=logger,
        )
    
if __name__ == "__main__":
    main()
    