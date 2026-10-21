#!/usr/bin/env python3
"""Plot CAFA6 evaluator weighted F-measure results by subset and GO aspect."""

import argparse
import colorsys
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

DEFAULT_SANS_SERIF_FONT = "DejaVu Sans"
PREFERRED_CJK_FONTS = [
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Sans CJK HK",
    "Noto Sans CJK KR",
    "Source Han Sans SC",
    "Source Han Sans TW",
    "SimHei",
]
CJK_FONT_PROPERTIES = None
CJK_UNICODE_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
)


def configure_matplotlib_fonts() -> None:
    """Register conda-installed CJK fonts as fallback for missing glyphs."""
    global CJK_FONT_PROPERTIES

    font_patterns = (
        "*Noto*Sans*CJK*.ttf",
        "*Noto*Sans*CJK*.ttc",
        "*Noto*Sans*CJK*.otf",
        "*NotoSansCJK*.ttf",
        "*NotoSansCJK*.ttc",
        "*NotoSansCJK*.otf",
        "*Source*Han*Sans*.ttf",
        "*Source*Han*Sans*.ttc",
        "*Source*Han*Sans*.otf",
        "*SimHei*.ttf",
    )
    font_dirs = []
    for env_var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX"):
        prefix_value = os.environ.get(env_var)
        if prefix_value:
            prefix = Path(prefix_value)
            font_dirs.extend([prefix / "fonts", prefix / "share" / "fonts"])

    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for pattern in font_patterns:
            for font_path in font_dir.rglob(pattern):
                try:
                    font_manager.fontManager.addfont(str(font_path))
                except RuntimeError:
                    continue

    installed_font_entries = {
        font.name: font
        for font in font_manager.fontManager.ttflist
    }
    installed_fonts = set(installed_font_entries)
    available_cjk_fonts = [font for font in PREFERRED_CJK_FONTS if font in installed_fonts]
    if available_cjk_fonts:
        CJK_FONT_PROPERTIES = font_manager.FontProperties(
            fname=installed_font_entries[available_cjk_fonts[0]].fname,
        )

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [DEFAULT_SANS_SERIF_FONT, *available_cjk_fonts]
    mpl.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


def contains_cjk(text: object) -> bool:
    """Return True when text contains CJK unified ideographs."""
    for char in str(text):
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in CJK_UNICODE_RANGES):
            return True
    return False


def cjk_font_kwargs(text: object) -> dict:
    """Return explicit font kwargs for strings that need CJK glyph support."""
    if CJK_FONT_PROPERTIES is not None and contains_cjk(text):
        return {"fontproperties": CJK_FONT_PROPERTIES}
    return {}


def apply_cjk_font_to_text(text_artist, text: object, fontsize: float | None = None) -> None:
    """Apply the CJK font to a Matplotlib text artist only when needed."""
    if CJK_FONT_PROPERTIES is not None and contains_cjk(text):
        text_artist.set_fontproperties(CJK_FONT_PROPERTIES)
        if fontsize is not None:
            text_artist.set_fontsize(fontsize)

EVALUATION_FILE = "evaluation_best_f_micro.tsv"
EVALUATION_ALL_FILE = "evaluation_all.tsv"
SUBSET_DIRS = {
    "NK": "cafa6_nr_cafaeval_NK",
    "LK": "cafa6_nr_cafaeval_LK",
    "PK": "cafa6_nr_cafaeval_PK",
}
SUBSET_LABELS = {
    "NK": "No knowledge",
    "LK": "Limited knowledge",
    "PK": "Partial knowledge",
}
ASPECTS = {
    "biological_process": "Biological process",
    "cellular_component": "Cellular component",
    "molecular_function": "Molecular function",
}
ONTOLOGY_SHORT_LABELS = {
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "molecular_function": "MFO",
}
AXIS_TITLE_DICT = {
    # "pr": "Precision",
    # "rc": "Recall",
    # "f": "F-score",
    # "pr_w": "Weighted Precision",
    # "rc_w": "Weighted Recall",
    # "f_w": "Weighted F-score",
    # "mi": "Misinformation (Unweighted)",
    # "ru": "Remaining Uncertainty (Unweighted)",
    # "mi_w": "Misinformation",
    # "ru_w": "Remaining Uncertainty",
    # "s": "S-score",
    "pr_micro": "Precision (Micro)",
    "rc_micro": "Recall (Micro)",
    "f_micro": "F-score (Micro)",
    "pr_macro": "Weighted Precision (Macro)",
    "rc_macro": "Weighted Recall (Macro)",
    "f_macro": "Weighted F-score (Macro)",
}
PR_METRIC = "f_micro"
PR_CURVE_COLUMNS = ("rc_micro", "pr_micro")
BASELINE_STYLE_BY_TYPE = {
    "goa_nonexp": {
        "display_label": "GOA Non-exp",
        "color": "#1f4e79",
        "marker": "o",
        "patterns": ("goa non-exp", "goa nonexp", "goa_nonexp", "goa-nonexp", "goa"),
    },
    "prott5": {
        "display_label": "ProtT5",
        "color": "#e68613",
        "marker": "P",
        "patterns": ("prott5", "prot t5", "prot-t5", "prot_t5"),
    },
    "blast": {
        "display_label": "BLAST",
        "color": "#228833",
        "marker": "X",
        "patterns": ("blast",),
    },
    "naive": {
        "display_label": "Naive",
        "color": "#5b2a86",
        "marker": "^",
        "patterns": ("naive",),
    },
}


def strip_prediction_suffix(filename: str) -> str:
    """Return a compact prediction label from a filename/path."""
    path = Path(str(filename)).name
    while Path(path).suffix:
        path = Path(path).stem
    return path


def extract_submission_id(filename: str) -> str:
    """Return the submission ID portion of an evaluator filename."""
    return strip_prediction_suffix(filename)


def read_evaluator_scores(result_file: str | Path, subset: str) -> pd.DataFrame:
    """Read one CAFA-evaluator ``evaluation_best_f_micro.tsv`` file."""
    result_file = Path(result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    df = pd.read_csv(result_file, sep="\t", dtype={"filename": str, "ns": str})
    required = {"filename", "ns", PR_METRIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{result_file} is missing columns: {sorted(missing)}")

    numeric_columns = [PR_METRIC]
    for optional_column in ("cov_w", "cov", "cov_max"):
        if optional_column in df.columns:
            numeric_columns.append(optional_column)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    aggregations = {PR_METRIC: (PR_METRIC, "mean")}
    for optional_column in ("cov_w", "cov", "cov_max"):
        if optional_column in df.columns:
            aggregations[optional_column] = (optional_column, "mean")

    scores = (
        df.groupby(["filename", "ns"], as_index=False)
        .agg(**aggregations)
        .assign(
            subset=subset,
            subset_label=SUBSET_LABELS.get(subset, subset),
            submission_id=lambda frame: frame["filename"].map(extract_submission_id),
            prediction_label=lambda frame: frame["filename"].map(strip_prediction_suffix),
        )
    )
    scores[PR_METRIC] = scores[PR_METRIC].fillna(0.0)
    return scores


def read_evaluator_curves(result_file: str | Path, subset: str) -> pd.DataFrame:
    """Read one CAFA-evaluator ``evaluation_all.tsv`` file for PR curves."""
    result_file = Path(result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    df = pd.read_csv(result_file, sep="\t", dtype={"filename": str, "ns": str})
    required = {"filename", "ns", "tau", PR_METRIC, *PR_CURVE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{result_file} is missing columns: {sorted(missing)}")

    numeric_columns = ["tau", PR_METRIC, *PR_CURVE_COLUMNS]
    if "cov" in df.columns:
        numeric_columns.append("cov")
    if "cov_w" in df.columns:
        numeric_columns.append("cov_w")
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.assign(
        subset=subset,
        subset_label=SUBSET_LABELS.get(subset, subset),
        submission_id=lambda frame: frame["filename"].map(extract_submission_id),
        prediction_label=lambda frame: frame["filename"].map(strip_prediction_suffix),
    )
    return df


def read_score_metadata(metadata_file: str | Path) -> dict[str, str]:
    """Read score metadata CSV and return ``SubmissionId`` to ``TeamName``."""
    metadata = pd.read_csv(metadata_file, dtype={"SubmissionId": str, "TeamName": str})
    required = {"SubmissionId", "TeamName", "PrivateScore"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_file} is missing columns: {sorted(missing)}")

    metadata = metadata.dropna(subset=["SubmissionId"]).copy()
    metadata["SubmissionId"] = metadata["SubmissionId"].astype(str)
    metadata["TeamName"] = metadata["TeamName"].fillna("").astype(str)
    # Remove teams with scores = 0 (valid teams but score = 0, still counted in metadata)
    metadata = metadata[metadata["PrivateScore"] > 0] 
    return metadata.drop_duplicates("SubmissionId", keep="last").set_index("SubmissionId")["TeamName"].to_dict()


def parse_bool(value) -> bool:
    """Parse TSV/CSV boolean-like values."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def method_metadata_key_candidates(filename: str, submission_id: str | None = None) -> list[str]:
    """Return filename/submission keys used to match evaluator rows."""
    candidates = [
        str(filename),
        Path(str(filename)).name,
        strip_prediction_suffix(filename),
    ]
    if submission_id is not None:
        candidates.extend([str(submission_id), Path(str(submission_id)).name])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate and candidate != "nan"))


def read_method_metadata(metadata_file: str | Path) -> dict[str, dict[str, object]]:
    """Read method_names-style TSV and return filename-keyed metadata."""
    metadata = pd.read_csv(metadata_file, sep="\t", dtype=str)
    required = {"filename", "group", "label", "is_baseline"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_file} is missing columns: {sorted(missing)}")

    metadata = metadata.dropna(subset=["filename", "group"]).copy()
    metadata["filename"] = metadata["filename"].astype(str).str.strip()
    metadata["group"] = metadata["group"].fillna("").astype(str).str.strip()
    metadata["label"] = metadata["label"].fillna(metadata["group"]).astype(str).str.strip()
    metadata["is_baseline"] = metadata["is_baseline"].map(parse_bool)
    metadata = metadata[(metadata["filename"] != "") & (metadata["group"] != "")]

    lookup = {}
    for row in metadata.itertuples(index=False):
        entry = {
            "group": row.group,
            "label": row.label or row.group,
            "is_baseline": bool(row.is_baseline),
        }
        for key in method_metadata_key_candidates(row.filename):
            lookup[key] = entry
    return lookup


def lookup_method_metadata(
    metadata_lookup: dict[str, dict[str, object]],
    filename: str,
    submission_id: str | None = None,
) -> dict[str, object] | None:
    """Find method metadata by evaluator filename or stripped submission ID."""
    for key in method_metadata_key_candidates(filename, submission_id):
        if key in metadata_lookup:
            return metadata_lookup[key]
    return None


def normalize_method_columns(scores: pd.DataFrame) -> pd.DataFrame:
    """Ensure plotting metadata columns exist and have stable dtypes."""
    scores = scores.copy()
    if "prediction_label" not in scores.columns:
        scores["prediction_label"] = scores["filename"].map(strip_prediction_suffix)
    if "group" not in scores.columns:
        scores["group"] = scores["prediction_label"]
    if "label" not in scores.columns:
        scores["label"] = scores["prediction_label"]
    if "is_baseline" not in scores.columns:
        scores["is_baseline"] = False

    scores["prediction_label"] = scores["prediction_label"].fillna("").astype(str)
    scores["group"] = scores["group"].fillna(scores["prediction_label"]).astype(str)
    scores["label"] = scores["label"].fillna(scores["group"]).astype(str)
    scores["is_baseline"] = scores["is_baseline"].map(parse_bool)
    return scores


def create_method_names_tsv(
    metadata_file: str | Path,
    output_file: str | Path,
    filename_suffix: str = ".csv",
) -> pd.DataFrame:
    """Create a method-names TSV from CAFA score metadata.

    The output has columns expected by downstream plotting/evaluation helpers:
    ``filename``, ``group``, ``label``, and ``is_baseline``. Each submission ID
    is retained as a separate filename, while submissions from the same team
    share the same group/label.
    """
    metadata = pd.read_csv(metadata_file, dtype={"SubmissionId": str, "TeamName": str})
    required = {"SubmissionId", "TeamName"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_file} is missing columns: {sorted(missing)}")

    metadata = metadata.dropna(subset=["SubmissionId"]).copy()
    metadata["SubmissionId"] = metadata["SubmissionId"].astype(str).str.strip()
    metadata["TeamName"] = metadata["TeamName"].fillna("").astype(str).str.strip()
    metadata = metadata[(metadata["SubmissionId"] != "") & (metadata["TeamName"] != "")]
    metadata = metadata.drop_duplicates("SubmissionId", keep="last")

    method_names = pd.DataFrame(
        {
            "filename": metadata["SubmissionId"] + filename_suffix,
            "group": metadata["TeamName"],
            "label": metadata["TeamName"],
            "is_baseline": False,
        }
    )
    method_names = method_names.sort_values(["group", "filename"]).reset_index(drop=True)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    method_names.to_csv(output_file, sep="\t", index=False)
    return method_names


def apply_score_metadata(
    scores: pd.DataFrame,
    metadata_file: str | Path | None = None,
    baseline_metadata_file: str | Path | None = None,
) -> pd.DataFrame:
    """Replace filename labels with team/baseline metadata when available."""
    id_to_team = read_score_metadata(metadata_file) if metadata_file is not None else {}
    baseline_lookup = read_method_metadata(baseline_metadata_file) if baseline_metadata_file is not None else {}

    scores = normalize_method_columns(scores)
    rows = []
    removed_count = 0
    for row in scores.itertuples(index=False):
        row_dict = row._asdict()
        baseline_info = lookup_method_metadata(
            baseline_lookup,
            filename=row_dict["filename"],
            submission_id=row_dict.get("submission_id"),
        )
        if baseline_info and baseline_info.get("is_baseline", False):
            row_dict["group"] = str(baseline_info["group"])
            row_dict["label"] = str(baseline_info["label"])
            row_dict["prediction_label"] = str(baseline_info["label"])
            row_dict["is_baseline"] = True
            rows.append(row_dict)
            continue

        team_name = id_to_team.get(str(row_dict.get("submission_id", "")))
        if team_name:
            row_dict["group"] = str(team_name)
            row_dict["label"] = str(team_name)
            row_dict["prediction_label"] = str(team_name)
            row_dict["is_baseline"] = False
            rows.append(row_dict)
        elif metadata_file is None:
            row_dict["group"] = row_dict["prediction_label"]
            row_dict["label"] = row_dict["prediction_label"]
            row_dict["is_baseline"] = False
            rows.append(row_dict)
        else:
            removed_count += 1

    if removed_count:
        print(f"Remove {removed_count} rows not in final selection or baseline metadata")
    if not rows:
        return scores.iloc[0:0].copy()

    scores = pd.DataFrame(rows)
    scores = normalize_method_columns(scores)
    if metadata_file is not None:
        missing_teams = set(id_to_team.values()) - set(scores.loc[~scores["is_baseline"], "prediction_label"])
        if missing_teams:
            print(f"Warning: {len(missing_teams)} teams in metadata not found in scores: {sorted(missing_teams)}")
    return scores


def collect_evaluator_scores(data_dir: str | Path, subset_dirs: dict[str, str] | None = None) -> pd.DataFrame:
    """Collect weighted F-measure scores from NK/LK/PK evaluator folders."""
    data_dir = Path(data_dir)
    subset_dirs = subset_dirs or SUBSET_DIRS
    frames = []
    for subset, dirname in subset_dirs.items():
        result_file = data_dir / dirname / EVALUATION_FILE
        frames.append(read_evaluator_scores(result_file, subset=subset))
    return pd.concat(frames, ignore_index=True)


def collect_evaluator_curves(data_dir: str | Path, subset_dirs: dict[str, str] | None = None) -> pd.DataFrame:
    """Collect threshold-level evaluator results from available NK/LK/PK folders."""
    data_dir = Path(data_dir)
    subset_dirs = subset_dirs or SUBSET_DIRS
    frames = []
    for subset, dirname in subset_dirs.items():
        result_file = data_dir / dirname / EVALUATION_ALL_FILE
        if not result_file.exists():
            print(f"Skip missing PR-curve input: {result_file}")
            continue
        frames.append(read_evaluator_curves(result_file, subset=subset))
    if not frames:
        raise FileNotFoundError(f"No {EVALUATION_ALL_FILE} files found under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def write_score_table(scores: pd.DataFrame, output_file: str | Path) -> None:
    """Write the long-form score table used for plotting."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_file, sep="\t", index=False)


def read_score_table(score_file: str | Path) -> pd.DataFrame:
    """Read a cached long-form score table produced by ``write_score_table``."""
    score_file = Path(score_file)
    if not score_file.exists():
        raise FileNotFoundError(f"Cached score table not found: {score_file}")

    scores = pd.read_csv(
        score_file,
        sep="\t",
        dtype={
            "filename": str,
            "ns": str,
            "subset": str,
            "subset_label": str,
            "submission_id": str,
            "prediction_label": str,
            "group": str,
            "label": str,
        },
    )
    required = {"filename", "ns", PR_METRIC, "subset", "subset_label", "submission_id", "prediction_label"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{score_file} is missing cached score columns: {sorted(missing)}")

    scores[PR_METRIC] = pd.to_numeric(scores[PR_METRIC], errors="coerce").fillna(0.0)
    for optional_column in ("cov_w", "cov", "cov_max"):
        if optional_column in scores.columns:
            scores[optional_column] = pd.to_numeric(scores[optional_column], errors="coerce")
    return normalize_method_columns(scores)


def load_or_collect_scores(
    data_dir: str | Path,
    score_table: str | Path | None = None,
    metadata_file: str | Path | None = None,
    baseline_metadata_file: str | Path | None = None,
    force_reparse: bool = False,
) -> pd.DataFrame:
    """Reuse a cached post-metadata score table when available."""
    if score_table is not None:
        score_table = Path(score_table)
        if score_table.exists() and not force_reparse:
            print(f"Reuse cached score table: {score_table}")
            return read_score_table(score_table)

    scores = collect_evaluator_scores(data_dir)
    scores = apply_score_metadata(scores, metadata_file, baseline_metadata_file)
    if score_table is not None:
        write_score_table(scores, score_table)
    return scores


def read_curve_table(curve_file: str | Path) -> pd.DataFrame:
    """Read cached threshold-level evaluator rows used for PR curves."""
    curve_file = Path(curve_file)
    if not curve_file.exists():
        raise FileNotFoundError(f"Cached PR score table not found: {curve_file}")

    curves = pd.read_csv(
        curve_file,
        sep="\t",
        dtype={
            "filename": str,
            "ns": str,
            "subset": str,
            "subset_label": str,
            "submission_id": str,
            "prediction_label": str,
            "group": str,
            "label": str,
        },
    )
    required = {"filename", "ns", "tau", "subset", "subset_label", "submission_id", "prediction_label", PR_METRIC, *PR_CURVE_COLUMNS}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"{curve_file} is missing cached PR score columns: {sorted(missing)}")

    numeric_columns = ["tau", PR_METRIC, *PR_CURVE_COLUMNS]
    if "cov" in curves.columns:
        numeric_columns.append("cov")
    if "cov_w" in curves.columns:
        numeric_columns.append("cov_w")
    for column in numeric_columns:
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    return normalize_method_columns(curves)


def load_or_collect_curves(
    data_dir: str | Path,
    curve_table: str | Path | None = None,
    metadata_file: str | Path | None = None,
    baseline_metadata_file: str | Path | None = None,
    force_reparse: bool = False,
) -> pd.DataFrame:
    """Reuse cached post-metadata PR curve scores when available."""
    if curve_table is not None:
        curve_table = Path(curve_table)
        if curve_table.exists() and not force_reparse:
            print(f"Reuse cached PR score table: {curve_table}")
            return read_curve_table(curve_table)

    curves = collect_evaluator_curves(data_dir)
    curves = apply_score_metadata(curves, metadata_file, baseline_metadata_file)
    if curve_table is not None:
        write_score_table(curves, curve_table)
    return curves


def plot_cafa6_results(
    scores: pd.DataFrame,
    output_file: str | Path,
    figsize: tuple[float, float] | None = None,
    top_n: int | None = 10,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """Plot a 3x3 bar grid: rows are NK/LK/PK and columns are GO aspects.
    Limited to the top ``top_n`` predictions per panel, sorted by f_micro
    and then team name. Rankings are computed independently in each subplot.
    """
    scores = normalize_method_columns(scores)
    if figsize is None:
        n_predictions = max(scores["filename"].nunique(), 1)
        if top_n is not None and top_n > 0:
            n_predictions = min(n_predictions, top_n)
        figsize = (16.0, max(13.0, 0.45 * (n_predictions + 4) * 3))

    subsets = list(SUBSET_DIRS.keys())
    aspects = list(ASPECTS.keys())
    plot_panels = _select_plot_panels(scores, subsets=subsets, aspects=aspects, top_n=top_n)
    baseline_panels = _select_baseline_panels(scores, subsets=subsets, aspects=aspects)
    color_lookup = _make_display_color_lookup(plot_panels)
    baseline_style_lookup = _make_baseline_style_lookup(scores)
    x_max = max(1.0, scores[PR_METRIC].max() * 1.25)

    fig, axes = plt.subplots(len(subsets), len(aspects), figsize=figsize, sharey=False, squeeze=False)
    for row_index, subset in enumerate(subsets):
        for col_index, aspect in enumerate(aspects):
            ax = axes[row_index][col_index]
            panel = plot_panels[(subset, aspect)]
            baseline_panel = baseline_panels[(subset, aspect)]
            team_label_position = "above" if subset == "PK" else "inside"
            _plot_result_panel(
                ax,
                panel,
                color_lookup,
                x_max=x_max,
                team_label_position=team_label_position,
                baseline_panel=baseline_panel,
                baseline_style_lookup=baseline_style_lookup,
            )

            ax.set_title(f"{ASPECTS[aspect]}: {SUBSET_LABELS[subset]}", fontsize=12)
            if col_index == 0:
                ax.set_ylabel("Team")
            else:
                ax.set_ylabel("")
            if row_index == len(subsets) - 1:
                ax.set_xlabel(r"$F_{\mathrm{max}}$")
            else:
                ax.set_xlabel("")

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    return fig, axes.tolist()


def plot_precision_recall_curves(
    curves: pd.DataFrame,
    output_dir: str | Path,
    top_n: int | None = 10,
    display_n_other: int | None = 50,
    metric: str = PR_METRIC,
    curve_columns: tuple[str, str] = PR_CURVE_COLUMNS,
    coverage_threshold: float = 0.3,
    cumulate: bool = True,
    extreme_points: bool = False,
    figsize: tuple[float, float] = (8.5, 8.0),
) -> list[Path]:
    """Plot precision-recall curves from ``evaluation_all.tsv`` rows.

    The best submission per unique team is selected independently for each
    subset/aspect panel. The top ``top_n`` teams are colored and listed in the
    legend; the remaining displayed teams, up to ``display_n_other`` total
    methods per panel, are plotted in gray without legend entries.
    """
    x_col, y_col = curve_columns
    curve_df, best_df = _prepare_precision_recall_data(
        curves,
        metric=metric,
        x_col=x_col,
        y_col=y_col,
        coverage_threshold=coverage_threshold,
        cumulate=cumulate,
        extreme_points=extreme_points,
    )
    if best_df.empty:
        raise ValueError("No precision-recall curves passed the coverage threshold")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_color_lookup = _make_pr_panel_color_lookup(best_df, top_n=top_n, metric=metric)
    baseline_style_lookup = _make_baseline_style_lookup(best_df)
    output_files = []
    for subset in SUBSET_DIRS:
        for aspect in ASPECTS:
            panel_best = best_df[(best_df["subset"] == subset) & (best_df["ns"] == aspect)].copy()
            if panel_best.empty:
                continue

            # panel_best = panel_best.sort_values([metric, "max_cov", "group"], ascending=[False, False, True])
            panel_best = panel_best.sort_values([metric, "group"], ascending=[False, True])
            baseline_best = panel_best[panel_best["is_baseline"]].copy()
            method_best = panel_best[~panel_best["is_baseline"]].copy()
            if display_n_other is not None and display_n_other > 0:
                method_best = method_best.head(display_n_other)
            method_best = method_best.assign(rank=np.arange(1, len(method_best) + 1))
            baseline_best = baseline_best.assign(rank=np.arange(len(method_best) + 1, len(method_best) + len(baseline_best) + 1))
            panel_best = pd.concat([method_best, baseline_best], ignore_index=True)

            if top_n is None or top_n <= 0:
                top_groups = set(method_best["group"])
            else:
                top_groups = set(method_best.head(top_n)["group"])

            panel_curves = curve_df.merge(
                panel_best[["subset", "ns", "group", "rank"]],
                on=["subset", "ns", "group"],
                how="inner",
            )
            if subset == "PK":
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=(figsize[0]*1.3, figsize[1]))
            _plot_precision_recall_panel(
                ax,
                panel_curves=panel_curves,
                panel_best=panel_best,
                top_groups=top_groups,
                top_color_lookup=panel_color_lookup.get((subset, aspect), {}),
                baseline_style_lookup=baseline_style_lookup,
                metric=metric,
                x_col=x_col,
                y_col=y_col,
                subset=subset,
                aspect=aspect,
            )
            output_file = output_dir / f"precision_recall_{metric}_{subset}_{aspect}.png"
            fig.tight_layout()
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            output_files.append(output_file)

    return output_files


def _prepare_precision_recall_data(
    curves: pd.DataFrame,
    metric: str,
    x_col: str,
    y_col: str,
    coverage_threshold: float,
    cumulate: bool,
    extreme_points: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subset", "ns", "filename", "prediction_label", "tau", metric, x_col, y_col}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Precision-recall input is missing columns: {sorted(missing)}")

    coverage_col = "cov" if "cov" in curves.columns else "cov_w" if "cov_w" in curves.columns else None
    if coverage_col is None:
        # raise ValueError("Precision-recall input needs either a cov or cov_w column")
        pass  # Allow missing coverage column, but coverage_threshold will be ignored

    work = curves.copy()
    work["group"] = work.get("group", work["prediction_label"]).fillna("").astype(str)
    work["label"] = work.get("label", work["prediction_label"]).fillna(work["group"]).astype(str)
    work = work[(work["group"] != "") & work[metric].notna() & work[x_col].notna() & work[y_col].notna()]

    numeric_columns = ["tau", metric, x_col, y_col, coverage_col] if coverage_col is not None else ["tau", metric, x_col, y_col]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=numeric_columns)

    attempt_best = (
        work.groupby(["subset", "ns", "group", "filename"], as_index=False)
        # .agg(best_metric=(metric, "max"), max_cov=(coverage_col, "max"), label=("label", "first"))
        .agg(best_metric=(metric, "max"), label=("label", "first"))
        # .sort_values(["subset", "ns", "group", "best_metric", "max_cov"], ascending=[True, True, True, False, False])
        .sort_values(["subset", "ns", "group", "best_metric"], ascending=[True, True, True, False])
    )
    chosen_attempts = attempt_best.drop_duplicates(["subset", "ns", "group"], keep="first")
    chosen = work.merge(
        chosen_attempts[["subset", "ns", "group", "filename"]],
        on=["subset", "ns", "group", "filename"],
        how="inner",
    )

    # max_cov = chosen.groupby(["subset", "ns", "group"], as_index=False)[coverage_col].max()
    # max_cov = max_cov.rename(columns={coverage_col: "max_cov"})
    best_rows = (
        # chosen.sort_values(["subset", "ns", "group", metric, coverage_col, "tau"], ascending=[True, True, True, False, False, True])
        chosen.sort_values(["subset", "ns", "group", metric, "tau"], ascending=[True, True, True, False, True])
        .drop_duplicates(["subset", "ns", "group"], keep="first")
        # .merge(max_cov, on=["subset", "ns", "group"], how="left")
    )
    # best_rows = best_rows[best_rows["max_cov"] >= coverage_threshold].copy()
    if best_rows.empty:
        return chosen.iloc[0:0].copy(), best_rows

    keep_keys = best_rows[["subset", "ns", "group"]].drop_duplicates()
    chosen = chosen.merge(keep_keys, on=["subset", "ns", "group"], how="inner")
    chosen = chosen.sort_values(["subset", "ns", "group", "tau"]).copy()
    if cumulate and metric.startswith("f"):
        chosen[y_col] = chosen.groupby(["subset", "ns", "group"])[y_col].cummax()

    if extreme_points:
        chosen = _add_precision_recall_endpoints(chosen, x_col=x_col, y_col=y_col)
    return chosen, best_rows


def _add_precision_recall_endpoints(curves: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    frames = []
    for _, group in curves.groupby(["subset", "ns", "group"], sort=False):
        group = group.sort_values("tau").copy()
        start = group.head(1).copy()
        start.loc[:, "tau"] = 0.0
        start.loc[:, x_col] = 1.0
        start.loc[:, y_col] = 0.0

        end = group.tail(1).copy()
        end.loc[:, "tau"] = 1.1
        end.loc[:, x_col] = 0.0
        end.loc[:, y_col] = 1.0
        frames.append(pd.concat([start, group, end], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def _make_pr_panel_color_lookup(
    best_df: pd.DataFrame,
    top_n: int | None,
    metric: str,
) -> dict[tuple[str, str], dict[str, tuple[float, float, float, float]]]:
    top_groups_by_panel = {}
    best_df = normalize_method_columns(best_df)
    for panel_key, panel in best_df[~best_df["is_baseline"]].groupby(["subset", "ns"], sort=False):
        # panel = panel.sort_values([metric, "max_cov", "group"], ascending=[False, False, True])
        panel = panel.sort_values([metric, "group"], ascending=[False, True])        
        if top_n is not None and top_n > 0:
            panel = panel.head(top_n)
        top_groups_by_panel[panel_key] = panel["group"].dropna().astype(str).tolist()

    group_counts = {}
    for groups in top_groups_by_panel.values():
        for group in groups:
            group_counts[group] = group_counts.get(group, 0) + 1

    repeated_groups = sorted(
        (group for group, count in group_counts.items() if count > 1),
        key=lambda group: (-group_counts[group], group),
    )
    max_panel_size = max((len(groups) for groups in top_groups_by_panel.values()), default=0)
    base_colors = _make_team_color_variants(max(max_panel_size, len(repeated_groups), 1))
    repeated_color_lookup = {
        group: base_colors[index]
        for index, group in enumerate(repeated_groups)
    }

    panel_color_lookup = {}
    for panel_key, groups in top_groups_by_panel.items():
        occupied_colors = {
            repeated_color_lookup[group]
            for group in groups
            if group in repeated_color_lookup
        }
        available_colors = [color for color in base_colors if color not in occupied_colors]
        color_index = 0
        color_lookup = {}
        for group in groups:
            if group in repeated_color_lookup:
                color_lookup[group] = repeated_color_lookup[group]
            else:
                color_lookup[group] = available_colors[color_index]
                color_index += 1
        panel_color_lookup[panel_key] = color_lookup
    return panel_color_lookup


def _plot_precision_recall_panel(
    ax: plt.Axes,
    panel_curves: pd.DataFrame,
    panel_best: pd.DataFrame,
    top_groups: set[str],
    top_color_lookup: dict[str, tuple[float, float, float, float]],
    baseline_style_lookup: dict[str, dict[str, object]],
    metric: str,
    x_col: str,
    y_col: str,
    subset: str,
    aspect: str,
) -> None:
    x = np.arange(0.01, 1.0, 0.01)
    X, Y = np.meshgrid(x, x)
    Z = np.divide(2 * X * Y, X + Y, out=np.zeros_like(X), where=(X + Y) != 0)
    contours = ax.contour(X, Y, Z, np.arange(0.1, 1.0, 0.1), colors="0.75", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=9)

    gray = (0.68, 0.68, 0.68, 0.35)
    best_lookup = panel_best.set_index("group")
    legend_order = {}
    for group, data in panel_curves.sort_values("rank", ascending=False).groupby("group", sort=False):
        best = best_lookup.loc[group]
        is_baseline = bool(best.get("is_baseline", False))
        is_top = group in top_groups
        baseline_style = baseline_style_lookup.get(group, {})
        color = (baseline_style.get("color") or gray) if is_baseline else top_color_lookup.get(group, gray) if is_top else gray
        marker = baseline_style.get("marker", "s") if is_baseline else "o"
        linestyle = "--" if is_baseline else "-"
        linewidth = 2.2 if is_top or is_baseline else 0.8
        alpha = 1.0 if is_top or is_baseline else 0.45
        label = None
        if is_top or is_baseline:
            metric_label = r"$F_{\mathrm{max}}$"
            display_label = baseline_style.get("label", best["label"]) if is_baseline else best["label"]
            label = f"{display_label} ({metric_label}={best[metric]:.3f})"
            legend_order[label] = int(best["rank"])

        data = data.sort_values("tau")
        markevery = None
        if is_baseline:
            best_tau = float(best["tau"])
            best_index = int(np.argmin(np.abs(data["tau"].to_numpy(dtype=float) - best_tau)))
            markevery = [best_index]
        ax.plot(
            data[x_col],
            data[y_col],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            marker=marker if is_baseline else None,
            markersize=7 if is_baseline else 0,
            markerfacecolor=color if is_baseline else None,
            markeredgecolor=color if is_baseline else None,
            markevery=markevery,
            label=label,
        )

        if group in best_lookup.index:
            if is_baseline:
                ax.plot(
                    best[x_col],
                    best[y_col],
                    color=color,
                    marker="o",
                    markersize=11,
                    markerfacecolor="none",
                    markeredgewidth=1.7,
                    linestyle="None",
                    alpha=alpha,
                    zorder=20,
                )
                ax.plot(
                    best[x_col],
                    best[y_col],
                    color=color,
                    marker=marker,
                    markersize=7,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="None",
                    alpha=alpha,
                    zorder=21,
                )
            else:
                marker_size = 6 if is_top else 3
                ax.plot(best[x_col], best[y_col], color=color, marker="o", markersize=marker_size, alpha=alpha)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(AXIS_TITLE_DICT.get(x_col, x_col))
    ax.set_ylabel(AXIS_TITLE_DICT.get(y_col, y_col))
    ax.set_title(f"{ONTOLOGY_SHORT_LABELS.get(aspect, aspect)}: {SUBSET_LABELS.get(subset, subset)}")
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_entries = sorted(
            zip(handles, labels),
            key=lambda item: legend_order.get(item[1], len(legend_order) + 1),
        )
        handles, labels = zip(*legend_entries)
        if subset == "PK":
            legend = ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=True)
        else:
            legend = ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                fontsize=8,
                frameon=True,
            )
        for line in legend.get_lines():
            line.set_linewidth(4)
        for text_artist in legend.get_texts():
            apply_cjk_font_to_text(text_artist, text_artist.get_text(), fontsize=8)


def _select_plot_panels(
    scores: pd.DataFrame,
    subsets: list[str],
    aspects: list[str],
    top_n: int | None,
) -> dict[tuple[str, str], pd.DataFrame]:
    panels = {}
    for subset in subsets:
        for aspect in aspects:
            panel = scores[
                (scores["subset"] == subset)
                & (scores["ns"] == aspect)
                & (~scores["is_baseline"])
            ].copy()
            panel = panel.sort_values([PR_METRIC, "group"], ascending=[False, True]).reset_index(drop=True)
            panel = panel.drop_duplicates("group", keep="first")
            if top_n is not None and top_n > 0:
                panel = panel.head(top_n)
            panels[(subset, aspect)] = panel
    return panels


def _make_display_color_lookup(plot_panels: dict[tuple[str, str], pd.DataFrame]) -> dict[str, tuple[float, float, float, float]]:
    displayed_teams = sorted(
        {
            team
            for panel in plot_panels.values()
            for team in panel["prediction_label"].dropna().astype(str)
        }
    )
    if not displayed_teams:
        return {}

    colors = _make_team_color_variants(len(displayed_teams))
    return {
        team: colors[index]
        for index, team in enumerate(displayed_teams)
    }


def _select_baseline_panels(
    scores: pd.DataFrame,
    subsets: list[str],
    aspects: list[str],
    metric: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    panels = {}
    scores = normalize_method_columns(scores)
    baselines = scores[scores["is_baseline"]].copy()
    for subset in subsets:
        for aspect in aspects:
            panel = baselines[(baselines["subset"] == subset) & (baselines["ns"] == aspect)].copy()
            panel = panel.sort_values([PR_METRIC, "group"], ascending=[False, True])
            panels[(subset, aspect)] = panel.drop_duplicates("group", keep="first")
    return panels


def _make_baseline_style_lookup(scores: pd.DataFrame) -> dict[str, dict[str, object]]:
    scores = normalize_method_columns(scores)
    baselines = scores[scores["is_baseline"]].copy()
    if baselines.empty:
        return {}

    baseline_info = (
        baselines.sort_values(["group", "label"])
        .drop_duplicates("group", keep="first")[["group", "label"]]
        .sort_values("label")
    )
    return {
        row.group: {
            **_baseline_style_for_name(row.group, row.label),
        }
        for row in baseline_info.itertuples(index=False)
    }


def _make_baseline_legend_handles(
    baseline_style_lookup: dict[str, dict[str, object]],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle="--",
            linewidth=2.0,
            label=style["label"],
        )
        for _, style in sorted(baseline_style_lookup.items(), key=lambda item: str(item[1]["label"]))
    ]


def _baseline_style_for_name(group: str, label: str) -> dict[str, object]:
    text = f"{group} {label}".lower().replace("_", " ").replace("-", " ")
    for baseline_type, style in BASELINE_STYLE_BY_TYPE.items():
        patterns = [pattern.replace("_", " ").replace("-", " ") for pattern in style["patterns"]]
        if any(pattern in text for pattern in patterns):
            return {
                "baseline_type": baseline_type,
                "label": style["display_label"],
                "color": mcolors.to_rgba(style["color"]),
                "marker": style["marker"],
            }
    return {
        "baseline_type": str(group),
        "label": str(label),
        "color": mcolors.to_rgba("0.25"),
        "marker": "s",
    }


def _make_team_color_variants(n_colors: int) -> list[tuple[float, float, float, float]]:
    reserved_colors = {
        mcolors.to_hex(style["color"]).lower()
        for style in BASELINE_STYLE_BY_TYPE.values()
    }
    request_size = max(n_colors + len(reserved_colors) + 8, 16)
    colors = []
    while len(colors) < n_colors:
        colors = [
            color
            for color in _make_dark2_variants(request_size)
            if mcolors.to_hex(color).lower() not in reserved_colors
        ]
        request_size *= 2
    return colors[:n_colors]


def _make_dark2_variants(n_colors: int) -> list[tuple[float, float, float, float]]:
    """Return colors based on tab20, adding value variants when needed."""
    base_colors = [mcolors.to_rgb(color) for color in plt.get_cmap("tab20").colors]
    if n_colors <= len(base_colors):
        return [(*base_colors[index], 1.0) for index in range(n_colors)]

    value_factors = (1.0, 0.82, 1.16, 0.66, 0.94, 0.74)
    colors = []
    for index in range(n_colors):
        base = base_colors[index % len(base_colors)]
        round_index = index // len(base_colors)
        h, s, v = colorsys.rgb_to_hsv(*base)
        v = min(max(v * value_factors[round_index % len(value_factors)], 0.0), 1.0)
        s = min(max(s * (1.0 - 0.08 * (round_index % 3)), 0.0), 1.0)
        colors.append((*colorsys.hsv_to_rgb(h, s, v), 1.0))
    return colors


def _coverage_value(row) -> float | None:
    for column in ("cov_w", "cov", "cov_max"):
        if hasattr(row, column):
            value = getattr(row, column)
            if pd.notna(value):
                return float(value)
    return None


def _plot_result_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    color_lookup: dict[str, tuple[float, float, float, float]],
    x_max: float,
    team_label_position: str = "inside",
    baseline_panel: pd.DataFrame | None = None,
    baseline_style_lookup: dict[str, dict[str, object]] | None = None,
) -> None:
    baseline_panel = baseline_panel if baseline_panel is not None else pd.DataFrame()
    baseline_style_lookup = baseline_style_lookup or {}
    if panel.empty and baseline_panel.empty:
        ax.text(0.5, 0.5, "No results", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, x_max)
        ax.set_yticks([])
        return

    baseline_panel = baseline_panel.sort_values([PR_METRIC, "group"], ascending=[False, True]).copy()
    best_baseline = baseline_panel.head(1)
    remaining_baselines = baseline_panel.iloc[1:].copy()

    bar_rows = []
    for row in panel.itertuples(index=False):
        bar_rows.append(
            {
                "label": row.prediction_label,
                "group": row.group,
                "score": float(row.f_micro),
                "coverage": _coverage_value(row),
                "color": color_lookup[row.prediction_label],
                "is_baseline": False,
            }
        )
    for row in best_baseline.itertuples(index=False):
        style = baseline_style_lookup.get(row.group, {})
        bar_rows.append(
            {
                "label": style.get("label", row.label),
                "group": row.group,
                "score": float(row.f_micro),
                "coverage": _coverage_value(row),
                "color": style.get("color", "0.35"),
                "is_baseline": True,
            }
        )

    y_positions = np.arange(len(bar_rows))
    bars = ax.barh(
        y_positions,
        [row["score"] for row in bar_rows],
        color=[row["color"] for row in bar_rows],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_yticks([])
    # ax.set_yticklabels(["Team"] * len(bar_rows), fontsize=10)

    annotation_x_max = max([row["score"] for row in bar_rows] + [0.0])
    baseline_line_max = remaining_baselines[PR_METRIC].max() if not remaining_baselines.empty else 0.0
    x_limit = max(x_max, annotation_x_max * 1.35, float(baseline_line_max) * 1.15 if pd.notna(baseline_line_max) else 0.0)
    ax.set_xlim(0, x_limit)
    bottom_buffer = max(1.2, 0.35 * len(remaining_baselines))
    ax.set_ylim(len(bar_rows) + bottom_buffer, -1.3)
    ax.grid(axis="x", color="0.88", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    label_y = len(bar_rows) + 0.35
    for row in remaining_baselines.itertuples(index=False):
        style = baseline_style_lookup.get(row.group)
        if not style:
            continue
        ax.axvline(
            float(row.f_micro),
            color=style["color"],
            linestyle=":",
            linewidth=3.0,
            zorder=1,
        )
        ax.text(
            float(row.f_micro),
            label_y,
            style["label"],
            ha="right",
            va="top",
            rotation=30,
            fontsize=10,
            color=style["color"],
            clip_on=False,
            **cjk_font_kwargs(style["label"]),
        )

    for bar, row in zip(bars, bar_rows):
        width = float(row["score"])
        y = bar.get_y() + bar.get_height() / 2
        coverage = row["coverage"]
        if coverage is not None:
            ax.text(
                0.01,
                y,
                f"C={coverage:.2f}",
                ha="left",
                va="center",
                fontsize=10,
                color="white" if width > x_limit * 0.18 else "black",
                clip_on=True,
            )
        ax.text(
            width + x_limit * 0.03,
            y,
            f"{row['label']}\n{width:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            clip_on=False,
            **cjk_font_kwargs(row["label"]),
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CAFA6 f_micro evaluator results by subset and GO aspect.")
    parser.add_argument(
        "--data-dir",
        "--DATA_DIR",
        dest="data_dir",
        required=True,
        help="Directory containing test_cafaeval_NK, test_cafaeval_LK, and test_cafaeval_PK folders.",
    )
    parser.add_argument("--output-plot", required=True, help="Output plot path, e.g. cafa6_f_micro_w_grid.png.")
    parser.add_argument(
        "--output-pr-dir",
        default=None,
        help="Optional output directory for precision-recall curves from evaluation_all.tsv files.",
    )
    parser.add_argument(
        "--output-pr-table",
        default=None,
        help="Optional long-form TSV of parsed PR curve scores. If it already exists, it is reused as a cache.",
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Optional long-form TSV of plotted scores. If it already exists, it is reused as a cache.",
    )
    parser.add_argument(
        "--force-reparse-scores",
        action="store_true",
        help="Ignore existing score-table caches and rebuild scores from evaluator outputs.",
    )
    parser.add_argument(
        "--score-metadata",
        default=None,
        help="Optional CAFA score metadata CSV with SubmissionId and TeamName columns.",
    )
    parser.add_argument(
        "--baseline-metadata",
        default=None,
        help="Optional method_names-style TSV for baseline methods to retain and style separately.",
    )
    parser.add_argument(
        "--method-names-output",
        default=None,
        help="Optional path to write method_names.tsv from --score-metadata.",
    )
    parser.add_argument(
        "--method-name-filename-suffix",
        default=".csv",
        help="Suffix appended to SubmissionId in method_names.tsv filename column.",
    )
    parser.add_argument("--width", type=float, default=None, help="Optional figure width in inches.")
    parser.add_argument("--height", type=float, default=None, help="Optional figure height in inches.")
    parser.add_argument(
        "--top-n-teams",
        type=int,
        default=10,
        help="Top N teams to display independently in each subplot. Use 0 to plot all teams.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.method_names_output:
        if not args.score_metadata:
            raise ValueError("--method-names-output requires --score-metadata")
        create_method_names_tsv(
            metadata_file=args.score_metadata,
            output_file=args.method_names_output,
            filename_suffix=args.method_name_filename_suffix,
        )

    scores = load_or_collect_scores(
        data_dir=args.data_dir,
        score_table=args.output_table,
        metadata_file=args.score_metadata,
        baseline_metadata_file=args.baseline_metadata,
        force_reparse=args.force_reparse_scores,
    )

    figsize = (args.width, args.height) if args.width and args.height else None
    top_n = None if args.top_n_teams == 0 else args.top_n_teams
    plot_cafa6_results(scores, output_file=args.output_plot, figsize=figsize, top_n=top_n)

    if args.output_pr_dir:
        curves = load_or_collect_curves(
            data_dir=args.data_dir,
            curve_table=args.output_pr_table,
            metadata_file=args.score_metadata,
            baseline_metadata_file=args.baseline_metadata,
            force_reparse=args.force_reparse_scores,
        )
        plot_precision_recall_curves(curves, output_dir=args.output_pr_dir, top_n=top_n)
    # python3 '/work/idoerg/ahphan/democafa_package/democafa/groundtruth/plot_cafa6_results.py' --data-dir data/cafa6/scratch --output-plot data/cafa6/for_pub/cafa6_f_micro_w_baseline.png --output-pr-dir data/cafa6/for_pub/pr_curves --score-metadata data/cafa6/raw/cafa-6-protein-function-prediction-all-scores-2026-06-01T18:12:15.csv --method-names-output data/cafa6/for_pub/method_names.tsv --top-n-teams 10 --baseline-metadata '/work/idoerg/ahphan/democafa_package/data/cafa6/for_pub/baseline_names.tsv' --output-pr-table '/work/idoerg/ahphan/democafa_package/data/cafa6/for_pub/cafa6_pr_curve_scores.tsv' --output-table '/work/idoerg/ahphan/democafa_package/data/cafa6/for_pub/cafa6_f_micro_w_scores.tsv'

if __name__ == "__main__":
    main()
