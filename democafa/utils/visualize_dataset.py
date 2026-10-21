#!/usr/bin/env python3
"""Visualization helpers for CAFA-style ground-truth datasets."""

import argparse
import gzip
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio import SeqIO


GROUND_TRUTH_COLUMNS = ("EntryID", "term", "aspect")
DEFAULT_KNOWLEDGE_LABELS = ("No knowledge", "Limited knowledge", "Partial knowledge")
DEFAULT_BROAD_TAXON_GROUPS: tuple[dict[str, Any], ...] = (
    {"group": "Animals", "taxon_ids": {"33208"}, "names": {"Metazoa", "Animalia"}},
    {"group": "Plants", "taxon_ids": {"33090"}, "names": {"Viridiplantae"}},
    {"group": "Fungi", "taxon_ids": {"4751"}, "names": {"Fungi"}},
    {"group": "Bacteria", "taxon_ids": {"2"}, "names": {"Bacteria"}},
    {"group": "Archaea", "taxon_ids": {"2157"}, "names": {"Archaea"}},
    {"group": "Viruses", "taxon_ids": {"10239"}, "names": {"Viruses", "Virus"}},
    {"group": "Other eukaryotes", "taxon_ids": {"2759"}, "names": {"Eukaryota"}},
)
DEFAULT_TESTSET_BROAD_GROUPS = ("Animals", "Plants", "Fungi", "Bacteria", "Archaea", "Viruses")


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def extract_uniprot_entry_id(record_id: str) -> str:
    """Return a UniProt accession from common FASTA IDs."""
    fields = record_id.split("|")
    if len(fields) >= 2 and fields[0] in {"sp", "tr"}:
        return fields[1]
    return record_id.split()[0]


def extract_taxon_id_from_description(description: str) -> str | None:
    """Extract a taxonomy ID from UniProt-like FASTA header text."""
    patterns = (
        r"^\S+\s+(\d+)(?:\s|$)",
        r"\bOX=(\d+)\b",
        r"\bTax(?:on)?ID=(\d+)\b",
        r"\btax(?:on|id)[:=](\d+)\b",
        r"\btax(?:on|id)\s+(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def read_fasta_taxa(fasta_path: str | Path) -> pd.DataFrame:
    """Read ``EntryID`` to taxon-ID mapping from a FASTA file.
    
    UniProt headers usually store taxonomy as ``OX=<taxon_id>``. This parser
    also accepts several simple ``taxid``/``taxon`` variants for locally
    rewritten FASTA headers.
    """
    rows: list[dict[str, str | None]] = []
    with _open_text(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            rows.append(
                {
                    "EntryID": extract_uniprot_entry_id(record.id),
                    "taxon_id": extract_taxon_id_from_description(record.description),
                }
            )
    
    taxa = pd.DataFrame(rows, columns=["EntryID", "taxon_id"])
    if taxa.empty:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    if taxa["EntryID"].duplicated().any():
        taxa = taxa.drop_duplicates("EntryID", keep="first")
    return taxa


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read a CAFA ground-truth TSV with ``EntryID``, ``term``, and ``aspect``."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = set(GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df.loc[:, GROUND_TRUTH_COLUMNS].dropna(subset=["EntryID"]).drop_duplicates()


def read_ground_truth_sets(
    no_knowledge_path: str | Path,
    limited_knowledge_path: str | Path,
    partial_knowledge_path: str | Path,
    labels: Sequence[str] = DEFAULT_KNOWLEDGE_LABELS,
) -> pd.DataFrame:
    """Read and combine NK/LK/PK ground-truth files."""
    paths = (no_knowledge_path, limited_knowledge_path, partial_knowledge_path)
    if len(paths) != len(labels):
        raise ValueError(f"Expected {len(paths)} labels, got {len(labels)}")
    
    frames = []
    for path, label in zip(paths, labels):
        frame = read_ground_truth(path)
        frame["knowledge_gain"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_uniprot_lineage(lineage_path: str | Path) -> pd.DataFrame:
    """Load a UniProt taxonomy/lineage TSV and normalize likely column names.

    Expected input is a UniProt TSV containing a taxon ID column plus optional
    lineage names and/or lineage IDs. Common columns include ``Organism (ID)``,
    ``Organism ID``, ``Taxonomic lineage``, and ``Taxonomic lineage (Ids)``.
    """
    df = pd.read_csv(lineage_path, sep="\t", dtype=str)
    normalized = {_normalize_column_name(column): column for column in df.columns}
    
    taxon_col = _first_existing(
        normalized,
        (
            "taxon_id",
            "organism_id",
            "organism_ids",
            "organism_id_taxonomy",
            "taxonomy_id",
            "taxonomy_ids",
            "taxid",
            "tax_id",
            "taxon",
        ),
    )
    lineage_col = _first_existing(
        normalized,
        (
            "taxonomic_lineage",
            "lineage",
            "taxonomy_lineage",
            "taxonomic_lineage_names",
        ),
        required=False,
    )
    lineage_ids_col = _first_existing(
        normalized,
        (
            "taxonomic_lineage_ids",
            "lineage_ids",
            "taxonomy_lineage_ids",
            "taxonomic_lineage_id",
            "taxonomic_lineage_id_s",
        ),
        required=False,
    )
    name_col = _first_existing(
        normalized,
        ("organism", "organism_name", "scientific_name", "taxon_name", "name"),
        required=False,
    )
    
    out = pd.DataFrame({"taxon_id": df[taxon_col].astype(str).str.extract(r"(\d+)", expand=False)})
    out["lineage"] = df[lineage_col].fillna("") if lineage_col else ""
    out["lineage_ids"] = df[lineage_ids_col].fillna("") if lineage_ids_col else ""
    out["taxon_name"] = df[name_col].fillna("") if name_col else ""
    out["species_taxon_id"] = out.apply(
        lambda row: extract_species_taxon_id_from_lineage_ids(row["lineage_ids"], fallback_taxon_id=row["taxon_id"]),
        axis=1,
    )
    out["species_taxon_name"] = out["taxon_name"].map(clean_species_taxon_name)
    out = out.dropna(subset=["taxon_id"]).drop_duplicates("taxon_id", keep="first")
    return out


def extract_species_taxon_id_from_lineage_ids(lineage_ids: str, fallback_taxon_id: str | None = None) -> str | None:
    """Return the species-level taxon ID from UniProt lineage IDs when present.

    UniProt strain records often have an organism taxon ID below species level,
    while the lineage contains the species ancestor, for example
    ``287 (species), 208693 (strain)``. In those cases the species ID should be
    used for distribution-level species counts.
    """
    if pd.isna(lineage_ids):
        return fallback_taxon_id
    
    species_matches = re.findall(r"(\d+)\s*\(\s*species\s*\)", str(lineage_ids), flags=re.IGNORECASE)
    if species_matches:
        return species_matches[-1]
    return fallback_taxon_id


def clean_species_taxon_name(taxon_name: str) -> str:
    """Trim common strain/isolate suffixes from organism names for plot labels."""
    if pd.isna(taxon_name):
        return ""
    
    name = str(taxon_name).strip()
    name = re.split(r"\s*\((?:strain|substrain|isolate|serotype)\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    name = re.sub(r"\s+(?:strain|str\.|substrain|isolate|serotype)\s+.+$", "", name, flags=re.IGNORECASE)
    return name.strip()


def _normalize_column_name(column: str) -> str:
    normalized = column.strip().lower()
    normalized = re.sub(r"[\[\]()/.-]+", " ", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized.strip("_")


def _first_existing(
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    if required:
        raise ValueError(
            "Could not find required column. Tried: "
            + ", ".join(candidates)
            + f". Available columns: {sorted(normalized_columns.values())}"
        )
    return None


def _first_non_empty(values: Sequence[str]) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _first_non_empty_or(values: Sequence[str], fallback: str) -> str:
    value = _first_non_empty(values)
    return value if value else fallback


def assign_broad_taxon_group(
    taxon_id: str | None,
    lineage: str = "",
    lineage_ids: str = "",
    taxon_name: str = "",
    broad_groups: Sequence[Mapping[str, Any]] = DEFAULT_BROAD_TAXON_GROUPS,
) -> str:
    """Assign a taxon to a broad lineage group."""
    if pd.isna(taxon_id):
        return "Missing taxon"
    
    taxon_id = str(taxon_id)
    lineage_id_set = set(re.findall(r"\d+", f"{lineage_ids};{taxon_id}"))
    lineage_names = {name.strip().lower() for name in re.split(r"[;,]", f"{lineage};{taxon_name}") if name.strip()}
    
    for group in broad_groups:
        group_ids = {str(value) for value in group.get("taxon_ids", set())}
        group_names = {str(value).lower() for value in group.get("names", set())}
        if lineage_id_set & group_ids:
            return str(group["group"])
        if lineage_names & group_names:
            return str(group["group"])
    return "Other/unknown"


def attach_taxa_to_ground_truth(
    ground_truth: pd.DataFrame,
    fasta_taxa: pd.DataFrame,
    lineage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach taxon IDs and optional lineage groups to ground-truth rows."""
    merged = ground_truth.merge(fasta_taxa, on="EntryID", how="left", validate="many_to_one")
    if lineage is not None:
        merged = merged.merge(lineage, on="taxon_id", how="left", validate="many_to_one")
        merged["species_taxon_id"] = merged["species_taxon_id"].fillna(merged["taxon_id"])
        merged["species_taxon_name"] = merged["species_taxon_name"].fillna(merged["taxon_name"]).fillna("")
        merged["broad_taxon_group"] = merged.apply(
            lambda row: assign_broad_taxon_group(
                row.get("taxon_id"),
                lineage=row.get("lineage", ""),
                lineage_ids=row.get("lineage_ids", ""),
                taxon_name=row.get("taxon_name", ""),
            ),
            axis=1,
        )
    else:
        merged["broad_taxon_group"] = merged["taxon_id"].fillna("Missing taxon")
        merged["species_taxon_id"] = merged["taxon_id"]
        merged["species_taxon_name"] = merged["taxon_id"]
    return merged


def make_taxon_distribution_table(
    annotated_ground_truth: pd.DataFrame,
    group_by: str = "taxon_id",
    top_n: int = 25,
    other_label: str = "Other taxa",
    segment_top_n: int = 5,
    segment_min_count: int = 10,
    other_segment_label: str = "Other species",
) -> pd.DataFrame:
    """Return protein counts by knowledge-gain subset, taxon/group, and species.

    The top ``segment_top_n`` taxon IDs are kept separately within each
    knowledge-gain/group bar when they have at least ``segment_min_count``
    proteins; all remaining taxa in that bar are pooled into
    ``other_segment_label``. Segment colors are keyed by taxon ID in the plot.
    """
    required = {"EntryID", "knowledge_gain", group_by, "taxon_id"}
    missing = required - set(annotated_ground_truth.columns)
    if missing:
        raise ValueError(f"annotated_ground_truth is missing columns: {sorted(missing)}")
    
    species_taxon_col = "species_taxon_id" if "species_taxon_id" in annotated_ground_truth.columns else "taxon_id"
    columns = list(dict.fromkeys(["EntryID", "knowledge_gain", group_by, "taxon_id", species_taxon_col]))
    if "taxon_name" in annotated_ground_truth.columns:
        columns.append("taxon_name")
    if "species_taxon_name" in annotated_ground_truth.columns:
        columns.append("species_taxon_name")
    
    protein_groups = annotated_ground_truth[columns].drop_duplicates()
    protein_groups[group_by] = protein_groups[group_by].fillna("Missing taxon").astype(str)
    protein_groups["taxon_id"] = protein_groups["taxon_id"].fillna("Missing taxon").astype(str)
    protein_groups["distribution_taxon_id"] = protein_groups[species_taxon_col].fillna(
        protein_groups["taxon_id"]
    ).astype(str)
    if "taxon_name" not in protein_groups.columns:
        protein_groups["taxon_name"] = ""
    protein_groups["taxon_name"] = protein_groups["taxon_name"].fillna("")
    if "species_taxon_name" not in protein_groups.columns:
        protein_groups["species_taxon_name"] = protein_groups["taxon_name"].map(clean_species_taxon_name)
    protein_groups["species_taxon_name"] = protein_groups["species_taxon_name"].fillna("")
    
    group_values = protein_groups["distribution_taxon_id"] if group_by == "taxon_id" else protein_groups[group_by]
    if group_by == "taxon_id" and top_n:
        top_groups = set(group_values.value_counts().head(top_n).index)
        protein_groups["plot_group"] = group_values.where(
            group_values.isin(top_groups), other_label
        )
    else:
        protein_groups["plot_group"] = group_values
    
    taxon_counts = (
        protein_groups.groupby(["knowledge_gain", "plot_group", "distribution_taxon_id"], as_index=False)
        .agg(
            protein_count=("EntryID", "nunique"),
            species_taxon_name=("species_taxon_name", _first_non_empty),
        )
        .sort_values(
            ["knowledge_gain", "plot_group", "protein_count", "distribution_taxon_id"],
            ascending=[True, True, False, True],
        )
    )
    taxon_counts["segment_rank"] = taxon_counts.groupby(["knowledge_gain", "plot_group"])[
        "protein_count"
    ].rank(method="first", ascending=False)
    keep_segment = (taxon_counts["segment_rank"] <= segment_top_n) & (
        taxon_counts["protein_count"] >= segment_min_count
    )
    taxon_counts["segment_taxon_id"] = taxon_counts["distribution_taxon_id"].where(
        keep_segment,
        other_segment_label,
    )
    taxon_counts["segment_label"] = taxon_counts.apply(
        lambda row: _format_taxon_segment_label(
            row["segment_taxon_id"],
            row["species_taxon_name"],
            other_segment_label=other_segment_label,
        ),
        axis=1,
    )
    taxon_counts["is_other_segment"] = taxon_counts["segment_taxon_id"].eq(other_segment_label)
    
    distribution = (
        taxon_counts.groupby(
            ["knowledge_gain", "plot_group", "segment_taxon_id", "segment_label", "is_other_segment"],
            as_index=False,
        )
        .agg(protein_count=("protein_count", "sum"))
        .sort_values(
            ["knowledge_gain", "plot_group", "is_other_segment", "protein_count", "segment_label"],
            ascending=[True, True, True, False, True],
        )
    )
    distribution["total_protein_count"] = distribution.groupby(["knowledge_gain", "plot_group"])[
        "protein_count"
    ].transform("sum")
    return distribution


def _format_taxon_segment_label(
    taxon_id: str,
    taxon_name: str,
    other_segment_label: str = "Other species",
) -> str:
    if taxon_id == other_segment_label:
        return other_segment_label
    if taxon_id == "Missing taxon":
        return "Missing taxon"
    # newline='\n('
    if taxon_name:
        return f"{taxon_name.split('(strain')[0]}"
        # return f"{taxon_name.split('(strain')[0].replace('(',newline,1)} ({taxon_id})" # strip strain info for readability
    return str(taxon_id)


def summarize_taxon_gain_coverage(
    annotated_ground_truth: pd.DataFrame,
    fasta_taxa: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how many FASTA taxa have proteins with gained GO terms."""
    all_taxa = set(fasta_taxa["taxon_id"].dropna().astype(str))
    gt_proteins = annotated_ground_truth[["EntryID", "knowledge_gain", "taxon_id"]].drop_duplicates()
    gt_proteins = gt_proteins.dropna(subset=["taxon_id"])
    
    rows = []
    total_taxa = len(all_taxa)
    for label, subset in gt_proteins.groupby("knowledge_gain", sort=False):
        gained_taxa = set(subset["taxon_id"].astype(str))
        rows.append(
            {
                "knowledge_gain": label,
                "taxa_with_gains": len(gained_taxa),
                "total_fasta_taxa": total_taxa,
                "percent_fasta_taxa_with_gains": (len(gained_taxa) / total_taxa * 100) if total_taxa else 0.0,
                "proteins_with_gains": subset["EntryID"].nunique(),
            }
        )
    
    gained_any = set(gt_proteins["taxon_id"].astype(str))
    rows.append(
        {
            "knowledge_gain": "Any knowledge gain",
            "taxa_with_gains": len(gained_any),
            "total_fasta_taxa": total_taxa,
            "percent_fasta_taxa_with_gains": (len(gained_any) / total_taxa * 100) if total_taxa else 0.0,
            "proteins_with_gains": gt_proteins["EntryID"].nunique(),
        }
    )
    return pd.DataFrame(rows)


def make_testset_species_distribution(
    merged_testset: pd.DataFrame,
    top_n: int = 20,
    other_label: str = "Other species",
) -> pd.DataFrame:
    """Summarize full test-set species composition from merged FASTA metadata."""
    required = {"EntryID", "species_taxon_id", "species_taxon_name"}
    missing = required - set(merged_testset.columns)
    if missing:
        raise ValueError(f"merged_testset is missing columns: {sorted(missing)}")

    species = merged_testset[["EntryID", "species_taxon_id", "species_taxon_name"]].drop_duplicates()
    species["species_taxon_id"] = species["species_taxon_id"].fillna("Missing taxon").astype(str)
    species["species_taxon_name"] = species["species_taxon_name"].fillna("")

    species_counts = (
        species.groupby("species_taxon_id", as_index=False)
        .agg(
            protein_count=("EntryID", "nunique"),
            species_taxon_name=("species_taxon_name", lambda values: _first_non_empty_or(values, "Unknown species")),
        )
        .sort_values(["protein_count", "species_taxon_id"], ascending=[False, True])
    )
    total = species_counts["protein_count"].sum()
    species_counts["percent_testset"] = species_counts["protein_count"] / total * 100 if total else 0.0
    species_counts["plot_label"] = species_counts.apply(
        lambda row: _format_taxon_segment_label(row["species_taxon_id"], row["species_taxon_name"]),
        axis=1,
    )

    if top_n and len(species_counts) > top_n:
        top = species_counts.head(top_n).copy()
        other = pd.DataFrame(
            [
                {
                    "species_taxon_id": other_label,
                    "protein_count": species_counts.iloc[top_n:]["protein_count"].sum(),
                    "species_taxon_name": other_label,
                    "percent_testset": species_counts.iloc[top_n:]["protein_count"].sum() / total * 100 if total else 0.0,
                    "plot_label": other_label,
                }
            ]
        )
        species_counts = pd.concat([top, other], ignore_index=True)

    return species_counts


def plot_testset_species_pie(
    species_distribution: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Test-set species distribution",
    figsize: tuple[float, float] = (10, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a pie chart of the full test-set species distribution."""
    required = {"plot_label", "protein_count", "percent_testset"}
    missing = required - set(species_distribution.columns)
    if missing:
        raise ValueError(f"species_distribution is missing columns: {sorted(missing)}")

    plot_df = species_distribution[species_distribution["protein_count"] > 0].copy()
    colors = sns.mpl_palette("tab20", n_colors=min(len(plot_df), 20))
    # if len(plot_df) > 10:
    #     colors = sns.husl_palette(len(plot_df), s=0.85, l=0.46)

    fig, ax = plt.subplots(figsize=figsize)
    wedges, _, autotexts = ax.pie(
        plot_df["protein_count"],
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(9)
        text.set_color("white")

    legend_labels = [
        f"{row.plot_label}: {int(row.protein_count)} ({row.percent_testset:.1f}%)"
        for row in plot_df.itertuples(index=False)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Species",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


def make_testset_broad_group_species_distribution(
    merged_testset: pd.DataFrame,
    broad_groups: Sequence[str] = DEFAULT_TESTSET_BROAD_GROUPS,
    top_n_species: int = 5,
) -> pd.DataFrame:
    """Summarize top test-set species within broad taxon groups.

    Counts are based on unique proteins in the full test set, not proteins in
    the ground-truth set.
    """
    required = {"EntryID", "broad_taxon_group", "species_taxon_id", "species_taxon_name"}
    missing = required - set(merged_testset.columns)
    if missing:
        raise ValueError(f"merged_testset is missing columns: {sorted(missing)}")

    proteins = merged_testset[
        ["EntryID", "broad_taxon_group", "species_taxon_id", "species_taxon_name"]
    ].drop_duplicates()
    proteins["broad_taxon_group"] = proteins["broad_taxon_group"].fillna("Other/unknown").astype(str)
    proteins["species_taxon_id"] = proteins["species_taxon_id"].fillna("Missing taxon").astype(str)
    proteins["species_taxon_name"] = proteins["species_taxon_name"].fillna("")
    proteins["species_label"] = proteins.apply(
        lambda row: row["species_taxon_name"].strip()
        if str(row["species_taxon_name"]).strip()
        else row["species_taxon_id"],
        axis=1,
    )

    counts = (
        proteins.groupby(["broad_taxon_group", "species_taxon_id"], as_index=False)
        .agg(
            protein_count=("EntryID", "nunique"),
            species_label=("species_label", _first_non_empty),
        )
        .sort_values(["broad_taxon_group", "protein_count", "species_label"], ascending=[True, False, True])
    )
    counts["species_rank"] = counts.groupby("broad_taxon_group")["protein_count"].rank(
        method="first",
        ascending=False,
    )
    counts = counts[counts["species_rank"] <= top_n_species].copy()

    group_order = {group: index for index, group in enumerate(broad_groups)}
    counts["broad_group_order"] = counts["broad_taxon_group"].map(group_order).fillna(len(group_order)).astype(int)
    counts = counts.sort_values(
        ["broad_group_order", "broad_taxon_group", "protein_count", "species_label"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return counts


def plot_testset_broad_group_species_bars(
    species_distribution: pd.DataFrame,
    output_path: str | Path | None = None,
    broad_groups: Sequence[str] = DEFAULT_TESTSET_BROAD_GROUPS,
    title: str = "Top species represented in the CAFA test set",
    figsize: tuple[float, float] = (15, 8),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot top species counts in each broad taxon group as a 2x3 bar grid."""
    required = {"broad_taxon_group", "species_label", "protein_count"}
    missing = required - set(species_distribution.columns)
    if missing:
        raise ValueError(f"species_distribution is missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    axes_list = list(axes.ravel())
    palette = sns.color_palette("colorblind", n_colors=max(len(broad_groups), 1))

    for index, (ax, broad_group) in enumerate(zip(axes_list, broad_groups)):
        subset = species_distribution[species_distribution["broad_taxon_group"] == broad_group].copy()
        subset = subset.sort_values(["protein_count", "species_label"], ascending=[False, True]).head(5)
        color = palette[index % len(palette)]

        if subset.empty:
            ax.text(0.5, 0.5, "No species", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_ylim(0, 1)
        else:
            x_positions = range(len(subset))
            bars = ax.bar(
                x_positions,
                subset["protein_count"],
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            y_max = max(subset["protein_count"].max(), 1)
            ax.set_ylim(0, y_max * 1.18)
            ax.set_xticks(list(x_positions))
            ax.set_xticklabels(subset["species_label"], rotation=35, ha="right")
            for bar, count in zip(bars, subset["protein_count"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.02,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_title(broad_group)
        ax.set_ylabel("Proteins")
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)

    for ax in axes_list[len(broad_groups):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes_list


def plot_taxon_distribution(
    distribution: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    annotation_fontsize: int = 13,
    total_fontsize: int = 11,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot stacked horizontal taxon distributions for each knowledge-gain subset."""
    required = {
        "knowledge_gain",
        "plot_group",
        "segment_taxon_id",
        "segment_label",
        "is_other_segment",
        "protein_count",
    }
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(f"distribution is missing columns: {sorted(missing)}")

    labels = _ordered_knowledge_labels(distribution)
    if figsize is None:
        figsize = _default_taxon_distribution_figsize(distribution, labels)

    fig, axes_list, legend_ax = _make_taxon_distribution_axes(labels, figsize)
    color_map = _make_segment_color_map(distribution)
    label_map = (
        distribution[["segment_taxon_id", "segment_label"]]
        .drop_duplicates()
        .set_index("segment_taxon_id")["segment_label"]
        .to_dict()
    )

    shared_top_y_labels = _shared_top_row_y_labels(distribution, labels)
    for index, (ax, label) in enumerate(zip(axes_list, labels)):
        y_labels = shared_top_y_labels if index in (0, 1) else None
        show_y_labels = index != 1
        _plot_stacked_taxon_panel(
            ax,
            distribution[distribution["knowledge_gain"] == label],
            label,
            color_map,
            y_labels=y_labels,
            show_y_labels=show_y_labels,
            annotation_fontsize=annotation_fontsize,
            total_fontsize=total_fontsize,
        )

    _add_segment_legend(legend_ax, color_map, label_map)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes_list


def _ordered_knowledge_labels(distribution: pd.DataFrame) -> list[str]:
    present = list(distribution["knowledge_gain"].drop_duplicates())
    ordered = [label for label in DEFAULT_KNOWLEDGE_LABELS if label in present]
    ordered.extend(label for label in present if label not in ordered)
    return ordered


def _shared_top_row_y_labels(distribution: pd.DataFrame, labels: Sequence[str]) -> list[str] | None:
    if list(labels[:3]) != list(DEFAULT_KNOWLEDGE_LABELS) or len(labels) != 3:
        return None
    
    top_row = distribution[distribution["knowledge_gain"].isin(labels[:2])]
    if top_row.empty:
        return []
    
    totals = (
        top_row.groupby("plot_group", as_index=False)
        .agg(total=("protein_count", "sum"))
        .sort_values(["total", "plot_group"], ascending=[True, True])
    )
    return list(totals["plot_group"])


def _default_taxon_distribution_figsize(distribution: pd.DataFrame, labels: Sequence[str]) -> tuple[float, float]:
    if list(labels[:3]) == list(DEFAULT_KNOWLEDGE_LABELS) and len(labels) == 3:
        top_groups = distribution[distribution["knowledge_gain"].isin(labels[:2])].groupby("knowledge_gain")[
            "plot_group"
        ].nunique()
        bottom_groups = distribution[distribution["knowledge_gain"] == labels[2]]["plot_group"].nunique()
        height = max(7.5, 2.8 + 0.42 * max(top_groups.max() if len(top_groups) else 1, 1) + 0.42 * bottom_groups)
        return (13, height)
    
    max_groups = distribution.groupby("knowledge_gain")["plot_group"].nunique().max()
    return (10, max(3.2 * len(labels), 0.38 * max_groups * len(labels)))


def _make_taxon_distribution_axes(
    labels: Sequence[str],
    figsize: tuple[float, float],
) -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    if list(labels[:3]) == list(DEFAULT_KNOWLEDGE_LABELS) and len(labels) == 3:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1, 1, 0.62],
            height_ratios=[1, 1],
            hspace=0.35,
            wspace=0.12,
        )
        left_ax = fig.add_subplot(grid[0, 0])
        axes = [
            left_ax,
            fig.add_subplot(grid[0, 1], sharey=left_ax),
            fig.add_subplot(grid[1, 0:2]),
        ]
        legend_ax = fig.add_subplot(grid[:, 2])
        return fig, axes, legend_ax
    
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(len(labels), 2, width_ratios=[1, 0.45], hspace=0.35, wspace=0.25)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(len(labels))]
    legend_ax = fig.add_subplot(grid[:, 1])
    return fig, axes, legend_ax


def _make_segment_color_map(distribution: pd.DataFrame) -> dict[str, Any]:
    segment_totals = (
        distribution.groupby(["segment_taxon_id", "segment_label"], as_index=False)
        .agg(total=("protein_count", "sum"))
        .sort_values(["segment_taxon_id", "total"], ascending=[True, False])
        .drop_duplicates("segment_taxon_id")
        .sort_values("total", ascending=False)
    )
    special_colors = {
        "Other species": (0.85, 0.85, 0.85),
        # "Missing taxon": (0.45, 0.45, 0.45),
    }
    regular_segments = [segment_id for segment_id in segment_totals["segment_taxon_id"] if segment_id not in special_colors]
    regular_segments += [segment_id for segment_id in segment_totals["segment_taxon_id"] if segment_id in special_colors]
    if len(regular_segments) <= 10:
        palette = sns.color_palette("colorblind", n_colors=max(len(regular_segments), 1))
    else:
        palette = sns.mpl_palette("tab20", n_colors=max(len(regular_segments), 1))
    color_map = {segment_id: palette[index] for index, segment_id in enumerate(regular_segments)}
    color_map.update(special_colors)
    # color_map.update(
    #     {key: value for key, value in special_colors.items() if key in set(segment_totals["segment_taxon_id"])}
    # )
    return color_map


def _plot_stacked_taxon_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    label: str,
    color_map: Mapping[str, Any],
    y_labels: Sequence[str] | None = None,
    show_y_labels: bool = True,
    annotation_fontsize: int = 3,
    total_fontsize: int = 11,
) -> None:
    totals = (
        subset.groupby("plot_group", as_index=False)
        .agg(total=("protein_count", "sum"))
        .sort_values(["total", "plot_group"], ascending=[True, True])
    )
    if y_labels is None:
        y_labels = list(totals["plot_group"])
    else:
        y_labels = list(y_labels)
    y_positions = list(range(len(y_labels)))
    y_lookup = dict(zip(y_labels, y_positions))
    left_offsets = {plot_group: 0 for plot_group in y_labels}
    x_max = max(totals["total"].max(), 1)

    for plot_group in y_labels:
        bar_segments = subset[subset["plot_group"] == plot_group].sort_values(
            ["is_other_segment", "protein_count", "segment_label"],
            ascending=[True, False, True],
        )
        for _, row in bar_segments.iterrows():
            width = row["protein_count"]
            segment_id = row["segment_taxon_id"]
            left = left_offsets[plot_group]
            ax.barh(
                y_lookup[plot_group],
                width,
                left=left,
                color=color_map.get(segment_id, (0.5, 0.5, 0.5)),
                edgecolor="white",
                linewidth=0.5,
                height=0.72,
            )
            if width >= max(2, x_max * 0.08):
                ax.text(
                    left + width / 2,
                    y_lookup[plot_group],
                    str(int(width)),
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color="white",
                )
            left_offsets[plot_group] += width

    for _, row in totals.iterrows():
        if row["plot_group"] not in y_lookup or row["total"] == 0:
            continue
        ax.text(
            row["total"] + x_max * 0.01,
            y_lookup[row["plot_group"]],
            str(int(row["total"])),
            va="center",
            fontsize=total_fontsize,
        )

    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels(y_labels)
    ax.tick_params(axis="y", which="both", left=show_y_labels, labelleft=show_y_labels)
    ax.set_title(label)
    ax.set_xlabel("Proteins gaining new GO terms")
    ax.set_ylabel("")
    ax.set_xlim(0, x_max * 1.12)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)


def _add_segment_legend(
    legend_ax: plt.Axes,
    color_map: Mapping[str, Any],
    label_map: Mapping[str, str],
    max_entries: int = 35,
) -> None:
    legend_ax.axis("off")
    segment_items = list(color_map.items())
    truncated_count = max(len(segment_items) - max_entries, 0)
    segment_items = segment_items[:max_entries]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in segment_items]
    labels = [label_map.get(segment_id, segment_id) for segment_id, _ in segment_items]
    for i, label in enumerate(labels):
        if label == "Solanum lycopersicum (Tomato) (Lycopersicon esculentum)":
            labels[i] = "Solanum lycopersicum (Tomato)"
    # # Only show species names in the legend, skip other information
    # print(labels)
    # labels = [label.split("(")[1].split(")")[0] for label in labels if "(" in label and ")" in label]
    if truncated_count:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="white", alpha=0.0))
        labels.append(f"+{truncated_count} more taxa")
    if not handles:
        return
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        title="Species",
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )


def analyze_ground_truth_taxa(
    no_knowledge_path: str | Path,
    limited_knowledge_path: str | Path,
    partial_knowledge_path: str | Path,
    fasta_path: str | Path,
    lineage_path: str | Path | None = None,
    output_plot: str | Path | None = None,
    group_by: str | None = None,
    top_n: int = 25,
    segment_top_n: int = 5,
    segment_min_count: int = 10,
) -> dict[str, pd.DataFrame]:
    """Create taxon-distribution data, summary tables, and an optional plot.

    Use ``lineage_path`` and ``group_by="broad_taxon_group"`` when you want
    animals/plants/fungi/bacteria-style grouping. Without lineage data, the
    default is top-N taxon IDs plus an ``Other taxa`` bucket.
    """
    ground_truth = read_ground_truth_sets(no_knowledge_path, limited_knowledge_path, partial_knowledge_path)
    fasta_taxa = read_fasta_taxa(fasta_path)
    lineage = load_uniprot_lineage(lineage_path) if lineage_path else None
    annotated = attach_taxa_to_ground_truth(ground_truth, fasta_taxa, lineage=lineage)
    
    if group_by is None:
        group_by = "broad_taxon_group" if lineage is not None else "taxon_id"
    
    distribution = make_taxon_distribution_table(
        annotated,
        group_by=group_by,
        top_n=top_n,
        segment_top_n=segment_top_n,
        segment_min_count=segment_min_count,
    )
    coverage = summarize_taxon_gain_coverage(annotated, fasta_taxa)
    if output_plot:
        plot_taxon_distribution(distribution, output_path=output_plot)
    
    return {
        "annotated_ground_truth": annotated,
        "taxon_distribution": distribution,
        "taxon_gain_coverage": coverage,
    }


def merge_testset_taxa(
    fasta_path: str | Path,
    lineage_path: str | Path | None = None,
    species_name: str | Path | None = None,
) -> pd.DataFrame:
    """Attach species-level taxon metadata to every protein in a test-set FASTA."""
    fasta_taxa = read_fasta_taxa(fasta_path)
    lineage = load_uniprot_lineage(lineage_path) if lineage_path else None
    merged = fasta_taxa.merge(lineage, on="taxon_id", how="left", validate="many_to_one") if lineage is not None else fasta_taxa.copy()

    if "species_taxon_id" not in merged.columns:
        merged["species_taxon_id"] = merged["taxon_id"]
    else:
        merged["species_taxon_id"] = merged["species_taxon_id"].fillna(merged["taxon_id"])

    if "species_taxon_name" not in merged.columns:
        merged["species_taxon_name"] = ""
    merged["species_taxon_name"] = merged["species_taxon_name"].fillna("")

    if species_name:
        species_testset = pd.read_csv(species_name, sep="\t", dtype=str)
        if {"ID", "Species"} <= set(species_testset.columns):
            id_to_name = species_testset.dropna(subset=["ID"]).set_index("ID")["Species"].to_dict()
            merged["species_taxon_name"] = merged.apply(
                lambda row: row["species_taxon_name"]
                if str(row["species_taxon_name"]).strip()
                else id_to_name.get(str(row["species_taxon_id"]), id_to_name.get(str(row["taxon_id"]), "")),
                axis=1,
            )

    merged["species_taxon_name"] = merged.apply(
        lambda row: row["species_taxon_name"] if str(row["species_taxon_name"]).strip() else str(row["species_taxon_id"]),
        axis=1,
    )
    if "broad_taxon_group" not in merged.columns:
        if lineage is not None:
            merged["broad_taxon_group"] = merged.apply(
                lambda row: assign_broad_taxon_group(
                    row.get("taxon_id"),
                    lineage=row.get("lineage", ""),
                    lineage_ids=row.get("lineage_ids", ""),
                    taxon_name=row.get("taxon_name", ""),
                ),
                axis=1,
            )
        else:
            merged["broad_taxon_group"] = merged["taxon_id"].fillna("Missing taxon").astype(str)
    return merged


def analyze_testset_taxa(
    fasta_path: str | Path,
    lineage_path: str | Path | None = None,
    species_name: str | Path | None = None,
    output_pie: str | Path | None = None,
    output_broad_group_species_plot: str | Path | None = None,
    top_n: int = 20,
    top_n_species_per_group: int = 5,
) -> dict[str, pd.DataFrame]:
    """Summarize and plot species distribution of the full test-set FASTA."""
    merged = merge_testset_taxa(fasta_path, lineage_path=lineage_path, species_name=species_name)
    species_distribution = make_testset_species_distribution(merged, top_n=top_n)
    broad_group_species_distribution = make_testset_broad_group_species_distribution(
        merged,
        top_n_species=top_n_species_per_group,
    )
    if output_pie:
        plot_testset_species_pie(species_distribution, output_path=output_pie)
    if output_broad_group_species_plot:
        plot_testset_broad_group_species_bars(
            broad_group_species_distribution,
            output_path=output_broad_group_species_plot,
        )

    return {
        "merged_testset": merged,
        "species_distribution": species_distribution,
        "broad_group_species_distribution": broad_group_species_distribution,
    }


def read_cafa_team_metadata(
    metadata_path: str | Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Read CAFA team metadata into team-member and team-rank dictionaries.
    
    The expected CSV columns are ``Rank``, ``TeamName``, and
    ``TeamMemberUserNames``. Member names are split on commas. If a team appears
    more than once, members are unioned and the best rank is retained.
    """
    metadata = pd.read_csv(metadata_path, dtype=str)
    required = {"Rank", "TeamName", "TeamMemberUserNames"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
    
    team_members: dict[str, set[str]] = {}
    team_ranks: dict[str, int] = {}
    for row in metadata.itertuples(index=False):
        team_name = str(getattr(row, "TeamName")).strip()
        if not team_name or team_name.lower() == "nan":
            continue
            
        members = _parse_team_member_usernames(getattr(row, "TeamMemberUserNames"))
        team_members.setdefault(team_name, set()).update(members)
        
        rank = pd.to_numeric(getattr(row, "Rank"), errors="coerce")
        if pd.notna(rank):
            rank = int(rank)
            if team_name not in team_ranks or rank < team_ranks[team_name]:
                team_ranks[team_name] = rank
    
    return {team: sorted(members) for team, members in team_members.items()}, team_ranks


def compare_cafa_team_member_overlap(
    cafa5_metadata_path: str | Path,
    cafa6_metadata_path: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare CAFA5 and CAFA6 team-member overlap with a Jaccard index."""
    cafa5_team_members, cafa5_team_ranks = read_cafa_team_metadata(cafa5_metadata_path)
    cafa6_team_members, cafa6_team_ranks = read_cafa_team_metadata(cafa6_metadata_path)

    cafa5_members = _flatten_member_dict(cafa5_team_members)
    cafa6_members = _flatten_member_dict(cafa6_team_members)
    shared_members = cafa5_members & cafa6_members
    union_members = cafa5_members | cafa6_members
    jaccard_index = len(shared_members) / len(union_members) if union_members else 0.0

    # Plot a Venn diagram of team-member overlap
    from matplotlib_venn import venn2
    fig, ax = plt.subplots(figsize=(6, 6))
    out=venn2(
        subsets=(len(cafa5_members - cafa6_members), len(cafa6_members - cafa5_members), len(shared_members)),
        set_labels=("CAFA5 participants", "CAFA6 participants"),
        ax=ax,
    )
    for text in out.subset_labels:
        text.set_fontsize(14)
    ax.set_title(f"CAFA5 vs CAFA6 Member Overlap\nJaccard Index: {jaccard_index:.3f}")
    fig.tight_layout()
    fig.savefig("data/cafa6/for_pub/cafa5_cafa6_venn.png", dpi=300, bbox_inches="tight")
    
    # Plot boxplot of team ranks for 3 groups: CAFA5-only, CAFA6-only, and shared members
    rank_data = []
    member_to_team_cafa5=dict()
    for team, m in cafa5_team_members.items():
        for member in m:
            member_to_team_cafa5[member]=team
    member_to_team_cafa6=dict()
    for team, m in cafa6_team_members.items():
        for member in m:
            member_to_team_cafa6[member]=team
    shared_team = {member: member_to_team_cafa6[member] for member in shared_members}        
    # for member in cafa5_members - cafa6_members:
    #     rank_data.append({"member": member, "cafa": "CAFA5-only", "rank": member_to_team_cafa5.get(member, None)})
    for member in cafa6_members - cafa5_members:
        rank_data.append({"member": member, "cafa": "CAFA6-only", "rank": cafa6_team_ranks.get(member_to_team_cafa6.get(member, None), None)})
    for member in shared_members:
        rank_data.append({"member": member, "cafa": "Shared", "rank": cafa6_team_ranks.get(member_to_team_cafa6.get(member, None), None)})
    rank_df = pd.DataFrame(rank_data)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.swarmplot(x="cafa", y="rank", data=rank_df, ax=ax)
    # ax.set_title("Team Ranks by CAFA Membership")
    # ax.set_ylabel("Team Rank")
    # ax.set_xlabel("CAFA Membership")
    # fig.tight_layout()
    # fig.savefig("data/cafa6/for_pub/cafa5_cafa6_rank_boxplot.png", dpi=300, bbox_inches="tight")

    # return {
    #     "cafa5_team_members": cafa5_team_members,
    #     "cafa6_team_members": cafa6_team_members,
    #     "cafa5_team_ranks": cafa5_team_ranks,
    #     "cafa6_team_ranks": cafa6_team_ranks,
    #     "cafa5_members": cafa5_members,
    #     "cafa6_members": cafa6_members,
    #     "shared_members": shared_members,
    #     "cafa5_only_members": cafa5_members - cafa6_members,
    #     "cafa6_only_members": cafa6_members - cafa5_members,
    #     "jaccard_index": jaccard_index,
    #     "summary": summary,
    # }


def _parse_team_member_usernames(member_value: Any) -> set[str]:
    if pd.isna(member_value):
        return set()
    return {member.strip() for member in str(member_value).split(",") if member.strip()}


def _flatten_member_dict(team_members: Mapping[str, Sequence[str]]) -> set[str]:
    return {member for members in team_members.values() for member in members}


def summarize_t0_state_ground_truth_flow(
    fasta_path: str | Path,
    t0_known_path: str | Path,
    no_knowledge_path: str | Path,
    limited_knowledge_path: str | Path,
    partial_knowledge_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Summarize where t0-annotated and t0-unannotated test proteins end up.

    A protein is ``Unannotated at t0`` when it is present in the FASTA but absent
    from the t0 known-annotation TSV. A protein is ``Annotated at t0`` when it has
    at least one t0 GO annotation in any aspect.
    """
    test_ids = set(read_fasta_taxa(fasta_path)["EntryID"])
    t0_known = read_ground_truth(t0_known_path)
    t0_annotated_ids = set(t0_known[t0_known["EntryID"].isin(test_ids)]["EntryID"])
    t0_unannotated_ids = test_ids - t0_annotated_ids

    nk_ids = set(read_ground_truth(no_knowledge_path)["EntryID"]) & test_ids
    lk_ids = set(read_ground_truth(limited_knowledge_path)["EntryID"]) & test_ids
    pk_ids = set(read_ground_truth(partial_knowledge_path)["EntryID"]) & test_ids
    lk_pk_ids = lk_ids | pk_ids
    any_gt_ids = nk_ids | lk_pk_ids

    state_counts = pd.DataFrame(
        [
            _make_t0_state_count_row("Unannotated at t0", t0_unannotated_ids, len(test_ids)),
            _make_t0_state_count_row("Annotated at t0", t0_annotated_ids, len(test_ids)),
        ]
    )

    flow_rows = []
    flow_rows.extend(
        _make_t0_flow_rows(
            t0_state="Unannotated at t0",
            state_ids=t0_unannotated_ids,
            destinations=(
                ("No knowledge ground truth", nk_ids),
                ("No gained GO terms in ground truth", test_ids - any_gt_ids),
            ),
        )
    )
    flow_rows.extend(
        _make_t0_flow_rows(
            t0_state="Annotated at t0",
            state_ids=t0_annotated_ids,
            destinations=(
                ("Limited knowledge ground truth", lk_ids),
                ("Partial knowledge ground truth", pk_ids),
                ("Limited or partial knowledge ground truth", lk_pk_ids),
                ("No gained GO terms in ground truth", test_ids - any_gt_ids),
            ),
        )
    )
    flow = pd.DataFrame(flow_rows)

    validation = pd.DataFrame(
        [
            _make_t0_flow_row("Unannotated at t0", "Unexpected LK or PK ground truth", t0_unannotated_ids, lk_pk_ids),
            _make_t0_flow_row("Annotated at t0", "Unexpected NK ground truth", t0_annotated_ids, nk_ids),
        ]
    )

    return {
        "t0_state_counts": state_counts,
        "t0_ground_truth_flow": flow,
        "t0_ground_truth_validation": validation,
    }


def _make_t0_state_count_row(t0_state: str, state_ids: set[str], total_test_proteins: int) -> dict[str, Any]:
    count = len(state_ids)
    return {
        "t0_state": t0_state,
        "protein_count": count,
        "total_test_proteins": total_test_proteins,
        "percent_test_proteins": count / total_test_proteins * 100 if total_test_proteins else 0.0,
    }


def _make_t0_flow_rows(
    t0_state: str,
    state_ids: set[str],
    destinations: Sequence[tuple[str, set[str]]],
) -> list[dict[str, Any]]:
    return [_make_t0_flow_row(t0_state, destination, state_ids, destination_ids) for destination, destination_ids in destinations]


def _make_t0_flow_row(
    t0_state: str,
    destination: str,
    state_ids: set[str],
    destination_ids: set[str],
) -> dict[str, Any]:
    count = len(state_ids & destination_ids)
    total = len(state_ids)
    return {
        "t0_state": t0_state,
        "destination": destination,
        "protein_count": count,
        "total_in_t0_state": total,
        "percent_of_t0_state": count / total * 100 if total else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CAFA NK/LK/PK taxon distributions.")
    parser.add_argument("--no-knowledge", required=True, help="No-knowledge ground-truth TSV.")
    parser.add_argument("--limited-knowledge", required=True, help="Limited-knowledge ground-truth TSV.")
    parser.add_argument("--partial-knowledge", required=True, help="Partial-knowledge ground-truth TSV.")
    parser.add_argument("--fasta", required=True, help="FASTA file containing EntryID and taxon ID in headers.")
    parser.add_argument("--lineage", default=None, help="Optional UniProt lineage TSV for broad taxon grouping.")
    parser.add_argument("--species-name", default=None, help="Optional TSV with ID and Species columns for test-set species labels.")
    parser.add_argument("--output-plot", required=True, help="Path for the output plot, e.g. taxon_distribution.png.")
    parser.add_argument("--testset-output-pie", default=None, help="Optional path for a full-test-set species pie chart.")
    parser.add_argument(
        "--testset-output-broad-group-species-plot",
        default=None,
        help="Optional path for a 2x3 broad-group top-species bar chart for the full test set.",
    )
    parser.add_argument("--output-prefix", default=None, help="Optional prefix for summary TSV outputs.")
    parser.add_argument("--t0-known", default=None, help="Optional t0 known-annotation TSV for t0-state flow summaries.")
    parser.add_argument(
        "--group-by",
        choices=("taxon_id", "broad_taxon_group"),
        default=None,
        help="Grouping column. Defaults to broad groups when --lineage is given; otherwise taxon_id.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Top taxon IDs to show when grouping by taxon_id.")
    parser.add_argument(
        "--segment-top-n",
        type=int,
        default=5,
        help="Top taxon IDs to color separately inside each bar; remaining taxa are pooled.",
    )
    parser.add_argument(
        "--segment-min-count",
        type=int,
        default=10,
        help="Minimum proteins required for a taxon ID to receive its own colored segment.",
    )
    parser.add_argument("--testset-top-n", type=int, default=20, help="Top species to show in the test-set pie chart.")
    parser.add_argument(
        "--testset-top-n-per-group",
        type=int,
        default=5,
        help="Top species to show within each broad taxon group in the test-set bar chart.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = analyze_ground_truth_taxa(
        no_knowledge_path=args.no_knowledge,
        limited_knowledge_path=args.limited_knowledge,
        partial_knowledge_path=args.partial_knowledge,
        fasta_path=args.fasta,
        lineage_path=args.lineage,
        output_plot=args.output_plot,
        group_by=args.group_by,
        top_n=args.top_n,
        segment_top_n=args.segment_top_n,
        segment_min_count=args.segment_min_count,
    )

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        results["taxon_distribution"].to_csv(f"{prefix}_taxon_distribution.tsv", sep="\t", index=False)
        results["taxon_gain_coverage"].to_csv(f"{prefix}_taxon_gain_coverage.tsv", sep="\t", index=False)

    if args.testset_output_pie or args.testset_output_broad_group_species_plot:
        testset_results = analyze_testset_taxa(
            fasta_path=args.fasta,
            lineage_path=args.lineage,
            species_name=args.species_name,
            output_pie=args.testset_output_pie,
            output_broad_group_species_plot=args.testset_output_broad_group_species_plot,
            top_n=args.testset_top_n,
            top_n_species_per_group=args.testset_top_n_per_group,
        )
        if args.output_prefix:
            testset_results["species_distribution"].to_csv(f"{prefix}_testset_species_distribution.tsv", sep="\t", index=False)
            testset_results["broad_group_species_distribution"].to_csv(
                f"{prefix}_testset_broad_group_species_distribution.tsv",
                sep="\t",
                index=False,
            )
            testset_results["merged_testset"].to_csv(f"{prefix}_testset_taxa.tsv", sep="\t", index=False)

    if args.t0_known:
        t0_results = summarize_t0_state_ground_truth_flow(
            fasta_path=args.fasta,
            t0_known_path=args.t0_known,
            no_knowledge_path=args.no_knowledge,
            limited_knowledge_path=args.limited_knowledge,
            partial_knowledge_path=args.partial_knowledge,
        )
        if args.output_prefix:
            t0_results["t0_state_counts"].to_csv(f"{prefix}_t0_state_counts.tsv", sep="\t", index=False)
            t0_results["t0_ground_truth_flow"].to_csv(f"{prefix}_t0_ground_truth_flow.tsv", sep="\t", index=False)
            t0_results["t0_ground_truth_validation"].to_csv(
                f"{prefix}_t0_ground_truth_validation.tsv",
                sep="\t",
                index=False,
            )

    print(results["taxon_gain_coverage"].to_string(index=False))


if __name__ == "__main__":
    main()
