"""Create presentation-ready exploratory text-analysis figures.

Examples:
    python visualize_data.py
    python visualize_data.py --output-dir artifacts/text_analysis --dpi 220
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import config as cfg
from src.data import load_domain_file
from src.text_analysis import CorpusRecord, count_words, records_to_summary_rows, summarize_corpus


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "text_analysis"
SPLITS = ("train", "dev", "test")

NAVY = "#16324F"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
CORAL = "#E76F51"
PALE = "#EEF4F8"
GREY = "#667085"
SPLIT_COLORS = {"train": BLUE, "dev": TEAL, "test": GOLD}
SENTIMENT_COLORS = {
    "positive": TEAL,
    "negative": CORAL,
    "neutral": BLUE,
    "conflict": GOLD,
}


def load_records(data_dir: Path):
    records = []
    missing = []
    for domain in cfg.DOMAINS:
        for split in SPLITS:
            path = data_dir / cfg.DOMAIN_FILES[domain][split]
            if not path.is_file():
                missing.append(str(path))
                continue
            records.extend(
                CorpusRecord(domain, split, example)
                for example in load_domain_file(str(path), domain)
            )
    if missing:
        raise FileNotFoundError(
            "Missing configured dataset files:\n" + "\n".join(missing)
            + "\nRun `python download_data.py` first."
        )
    return records


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": "#D0D5DD",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_title(fig, title, subtitle):
    fig.text(0.055, 0.955, title, ha="left", va="top", fontsize=24, weight="bold", color=NAVY)
    fig.text(0.055, 0.91, subtitle, ha="left", va="top", fontsize=11.5, color=GREY)


def save_figure(fig, output_dir: Path, name: str, dpi: int):
    path = output_dir / name
    # Keep the exact 16:9 canvas so images drop into presentation software
    # without stretching or unexpected cropping.
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def draw_kpi(ax, value, label, detail="", color=BLUE):
    ax.axis("off")
    card = FancyBboxPatch(
        (0.01, 0.04), 0.98, 0.92,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=PALE, edgecolor="#D9E5ED", linewidth=1,
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(0.07, 0.68, value, transform=ax.transAxes, fontsize=24, weight="bold", color=color, va="center")
    ax.text(0.07, 0.39, label, transform=ax.transAxes, fontsize=10.5, weight="bold", color=NAVY, va="center")
    if detail:
        ax.text(0.07, 0.17, detail, transform=ax.transAxes, fontsize=8.5, color=GREY, va="center")


def coverage_arrays(summary):
    lookup = {(row["domain"], row["split"]): row["examples"] for row in summary["by_domain_split"]}
    return {split: np.array([lookup[(domain, split)] for domain in cfg.DOMAINS]) for split in SPLITS}


def plot_dashboard(summary, output_dir, dpi):
    overall = summary["overall"]
    validation = summary["validation"]
    lengths = summary["word_length"]["by_domain"]
    sentiments = summary["sentiment_counts"]["overall"]
    scope = summary["baseline_scope"]

    fig = plt.figure(figsize=(16, 9))
    add_title(fig, "M-ABSA data is loaded, validated, and ready for modelling", "Pre-model evidence from all seven English domains and official train/dev/test files")
    grid = fig.add_gridspec(3, 12, left=0.07, right=0.97, top=0.83, bottom=0.10, hspace=0.72, wspace=1.05)

    kpis = [
        (f"{validation['loaded_source_files']}/{validation['expected_source_files']}", "source files loaded", "0 parse failures", TEAL),
        (f"{overall['examples']:,}", "text records", f"{overall['domains']} domains", BLUE),
        (f"{overall['total_triplets']:,}", "aspect triplets", f"{overall['unique_categories']} categories", ORANGE),
        (f"{overall['model_ready_percent']:.1f}%", "triplets model-ready", "aligned single-pair scope", CORAL),
    ]
    for index, item in enumerate(kpis):
        draw_kpi(fig.add_subplot(grid[0, index * 3:(index + 1) * 3]), *item)

    ax_coverage = fig.add_subplot(grid[1:, 0:5])
    arrays = coverage_arrays(summary)
    y = np.arange(len(cfg.DOMAINS))
    left = np.zeros(len(cfg.DOMAINS))
    for split in SPLITS:
        ax_coverage.barh(y, arrays[split], left=left, color=SPLIT_COLORS[split], label=split.title(), height=0.64)
        left += arrays[split]
    ax_coverage.set_yticks(y, [domain.title() for domain in cfg.DOMAINS])
    ax_coverage.invert_yaxis()
    ax_coverage.set_ylim(len(cfg.DOMAINS) - 0.5, -0.95)
    ax_coverage.set_title("Coverage by domain and split", loc="left")
    ax_coverage.set_xlabel("Text records")
    ax_coverage.grid(axis="x", color="#E4E7EC", linewidth=0.8)
    ax_coverage.set_axisbelow(True)
    ax_coverage.legend(frameon=False, ncols=3, loc="upper right")
    for yi, total in zip(y, left):
        ax_coverage.text(total - max(left) * 0.012, yi, f"{int(total):,}", ha="right", va="center", fontsize=8.5, color=NAVY, weight="bold")

    ax_length = fig.add_subplot(grid[1:, 5:8])
    medians = [lengths[d]["median"] for d in cfg.DOMAINS]
    p95s = [lengths[d]["p95"] for d in cfg.DOMAINS]
    ypos = np.arange(len(cfg.DOMAINS))
    ax_length.hlines(ypos, medians, p95s, color="#AFC6D5", linewidth=6)
    ax_length.scatter(medians, ypos, color=BLUE, s=65, label="Median", zorder=3)
    ax_length.scatter(p95s, ypos, color=CORAL, marker="D", s=50, label="95th percentile", zorder=3)
    short_domains = ["Course.", "Hotel", "Laptop", "Rest.", "Phone", "Sight", "Food"]
    ax_length.set_yticks(ypos, short_domains, fontsize=8.5)
    ax_length.invert_yaxis()
    ax_length.set_title("Text length", loc="left")
    ax_length.set_xlabel("Words per record")
    ax_length.grid(axis="x", color="#E4E7EC", linewidth=0.8)
    ax_length.set_axisbelow(True)
    ax_length.legend(frameon=False, loc="lower right", fontsize=9)

    ax_sentiment = fig.add_subplot(grid[1, 9:12])
    sentiment_order = [label for label in cfg.SENTIMENTS if label in sentiments]
    values = [sentiments[label] for label in sentiment_order]
    wedges, _ = ax_sentiment.pie(
        values,
        colors=[SENTIMENT_COLORS.get(label, GREY) for label in sentiment_order],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax_sentiment.text(0, 0.07, f"{sum(values):,}", ha="center", va="center", fontsize=18, weight="bold", color=NAVY)
    ax_sentiment.text(0, -0.15, "triplets", ha="center", va="center", fontsize=9, color=GREY)
    ax_sentiment.set_title("Raw sentiment mix", loc="left")
    ax_sentiment.legend(wedges, [label.title() for label in sentiment_order], frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.96, 0.5))

    ax_scope = fig.add_subplot(grid[2, 9:12])
    scope_labels = ["Model-ready", "Implicit / NULL", "Unaligned", "Duplicate", "Extra label pair", "Overlapping"]
    scope_values = [
        scope["included_explicit_triplets"],
        scope["implicit_triplets_excluded"],
        scope["unaligned_triplets_excluded"],
        scope["duplicate_triplets_excluded"],
        scope["additional_label_pairs_excluded"],
        scope["overlapping_triplets_excluded"],
    ]
    order = np.argsort(scope_values)[::-1]
    scope_labels = [scope_labels[i] for i in order if scope_values[i] > 0]
    scope_values = [scope_values[i] for i in order if scope_values[i] > 0]
    colors = [TEAL if label == "Model-ready" else CORAL for label in scope_labels]
    bars = ax_scope.barh(np.arange(len(scope_labels)), scope_values, color=colors, height=0.58)
    ax_scope.set_yticks(np.arange(len(scope_labels)), scope_labels)
    ax_scope.tick_params(axis="y", labelsize=8.5, pad=2)
    ax_scope.invert_yaxis()
    ax_scope.set_title("Baseline annotation scope", loc="left")
    ax_scope.set_xlabel("Triplets")
    ax_scope.grid(axis="x", color="#E4E7EC", linewidth=0.8)
    ax_scope.set_axisbelow(True)
    for bar, value in zip(bars, scope_values):
        ax_scope.text(value + max(scope_values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=8.5, color=GREY)

    save_figure(fig, output_dir, "slide_01_data_readiness_dashboard.png", dpi)


def plot_coverage(summary, output_dir, dpi):
    arrays = coverage_arrays(summary)
    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.12, wspace=0.28)
    add_title(fig, "Dataset coverage is broad and split structure is consistent", "Every configured domain contributes train, development, and test records")
    y = np.arange(len(cfg.DOMAINS))
    left = np.zeros(len(cfg.DOMAINS))
    for split in SPLITS:
        ax.barh(y, arrays[split], left=left, color=SPLIT_COLORS[split], label=split.title(), height=0.66)
        left += arrays[split]
    ax.set_yticks(y, [domain.title() for domain in cfg.DOMAINS])
    ax.invert_yaxis()
    ax.set_xlabel("Number of text records")
    ax.set_title("Absolute volume", loc="left")
    ax.grid(axis="x", color="#E4E7EC")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3)
    for yi, total in zip(y, left):
        ax.text(total + max(left) * 0.012, yi, f"{int(total):,}", va="center", color=GREY)

    totals = sum(arrays.values())
    ratio_left = np.zeros(len(cfg.DOMAINS))
    for split in SPLITS:
        ratios = arrays[split] / totals * 100
        ax_ratio.barh(y, ratios, left=ratio_left, color=SPLIT_COLORS[split], height=0.66)
        for yi, start, value in zip(y, ratio_left, ratios):
            if value >= 8:
                ax_ratio.text(start + value / 2, yi, f"{value:.0f}%", ha="center", va="center", fontsize=9, color="white", weight="bold")
        ratio_left += ratios
    ax_ratio.set_yticks(y, [domain.title() for domain in cfg.DOMAINS])
    ax_ratio.invert_yaxis()
    ax_ratio.set_xlim(0, 100)
    ax_ratio.set_xlabel("Share of domain records (%)")
    ax_ratio.set_title("Split composition", loc="left")
    ax_ratio.grid(axis="x", color="#E4E7EC")
    ax_ratio.set_axisbelow(True)
    save_figure(fig, output_dir, "slide_02_domain_and_split_coverage.png", dpi)


def plot_text_lengths(records, summary, output_dir, dpi):
    lengths_by_domain = {
        domain: [count_words(r.example.sentence) for r in records if r.domain == domain]
        for domain in cfg.DOMAINS
    }
    all_lengths = np.array([count_words(record.example.sentence) for record in records])
    median = summary["word_length"]["overall"]["median"]
    p95 = summary["word_length"]["overall"]["p95"]

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.12, wspace=0.25)
    add_title(fig, "Most texts are short, with a manageable long tail", "Word counts are a tokenizer-independent pre-check; BERT truncation is audited separately")

    upper = max(20, int(np.percentile(all_lengths, 99)))
    bins = np.arange(0, upper + 3, 2)
    ax_hist.hist(all_lengths[all_lengths <= upper], bins=bins, color=BLUE, alpha=0.9, edgecolor="white")
    ax_hist.axvline(median, color=NAVY, linewidth=2, label=f"Median: {median:g} words")
    ax_hist.axvline(p95, color=CORAL, linewidth=2, linestyle="--", label=f"95th percentile: {p95:g}")
    ax_hist.set_title("Overall distribution", loc="left")
    ax_hist.set_xlabel("Words per text record")
    ax_hist.set_ylabel("Records")
    ax_hist.set_xlim(0, upper)
    ax_hist.grid(axis="y", color="#E4E7EC")
    ax_hist.set_axisbelow(True)
    ax_hist.legend(frameon=False)
    ax_hist.text(0.98, 0.96, "Axis capped at the 99th percentile", transform=ax_hist.transAxes, ha="right", va="top", fontsize=9, color=GREY)

    box = ax_box.boxplot(
        [lengths_by_domain[d] for d in cfg.DOMAINS],
        orientation="horizontal",
        tick_labels=[d.title() for d in cfg.DOMAINS],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NAVY, "linewidth": 2},
        boxprops={"facecolor": "#CFE3EF", "edgecolor": BLUE},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
    )
    del box
    ax_box.invert_yaxis()
    ax_box.set_title("Variation by domain", loc="left")
    ax_box.set_xlabel("Words per text record")
    ax_box.grid(axis="x", color="#E4E7EC")
    ax_box.set_axisbelow(True)
    save_figure(fig, output_dir, "slide_03_text_length_profile.png", dpi)


def plot_labels(summary, output_dir, dpi):
    sentiment_data = summary["sentiment_counts"]["by_domain"]
    category_data = summary["category_counts"]
    sentiment_order = [label for label in cfg.SENTIMENTS if any(label in sentiment_data[d] for d in cfg.DOMAINS)]
    top_categories = [label for label, _ in list(category_data["overall"].items())[:10]]

    fig, (ax_sent, ax_cat) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.15, wspace=0.32)
    add_title(fig, "Labels are imbalanced and strongly domain-specific", "This motivates per-label reporting and the planned class-weighted loss comparison")

    y = np.arange(len(cfg.DOMAINS))
    left = np.zeros(len(cfg.DOMAINS))
    for sentiment in sentiment_order:
        counts = np.array([sentiment_data[d].get(sentiment, 0) for d in cfg.DOMAINS], dtype=float)
        totals = np.array([sum(sentiment_data[d].values()) for d in cfg.DOMAINS], dtype=float)
        shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
        ax_sent.barh(y, shares, left=left, color=SENTIMENT_COLORS.get(sentiment, GREY), height=0.65, label=sentiment.title())
        for yi, start, value in zip(y, left, shares):
            if value >= 8:
                ax_sent.text(start + value / 2, yi, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
        left += shares
    ax_sent.set_yticks(y, [d.title() for d in cfg.DOMAINS])
    ax_sent.invert_yaxis()
    ax_sent.set_xlim(0, 100)
    ax_sent.set_xlabel("Share of raw triplets (%)")
    ax_sent.set_title("Sentiment distribution", loc="left")
    ax_sent.grid(axis="x", color="#E4E7EC")
    ax_sent.set_axisbelow(True)
    ax_sent.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.18))

    matrix = np.array(
        [[category_data["by_domain"][domain].get(category, 0) for category in top_categories] for domain in cfg.DOMAINS],
        dtype=float,
    )
    row_totals = np.array(
        [[sum(category_data["by_domain"][domain].values())] for domain in cfg.DOMAINS],
        dtype=float,
    )
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0) * 100
    image = ax_cat.imshow(normalized, aspect="auto", cmap="Blues", vmin=0)
    wrapped = ["\n".join(textwrap.wrap(label.replace("_", " "), 13)) for label in top_categories]
    ax_cat.set_xticks(np.arange(len(top_categories)), wrapped, rotation=42, ha="right", fontsize=8)
    ax_cat.set_yticks(np.arange(len(cfg.DOMAINS)), [d.title() for d in cfg.DOMAINS])
    ax_cat.set_title("Top 10 categories: share of each domain", loc="left")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            if normalized[row, col] >= 4:
                ax_cat.text(col, row, f"{normalized[row, col]:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if normalized[row, col] > normalized.max() * 0.48 else NAVY)
    colorbar = fig.colorbar(image, ax=ax_cat, fraction=0.035, pad=0.025)
    colorbar.set_label("Share of all domain triplets (%)", color=NAVY)
    save_figure(fig, output_dir, "slide_04_label_distribution.png", dpi)


def write_outputs(summary, output_dir):
    json_path = output_dir / "text_analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {json_path}")

    rows = records_to_summary_rows(summary)
    csv_path = output_dir / "domain_split_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    overall = summary["overall"]
    scope = summary["baseline_scope"]
    notes = f"""# Presentation notes: pre-model text analysis

## Headline

All {summary['validation']['loaded_source_files']} configured English source files load successfully: {overall['examples']:,} text records across {overall['domains']} domains, with {summary['validation']['parse_failures']} parse failures.

## Defensible observations

- The corpus contains {overall['total_triplets']:,} raw aspect-category-sentiment annotations across {overall['unique_categories']} category labels and {overall['unique_sentiments']} normalized sentiment labels.
- {overall['model_ready_triplets']:,} triplets ({overall['model_ready_percent']:.1f}%) fit the current explicit-aspect, aligned, single-label-pair baseline.
- {scope['implicit_triplets_excluded']:,} implicit/NULL triplets are outside the current BIO span architecture and must be disclosed as a scope limitation.
- Overall text length is a median of {summary['word_length']['overall']['median']:g} words; 95% of records contain at most {summary['word_length']['overall']['p95']:g} words by the regex count used here.
- The label charts show why aggregate accuracy is insufficient: report per-label precision/recall and compare standard versus class-weighted loss as planned.

## Method note

These are pre-model descriptive statistics from raw local M-ABSA files. Sentiment/category plots include all raw annotations, including NULL aspects. “Model-ready” uses the exact same baseline selection logic as training. Word counts are not BERT WordPiece counts; token-level truncation/alignment remains covered by `data_audit.py`.
"""
    notes_path = output_dir / "PRESENTATION_NOTES.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"Saved {notes_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(cfg.DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data_dir.resolve())
    summary = summarize_corpus(records, expected_files=len(cfg.DOMAINS) * len(SPLITS))
    set_style()
    plot_dashboard(summary, output_dir, args.dpi)
    plot_coverage(summary, output_dir, args.dpi)
    plot_text_lengths(records, summary, output_dir, args.dpi)
    plot_labels(summary, output_dir, args.dpi)
    write_outputs(summary, output_dir)

    print(
        f"\nComplete: {summary['overall']['examples']:,} records and "
        f"{summary['overall']['total_triplets']:,} triplets analysed."
    )


if __name__ == "__main__":
    main()
