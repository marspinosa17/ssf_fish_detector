"""
Generate diagnostic plots from analyze_misses.py output.

Works with the plain v1 CSV (source/duration/freq breakdowns) and
automatically adds a per-original-label breakdown if the CSV has been run
through join_original_labels.py (i.e. has an 'orig_label' column).

Usage:
    python plot_miss_analysis.py --csv miss_analysis_val_119431.csv --out-dir plots/
    python plot_miss_analysis.py --csv miss_analysis_labeled.csv --out-dir plots/
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

STATUS_COLORS = {"FN": "#d62728", "TP_low_conf": "#ff7f0e", "TP_confident": "#2ca02c", "FP": "#9467bd"}
STATUS_ORDER = ["FN", "TP_low_conf", "TP_confident"]  # GT-box statuses only, for the stacked bars


def stacked_bar(df, groupby_col, out_path, title, min_count=0):
    rates = (df.groupby(groupby_col, observed=True)["status"]
             .value_counts(normalize=True).unstack().reindex(columns=STATUS_ORDER).fillna(0))
    counts = df[groupby_col].value_counts()
    rates = rates.loc[counts[counts >= min_count].index]
    rates = rates.reindex(sorted(rates.index, key=str))

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rates)), 5))
    bottom = None
    for status in STATUS_ORDER:
        vals = rates[status].values
        ax.bar(rates.index.astype(str), vals, bottom=bottom, label=status,
               color=STATUS_COLORS[status])
        bottom = vals if bottom is None else bottom + vals

    for i, (idx, n) in enumerate(counts.reindex(rates.index).items()):
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel("Fraction of GT boxes")
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # v2 CSVs mix box_type='gt' (status FN/TP_*) and box_type='pred' (status FP)
    # rows in one file. Stats below describe "what happened to each GT box",
    # so FP rows -- which aren't about any GT box -- need to be split out
    # first or they'd silently inflate group denominators.
    if "box_type" in df.columns:
        fp_df = df[df["box_type"] == "pred"].copy()
        df = df[df["box_type"] == "gt"].copy()
    else:
        fp_df = pd.DataFrame(columns=df.columns)  # v1 CSV has no FP rows at all

    # overall
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df["status"].value_counts().reindex(STATUS_ORDER).fillna(0)
    ax.pie(counts, labels=[f"{s}\n{c} ({100*c/len(df):.1f}%)" for s, c in counts.items()],
           colors=[STATUS_COLORS[s] for s in STATUS_ORDER])
    ax.set_title(f"Overall status (n={len(df)} GT boxes)")
    fig.savefig(out_dir / "overall_status.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'overall_status.png'}")

    # by source
    stacked_bar(df, "source", out_dir / "status_by_source.png", "Status by source")

    # by duration bucket
    df["dur_bucket"] = pd.cut(df["approx_duration_s"], bins=[0, 0.2, 0.5, 1.0, 2.0, 3.0],
                               labels=["<0.2s", "0.2-0.5s", "0.5-1.0s", "1.0-2.0s", "2.0-3.0s"])
    stacked_bar(df, "dur_bucket", out_dir / "status_by_duration.png", "Status by call duration")

    # by freq bucket
    df["freq_bucket"] = pd.cut(df["approx_freq_lo_hz"], bins=[0, 100, 250, 500, 1000, 2000],
                                labels=["0-100Hz", "100-250Hz", "250-500Hz", "500-1000Hz", "1000-2000Hz"])
    stacked_bar(df, "freq_bucket", out_dir / "status_by_frequency.png", "Status by frequency band")

    # by original label, if present (min_count filters out ultra-rare labels like HS/KW/fish)
    if "orig_label" in df.columns:
        stacked_bar(df, "orig_label", out_dir / "status_by_original_label.png",
                    "Status by original source label", min_count=20)

    # TP_low_conf confidence histogram -- how close to threshold are the "almost caught" boxes?
    low_conf = df[df["status"] == "TP_low_conf"]["match_conf"]
    if len(low_conf):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(low_conf, bins=30, color="#ff7f0e")
        ax.set_xlabel("Model confidence (below operating threshold)")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution of TP_low_conf boxes\n"
                      "(model found these, just not confidently)")
        plt.tight_layout()
        fig.savefig(out_dir / "tp_low_conf_confidence_hist.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / 'tp_low_conf_confidence_hist.png'}")

    # FN IoU histogram -- true absence (iou=0) vs near-miss localization (iou 0.3-0.5)
    fn_iou = df[df["status"] == "FN"]["match_iou"]
    if len(fn_iou):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fn_iou, bins=30, color="#d62728")
        ax.set_xlabel("Best candidate IoU (0 = nothing predicted nearby)")
        ax.set_ylabel("Count")
        ax.set_title("FN breakdown: true absence vs near-miss localization")
        plt.tight_layout()
        fig.savefig(out_dir / "fn_iou_hist.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / 'fn_iou_hist.png'}")

    # False positives (v2 only) -- counts by source and confidence distribution
    if len(fp_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        fp_df["source"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#9467bd")
        ax.set_ylabel("False positive count")
        ax.set_title(f"False positives by source (n={len(fp_df)} total)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(out_dir / "fp_by_source.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / 'fp_by_source.png'}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fp_df["match_conf"], bins=30, color="#9467bd")
        ax.set_xlabel("Model confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution of false positives\n"
                      "(high-confidence FPs here are the costliest false alarms)")
        plt.tight_layout()
        fig.savefig(out_dir / "fp_confidence_hist.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / 'fp_confidence_hist.png'}")


if __name__ == "__main__":
    main()
