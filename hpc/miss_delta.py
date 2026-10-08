#!/usr/bin/env python
"""
Per-tile discrepancy analysis between two (or more) models' miss-analysis
CSVs, plus per-source and per-covariate (duration/bandwidth) breakdowns.

Expects CSVs shaped like your existing miss-analysis output: one row per
GT box (box_type='gt', status in {TP, FN}) and one row per predicted box
(box_type='pred', status in {TP, FP}), keyed by `stem` (the tile id).
Column names assumed: stem, audio_path, source, box_type, status, plus
optionally approx_duration_s, approx_freq_lo_hz, approx_freq_hi_hz.

This is pure pandas/numpy -- no GPU needed. It'll finish in well under a
minute even on the full val+test sets, so the slurm job is provided
mainly for convenience/reproducibility; running it directly on a login
node is also fine.

Usage:
    python miss_delta_analysis.py \
        --csv yolo   val  runs/fish_yolo26s/analysis/miss_val.csv \
        --csv yolo   test runs/fish_yolo26s/analysis/miss_test.csv \
        --csv rtdetr val  runs/ul_rtdetr_l_v1/analysis/miss_val.csv \
        --csv rtdetr test runs/ul_rtdetr_l_v1/analysis/miss_test.csv \
        --out-dir analysis/miss_delta

Each --csv flag takes three tokens: MODEL, SPLIT, PATH.

Outputs in --out-dir (per model pair that shares a split):
    tile_scores_<model>_<split>.csv           one row per tile: tp/fp/fn/score
    delta_<a>_vs_<b>_<split>.csv               full per-tile join, sorted by |delta|
    top_discrepancies_<a>_vs_<b>_<split>.csv   shortlist, both tails (for the
                                                qualitative poster panel)
    top_discrepancies_<a>_vs_<b>_<split>_detail.csv   full box-level rows for
                                                        those same tiles
    per_source_metrics.csv                     pooled P/R/F1 by source, per model/split
    recall_by_duration.csv                     recall vs. call duration, per model/split
    recall_by_bandwidth.csv                    recall vs. call bandwidth, per model/split
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ───────────────────────────── CLI ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-tile discrepancy analysis across models")
    p.add_argument("--csv", nargs=3, action="append", required=True,
                   metavar=("MODEL", "SPLIT", "PATH"))
    p.add_argument("--out-dir", type=Path, default=Path("analysis/miss_delta"))
    p.add_argument("--top-n", type=int, default=40,
                   help="rows to keep in EACH tail of the top-discrepancy shortlist")
    p.add_argument("--n-bins", type=int, default=5,
                   help="quantile bins for the duration/bandwidth recall breakdown")
    return p.parse_args()


# ───────────────────────────── loading ───────────────────────────────────

def read_miss_csv(path: Path) -> pd.DataFrame:
    """Tries tab-separated first (matches the sample format), falls back to
    letting pandas sniff the delimiter."""
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


REQUIRED_COLS = {"stem", "audio_path", "source", "box_type", "status"}

def load_and_check(path: Path) -> pd.DataFrame:
    df = read_miss_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}. "
                          f"Found columns: {list(df.columns)}")
    return df


# ───────────────────────────── per-tile scoring ──────────────────────────

def tile_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a miss-analysis CSV to one row per tile.

    TP/FN are counted once each from the `gt` rows (every GT box gets
    exactly one such row); FP is counted once from the `pred` rows (every
    predicted box gets exactly one such row). This avoids double-counting
    the TP rows that appear on both the gt and pred side of a match.
    """
    gt = df[df["box_type"] == "gt"]
    pred = df[df["box_type"] == "pred"]

    tp = gt[gt["status"] == "TP"].groupby("stem").size().rename("tp")
    fn = gt[gt["status"] == "FN"].groupby("stem").size().rename("fn")
    fp = pred[pred["status"] == "FP"].groupby("stem").size().rename("fp")

    meta = df.drop_duplicates("stem")[["stem", "audio_path", "source"]].set_index("stem")

    out = pd.concat([tp, fn, fp], axis=1).reindex(meta.index).fillna(0).astype(int)
    out = meta.join(out)
    out["n_gt"] = out["tp"] + out["fn"]
    out["n_pred"] = out["tp"] + out["fp"]
    out["score"] = out["tp"] - out["fp"] - out["fn"]  # net correctness for the tile
    return out.reset_index()


# ───────────────────────────── pairwise delta ────────────────────────────

def pairwise_delta(a_name: str, a_scores: pd.DataFrame,
                    b_name: str, b_scores: pd.DataFrame) -> pd.DataFrame:
    merged = a_scores.merge(b_scores, on=["stem", "audio_path", "source"],
                            suffixes=(f"_{a_name}", f"_{b_name}"), how="outer")
    for col in ("tp", "fp", "fn", "n_gt", "n_pred", "score"):
        merged[f"{col}_{a_name}"] = merged[f"{col}_{a_name}"].fillna(0)
        merged[f"{col}_{b_name}"] = merged[f"{col}_{b_name}"].fillna(0)
    merged["delta_score"] = merged[f"score_{a_name}"] - merged[f"score_{b_name}"]
    merged["abs_delta"] = merged["delta_score"].abs()
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def save_top_detail(a_name: str, b_name: str, split: str, top: pd.DataFrame,
                    raw: dict, out_dir: Path) -> None:
    """Full box-level rows (coords, conf, iou) for the shortlisted tiles --
    what you actually want open when hand-picking the qualitative panel."""
    stems = set(top["stem"])
    a_df, b_df = raw[(a_name, split)], raw[(b_name, split)]
    detail = pd.concat([
        a_df[a_df["stem"].isin(stems)].assign(model=a_name),
        b_df[b_df["stem"].isin(stems)].assign(model=b_name),
    ], ignore_index=True)
    detail.to_csv(out_dir / f"top_discrepancies_{a_name}_vs_{b_name}_{split}_detail.csv", index=False)


# ───────────────────────────── bonus: pooled + stratified metrics ────────

def pooled_prf(df: pd.DataFrame) -> dict:
    tp = df[(df.box_type == "gt") & (df.status == "TP")].shape[0]
    fn = df[(df.box_type == "gt") & (df.status == "FN")].shape[0]
    fp = df[(df.box_type == "pred") & (df.status == "FP")].shape[0]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision + recall) else float("nan"))
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def per_source_metrics(model: str, split: str, df: pd.DataFrame) -> list[dict]:
    rows = [dict(model=model, split=split, source=source, **pooled_prf(grp))
            for source, grp in df.groupby("source")]
    rows.append(dict(model=model, split=split, source="ALL", **pooled_prf(df)))
    return rows


def recall_by_covariate(model: str, split: str, df: pd.DataFrame, col: str,
                        n_bins: int) -> pd.DataFrame:
    """Recall (TP rate) among GT boxes, bucketed by a numeric covariate
    (e.g. approx_duration_s, or bandwidth = hi - lo)."""
    gt = df[df["box_type"] == "gt"].dropna(subset=[col]).copy()
    if gt.empty:
        return pd.DataFrame()
    try:
        gt["bin"] = pd.qcut(gt[col], q=min(n_bins, gt[col].nunique()), duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    gt["is_tp"] = (gt["status"] == "TP").astype(int)
    out = gt.groupby("bin", observed=True).agg(
        n_gt=("is_tp", "size"),
        recall=("is_tp", "mean"),
        **{f"{col}_min": (col, "min"), f"{col}_max": (col, "max")},
    ).reset_index()
    out.insert(0, "split", split)
    out.insert(0, "model", model)
    return out


# ───────────────────────────── main ──────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[tuple[str, str], pd.DataFrame] = {}
    scores: dict[tuple[str, str], pd.DataFrame] = {}
    source_rows, dur_rows, bw_rows = [], [], []

    for model, split, path in args.csv:
        print(f"loading {model}/{split}: {path}")
        df = load_and_check(Path(path))
        raw[(model, split)] = df

        sc = tile_scores(df)
        scores[(model, split)] = sc
        sc.to_csv(args.out_dir / f"tile_scores_{model}_{split}.csv", index=False)

        source_rows += per_source_metrics(model, split, df)

        if "approx_duration_s" in df.columns:
            dur_rows.append(recall_by_covariate(model, split, df, "approx_duration_s", args.n_bins))
        if {"approx_freq_lo_hz", "approx_freq_hi_hz"}.issubset(df.columns):
            df2 = df.copy()
            df2["bandwidth_hz"] = df2["approx_freq_hi_hz"] - df2["approx_freq_lo_hz"]
            bw_rows.append(recall_by_covariate(model, split, df2, "bandwidth_hz", args.n_bins))

    # ── pairwise per-tile deltas, for every model pair sharing a split ──
    models_by_split: dict[str, list[str]] = {}
    for (model, split) in scores:
        models_by_split.setdefault(split, []).append(model)

    for split, models in models_by_split.items():
        for a_name, b_name in combinations(sorted(set(models)), 2):
            delta = pairwise_delta(a_name, scores[(a_name, split)],
                                   b_name, scores[(b_name, split)])
            delta.to_csv(args.out_dir / f"delta_{a_name}_vs_{b_name}_{split}.csv", index=False)

            a_better = (delta[delta["delta_score"] > 0]
                        .sort_values("delta_score", ascending=False)
                        .head(args.top_n).assign(favors=a_name))
            b_better = (delta[delta["delta_score"] < 0]
                        .sort_values("delta_score", ascending=True)
                        .head(args.top_n).assign(favors=b_name))
            top = pd.concat([a_better, b_better], ignore_index=True)
            top.to_csv(args.out_dir / f"top_discrepancies_{a_name}_vs_{b_name}_{split}.csv", index=False)
            save_top_detail(a_name, b_name, split, top, raw, args.out_dir)

            n_a_better = (delta["delta_score"] > 0).sum()
            n_b_better = (delta["delta_score"] < 0).sum()
            n_tied = (delta["delta_score"] == 0).sum()
            print(f"\n[{split}] {a_name} vs {b_name}: "
                  f"{a_name} better on {n_a_better} tiles, {b_name} better on {n_b_better}, "
                  f"tied {n_tied} (of {len(delta)})")
            print("  mean delta by source:")
            print(delta.groupby("source")["delta_score"].mean().to_string())

    # ── bonus outputs ──
    pd.DataFrame(source_rows).to_csv(args.out_dir / "per_source_metrics.csv", index=False)
    if dur_rows:
        pd.concat(dur_rows, ignore_index=True).to_csv(args.out_dir / "recall_by_duration.csv", index=False)
    if bw_rows:
        pd.concat(bw_rows, ignore_index=True).to_csv(args.out_dir / "recall_by_bandwidth.csv", index=False)

    print(f"\nWrote all tables to {args.out_dir}/")


if __name__ == "__main__":
    main()
