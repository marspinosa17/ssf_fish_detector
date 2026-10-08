"""
Join miss_analysis_v2.csv (box_start_s/box_end_s/audio_path) back to the
original data/manifest.csv to recover the source-specific label
(pulse/tonal/chorus/knock/grunt/boat/other for Seth; FS/NN/UN/HS/KW/fish/other
for Xavier; m/w/lt for Tagus) for each box, via time-overlap on the same
audio_path.

Usage:
    python join_original_labels.py \
        --miss-csv miss_analysis_v2.csv \
        --manifest data/manifest.csv \
        --out miss_analysis_labeled.csv
"""
import argparse
import pandas as pd


def best_overlap_label(row, manifest_by_audio):
    ap = row["audio_path"]
    if ap not in manifest_by_audio.groups:
        return None
    cand = manifest_by_audio.get_group(ap)
    # overlap = intersection length between [box_start_s, box_end_s] and each
    # candidate annotation's [begin_s, end_s]
    overlap = (cand["end_s"].clip(upper=row["box_end_s"])
               - cand["begin_s"].clip(lower=row["box_start_s"])).clip(lower=0)
    if overlap.max() <= 0:
        return None
    return cand.loc[overlap.idxmax(), "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="miss_analysis_labeled.csv")
    args = ap.parse_args()

    miss = pd.read_csv(args.miss_csv)
    manifest = pd.read_csv(args.manifest)

    manifest_by_audio = manifest.groupby("audio_path")

    print(f"Joining labels for {len(miss)} rows...")
    miss["orig_label"] = miss.apply(lambda r: best_overlap_label(r, manifest_by_audio), axis=1)

    n_unmatched = miss["orig_label"].isna().sum()
    print(f"{n_unmatched} / {len(miss)} rows had no overlapping original annotation "
          f"({100 * n_unmatched / len(miss):.1f}%) -- likely padding/window-edge effects")

    miss.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    print("\n=== Status rate by original label ===")
    print((miss.groupby("orig_label")["status"]
           .value_counts(normalize=True).unstack().fillna(0) * 100).round(1))
    print("\n=== Counts by original label ===")
    print(miss["orig_label"].value_counts())


if __name__ == "__main__":
    main()
