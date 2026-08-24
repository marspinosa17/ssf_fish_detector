"""
Remap manifest.csv's audio_path column from the local Windows layout to an
HPC (Linux) layout with the same structure below the data root.

Also useful any time the data directory moves — not HPC-specific.

Run:
    python remap_manifest_paths.py \
        --manifest C:\\Users\\Marcello\\whoissf\\fd_framework\\data\\manifest.csv \
        --old-root "C:\\Users\\Marcello\\whoissf\\fd_framework\\data" \
        --new-root "/scratch/marcello/fd_framework/data" \
        --output manifest_hpc.csv

This is a pure path rewrite -- it does NOT rebuild the manifest or touch
split assignments. That's the point: rerunning build_manifest.py fresh on
a different filesystem risks a different (seeded-shuffle-dependent) split
than the one already validated locally. This preserves it exactly.

--check-exists optionally verifies the new paths actually resolve on disk
(only useful when run ON the HPC side, after the data has been copied over
-- Windows is case-insensitive, most HPC filesystems are not, so this is
the check that catches a casing mismatch the rewrite itself can't).
"""
from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath, PurePosixPath

import pandas as pd


def remap(path: str, old_root: str, new_root: str) -> str:
    rel = PureWindowsPath(path).relative_to(PureWindowsPath(old_root))
    return str(PurePosixPath(new_root, *rel.parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--old-root", required=True,
                    help=r'e.g. "C:\Users\Marcello\whoissf\fd_framework\data"')
    p.add_argument("--new-root", required=True,
                    help="e.g. /scratch/marcello/fd_framework/data")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--check-exists", action="store_true",
                    help="verify remapped paths exist on disk (run on the HPC side)")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"{len(df)} rows, {df['audio_path'].nunique()} unique audio files")

    bad_prefix = []
    def safe_remap(path: str) -> str:
        try:
            return remap(path, args.old_root, args.new_root)
        except ValueError:
            bad_prefix.append(path)
            return path

    df["audio_path"] = df["audio_path"].apply(safe_remap)

    if bad_prefix:
        print(f"WARNING: {len(bad_prefix)} path(s) did not share --old-root and were left "
              f"unchanged -- inspect these before trusting the output:")
        for bp in bad_prefix[:10]:
            print(" ", bp)
        if len(bad_prefix) > 10:
            print(f"  ... and {len(bad_prefix) - 10} more")

    if args.check_exists:
        unique_paths = df["audio_path"].unique()
        missing = [pth for pth in unique_paths if not Path(pth).exists()]
        print(f"Existence check: {len(unique_paths) - len(missing)}/{len(unique_paths)} resolve on disk")
        if missing:
            print(f"MISSING (first 20 of {len(missing)}) -- likely a case-sensitivity or "
                  f"incomplete-copy issue, since Windows doesn't distinguish case and Linux does:")
            for m in missing[:20]:
                print(" ", m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()