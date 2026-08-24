"""
Train YOLO26 on the fish-detection dataset produced by preprocess.py.

First run:
    python train_yolo.py --data /path/to/data/spectrograms/dataset.yaml --name fish_yolo26s_v2

Resuming after a time-limited run stopped without finishing (same --project/--name,
same dataset -- this is for continuing ONE training run across multiple SLURM
submissions, NOT for restarting training on a changed dataset/config):
    python train_yolo.py --resume --project runs --name fish_yolo26s_v2

Spectrogram-specific augmentation notes (defaults below, override to ablate):
    --mosaic 0         mosaicking glues 4 unrelated recordings into one spectrogram,
                        inventing time/frequency discontinuities that never occur in
                        a real deployment
    --auto-augment ""  RandAugment chains photographic color ops (posterize, solarize,
                        equalize) that don't correspond to anything on a dB-intensity map
    --hsv-h / --hsv-s   hue/saturation jitter; leave at 0 unless spectrograms are
                        rendered with a color colormap (check image_writer.py) -- on
                        true grayscale-as-RGB images these are inert no-ops anyway
    --scale             frequency axis is a physical measurement, not photographic
                        scale-invariant content; kept smaller than the YOLO default (0.5)
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spectrograms/dataset.yaml")
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=300,
                    help="upper bound only -- --time below is what actually ends the run")
    p.add_argument("--time", type=float, default=23.5,
                    help="hours; set with real margin below your SLURM --time wall-clock "
                         "limit so the run exits cleanly instead of being SIGKILLed "
                         "mid-epoch -- check the actual partition/job limit, don't assume 24h")
    p.add_argument("--batch", default=-1,
                    help="-1 = Ultralytics AutoBatch, targets ~60%% GPU memory")
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20,
                    help="the previous run (patience=100) showed val loss bottoming out "
                         "around epoch ~30-35 while train loss kept falling to epoch 85, "
                         "and mAP50-95 peaking ~epoch 55-60 then dropping ~20%% by epoch 85 "
                         "-- lowered so early stopping can actually fire instead of running "
                         "the full time budget into overfitting")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="fish_yolo26s")
    p.add_argument("--resume", action="store_true")

    # Augmentation -- defaults reflect the spectrogram-specific review; override
    # per-run to ablate any of these individually.
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--auto-augment", default="",
                    help="empty string disables it (passed to Ultralytics as None); "
                         "'randaugment' was the prior default")
    p.add_argument("--hsv-h", type=float, default=0.0)
    p.add_argument("--hsv-s", type=float, default=0.0)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--scale", type=float, default=0.2,
                    help="trimmed from the YOLO default of 0.5")
    p.add_argument("--fliplr", type=float, default=0.5,
                    help="time-reversal; candidate for a future ablation at 0.0 if call "
                         "envelopes turn out to be meaningfully time-asymmetric")
    args = p.parse_args()

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if args.resume:
        ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt} -- nothing to resume")
        print(f"Resuming from {ckpt}")
        model = YOLO(str(ckpt))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            time=args.time,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            exist_ok=True,
            mosaic=args.mosaic,
            auto_augment=args.auto_augment or None,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            scale=args.scale,
            fliplr=args.fliplr,
        )

    print("Training finished/stopped. Best weights (use this, not last.pt, for eval/deployment):",
          Path(args.project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
