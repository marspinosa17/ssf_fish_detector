"""
Train RT-DETR-large (rtdetr-l.pt) on the fish-detection dataset produced by preprocess.py.

RT-DETR is a transformer-based detector, not part of the YOLO family, but
Ultralytics exposes it through the same Model/.train() interface, so the rest
of this script (data loading, augmentation args, resume/warm-start logic)
works unchanged. Only two scales ship: rtdetr-l.pt (default here) and
rtdetr-x.pt -- no n/s/m the way YOLO has.

First run:
    python train_yolo.py --data /path/to/data/spectrograms/dataset.yaml --name fish_rtdetr_l_v2

Resuming after a time-limited run stopped WITHOUT finishing -- i.e. it was still
mid-epoch when the wall clock/job killed it, same --project/--name, same dataset:
    python train_yolo.py --resume --project runs --name fish_rtdetr_l_v2

--resume only works on an interrupted (incomplete) run -- Ultralytics strips the
optimizer/epoch state once a run actually finishes (epoch cap OR early stopping),
so there's nothing left to resume from at that point. If your run already finished
and you want to train further from its best weights, that's a WARM START, not a
resume -- a brand-new run (fresh optimizer/LR schedule) initialized from those
weights, with your real config passed explicitly (defaults below won't be inherited
the way a true resume inherits them):
    python train_yolo.py --model runs/fish_rtdetr_l_v2/weights/best.pt \
        --name fish_rtdetr_l_v2_cont --data /path/to/dataset.yaml --patience 40 \
        --mosaic 0.0 --auto-augment "" --hsv-h 0.0 --hsv-s 0.0 --scale 0.2

RT-DETR-specific notes:
    --batch     don't use -1 (AutoBatch): thop's profiler errors on RT-DETR's
                transformer blocks and silently falls back to a hardcoded
                batch=16 with just a warning, not the memory-aware value you
                asked for. Set a real number (default here: 16).
    AMP         on by default; can occasionally produce NaN losses during the
                bipartite/Hungarian matching step -- watch early epochs.
    determinism grid_sample (used internally) doesn't support deterministic=True.

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
from ultralytics import RTDETR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spectrograms/dataset.yaml")
    p.add_argument("--model", default="rtdetr-l.pt",
                    help="RT-DETR-large checkpoint or a rtdetr-l.yaml config for "
                         "training from scratch. rtdetr-x.pt is the only other "
                         "scale Ultralytics ships (no n/s/m like YOLO).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=300,
                    help="upper bound only -- --time below is what actually ends the run")
    p.add_argument("--time", type=float, default=23.5,
                    help="hours; set with real margin below your SLURM --time wall-clock "
                         "limit so the run exits cleanly instead of being SIGKILLed "
                         "mid-epoch -- check the actual partition/job limit, don't assume 24h")
    p.add_argument("--batch", type=int, default=16,
                    help="fixed batch size. Don't use -1 (AutoBatch) with RT-DETR: "
                         "thop's profiler errors on the transformer blocks, and "
                         "AutoBatch silently falls back to a hardcoded batch=16 with "
                         "only a warning printed rather than the memory-aware value "
                         "you asked for -- so a real number here is safer than -1 "
                         "even though it means less GPU memory gets used up-front")
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20,
                    help="the previous run (patience=100) showed val loss bottoming out "
                         "around epoch ~30-35 while train loss kept falling to epoch 85, "
                         "and mAP50-95 peaking ~epoch 55-60 then dropping ~20%% by epoch 85 "
                         "-- lowered so early stopping can actually fire instead of running "
                         "the full time budget into overfitting")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="fish_rtdetr_l")
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

        # Ultralytics strips optimizer/epoch state from checkpoints once a run
        # finishes (epoch cap OR early stopping) -- at that point there's nothing
        # left to resume from. Passing resume=True on a finished checkpoint doesn't
        # error: Ultralytics prints a warning and silently starts a brand-new run
        # with its OWN defaults (coco8.yaml, patience=100, mosaic=1.0, default
        # runs/detect output dir) since none of our real args get forwarded on this
        # path. Check for that up front and fail loudly instead.
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        if raw.get("epoch") is None or raw.get("epoch") < 0 or raw.get("optimizer") is None:
            best = ckpt.parent / "best.pt"
            raise RuntimeError(
                f"{ckpt} has no optimizer/epoch state -- this run already finished "
                f"(hit its epoch cap or early stopping), so there's nothing to resume. "
                f"--resume is only for continuing training an interrupted (mid-run) job "
                f"with the same wall-clock chunk. To train further from these weights, "
                f"warm-start a NEW run instead (fresh optimizer/LR schedule, your real "
                f"data/config explicitly passed):\n"
                f"  python train_yolo.py --model {best} --name {args.name}_cont "
                f"--data <your dataset.yaml> --patience 40 --mosaic 0.0 "
                f"--auto-augment \"\" --hsv-h 0.0 --hsv-s 0.0 --scale 0.2"
            )
        print(f"Resuming from {ckpt}")
        model = RTDETR(str(ckpt))
        results = model.train(resume=True)
    else:
        print(
            "RT-DETR: grid_sample doesn't support deterministic=True, and AMP (on "
            "by default) can occasionally produce NaN losses during bipartite/"
            "Hungarian matching -- watch early epochs for NaN loss values."
        )
        model = RTDETR(args.model)
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
