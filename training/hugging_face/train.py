"""RT-DETRv2 fine-tuning via HuggingFace transformers.

Independent of the Ultralytics-RT-DETR training path already in use on this
project — uses HF's Trainer directly against a pretrained RTDetrV2 checkpoint,
so the comparison doesn't depend on Ultralytics' training internals or its
AGPL license.

Run:
    python -m training.rtdetr_hf.train --smoke                # tiny subset, fast sanity check
    python -m training.rtdetr_hf.train --epochs 50 --batch-size 16

Uses HF `Trainer` (not `Accelerate` directly) — Trainer already wraps
Accelerate internally for multi-GPU/mixed precision, and a single-node
H200 run has no need for the extra manual control Accelerate's lower-level
API would add.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    from . import config
    from .augmentations import AugmentationConfig
    from .dataset import build_dataset, collate_fn
    from .logging_utils import setup_logging
except ImportError:
    import config
    from augmentations import AugmentationConfig
    from dataset import build_dataset, collate_fn
    from logging_utils import setup_logging

logger = logging.getLogger("rtdetr_hf.train")

SMOKE_MAX_STEPS = 20  # --smoke: total optimizer steps, used for both max_steps and warmup_steps


def build_param_groups(model: torch.nn.Module, head_lr: float, backbone_lr_mult: float) -> list[dict]:
    """Separate backbone params (lower LR) from the rest (full LR).

    Standard DETR-family practice: the backbone starts from ImageNet/COCO
    pretraining and needs gentler updates than the randomly-initialized (or
    less-converged) detection head, or it forgets useful low-level features
    before the head catches up. backbone_lr_mult ~0.1 is the common default.
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    logger.info("Param groups: backbone=%d tensors @ lr=%.2e, head=%d tensors @ lr=%.2e",
                len(backbone_params), head_lr * backbone_lr_mult, len(head_params), head_lr)
    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_mult},
        {"params": head_params, "lr": head_lr},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT-DETRv2 fine-tuning (HuggingFace transformers)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset (a few images/source) for a fast pipeline sanity check")
    p.add_argument("--smoke-per-source", type=int, default=4,
                   help="images per source when --smoke is set")
    p.add_argument("--checkpoint", default=config.DEFAULT_CHECKPOINT,
                   help="pretrained checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="head learning rate")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="backbone LR = lr * this multiplier")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="path to a checkpoint dir to resume from, or 'auto' to resume "
                        "from the latest checkpoint under --output-dir")
    p.add_argument("--hflip-prob", type=float, default=0.0,
                   help="time-axis flip probability; UNVALIDATED for spectrograms, off by default")
    p.add_argument("--scale-jitter-prob", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true", default=False,
                   help="use fp16 instead of the default bf16 (for H200)")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=2000,)
    p.add_argument("--save-steps", type=int, default=2000,
                   help="must match --eval-steps")
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="stop after this many eval checks with no improvement")
    p.add_argument("--early-stopping-threshold", type=float, default=0.01,
                   help="minimum eval_loss improvement to reset the patience counter")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    logger_local = setup_logging(output_dir, "train.log", "rtdetr_hf.train")

    logger_local.info("Args: %s", vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger_local.info("Device: %s (%s)", device,
                      torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    logger_local.info("Loading pretrained model %s (ignore_mismatched_sizes=True: COCO's 80 "
                      "classes -> our 1 'fish' class)", args.checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    aug_cfg = AugmentationConfig(
        hflip_prob=args.hflip_prob,
        scale_jitter_prob=args.scale_jitter_prob,
    )
    if aug_cfg.hflip_prob > 0:
        logger_local.warning(
            "hflip_prob=%.2f: horizontal (time-axis) flip is UNVALIDATED for this "
            "spectrogram data — confirm it doesn't hurt val mAP before trusting it.",
            aug_cfg.hflip_prob,
        )

    train_dataset = build_dataset(
        "train", image_processor, augmentation_config=aug_cfg,
        smoke=args.smoke, smoke_per_source=args.smoke_per_source,
    )
    val_dataset = build_dataset(
        "val", image_processor, augmentation_config=None,
        smoke=args.smoke, smoke_per_source=args.smoke_per_source,
    )

    param_groups = build_param_groups(model, head_lr=args.lr, backbone_lr_mult=args.backbone_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # TrainingArguments deprecates warmup_ratio (removed in v5.2) in favor of an
    # explicit warmup_steps count -- compute the equivalent step count here so
    # --warmup-ratio remains the CLI knob while the internal call takes steps.
    if args.smoke:
        total_steps = SMOKE_MAX_STEPS
    else:
        total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))

    # TrainingArguments deprecates logging_dir (removed in v5.2) in favor of the
    # TENSORBOARD_LOGGING_DIR env var; set it before constructing TrainingArguments.
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "tb_logs"))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch" if args.smoke else "steps",
        eval_steps=None if args.smoke else args.eval_steps,
        save_strategy="epoch" if args.smoke else "steps",
        save_steps=None if args.smoke else args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        bf16=not args.fp16 and device == "cuda",
        dataloader_num_workers=0 if args.smoke else args.num_workers,
        remove_unused_columns=False,  # RT-DETR's variable-length "labels" dicts aren't columns
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        max_steps=SMOKE_MAX_STEPS if args.smoke else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )],
    )

    # Resolved here (before model construction), not just before trainer.train(), so a
    # resume loads its weights via from_pretrained() below -- Trainer's own resume-time
    # state_dict load does not re-tie RTDetrV2's tied class_embed/bbox_embed parameters
    # (no tie_weights() call in Trainer._load_from_checkpoint), which silently reinitializes
    # the detection heads on every resume. from_pretrained() ties them correctly; loading
    # the correct weights this way first makes Trainer's later (incomplete) reload of the
    # same checkpoint redundant but harmless -- load_state_dict(strict=False) only
    # overwrites keys it finds and leaves anything already-correct alone.
    resume = args.resume_from_checkpoint
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(str(output_dir))
        logger_local.info("Auto-resume: found checkpoint %s", resume)
    model_source = resume if resume else args.checkpoint

    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    logger_local.info("Loading model weights from %s (ignore_mismatched_sizes=True: COCO's "
                      "80 classes -> our 1 'fish' class)", model_source)
    model = AutoModelForObjectDetection.from_pretrained(
        model_source,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    logger_local.info("Starting training: epochs=%d batch_size=%d lr=%.2e (backbone %.2e) smoke=%s",
                      training_args.num_train_epochs, args.batch_size, args.lr,
                      args.lr * args.backbone_lr_mult, args.smoke)
    trainer.train(resume_from_checkpoint=resume)
    
    logger_local.info("Peak GPU memory: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)

    # trainer.train()'s own end-of-run reload (_load_best_model, when
    # load_best_model_at_end=True and the best step differs from the final
    # step) has the same tied-weights gap as the resume path: no tie_weights()
    # call after its raw load_state_dict(). Re-load the best checkpoint the
    # safe way (from_pretrained, which ties correctly) before saving to
    # final/ -- same fix already applied to the resume path above.
    if trainer.state.best_model_checkpoint:
        logger_local.info("Re-loading best checkpoint %s via from_pretrained (safe path) "
                          "before saving final/ -- Trainer's own end-of-run reload has the "
                          "same tied-weights gap as resume", trainer.state.best_model_checkpoint)
        trainer.model = AutoModelForObjectDetection.from_pretrained(
            trainer.state.best_model_checkpoint,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
        ).to(device)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    image_processor.save_pretrained(str(final_dir))
    logger_local.info("Saved final model + processor to %s", final_dir)

    metrics = trainer.evaluate()
    logger_local.info("Final eval_loss: %s", metrics)


if __name__ == "__main__":
    main()
