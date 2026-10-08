"""Environment / dependency sanity check for the RT-DETRv2 (HF) training pipeline.

Run this BEFORE submitting a SLURM job — it's meant to fail in seconds on a
missing package, a torch/torchvision CUDA mismatch, or an uncached checkpoint,
rather than after you've sat in the queue and burned GPU allocation time.

Run:
    python -m training.hugging_face.check_env
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REQUIRED = ("torch", "transformers", "accelerate")
RECOMMENDED = {
    "torchvision": "torchvision",
    "pycocotools": "pycocotools",
    "torchmetrics": "torchmetrics",
    "matplotlib": "matplotlib",
    "tensorboard": "tensorboard",
    "PIL": "Pillow",
    "huggingface_hub": "huggingface_hub",
}


def _report(module_name: str, pip_name: str, required: bool) -> bool:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [ok] {module_name:<15} {version}")
        return True
    except ImportError as e:
        tag = "MISSING (required)" if required else "missing (optional, see note below)"
        print(f"  [!!] {module_name:<15} {tag} — pip install {pip_name} ({e.__class__.__name__})")
        return not required


def check_packages() -> bool:
    print("=== Packages ===")
    ok = True
    for name in REQUIRED:
        ok &= _report(name, name, required=True)
    for module_name, pip_name in RECOMMENDED.items():
        _report(module_name, pip_name, required=False)
    print("  (accelerate is required for Trainer; tensorboard for report_to=['tensorboard'];")
    print("   pycocotools/torchmetrics/matplotlib are used by dataset.py / eval.py — missing")
    print("   ones here will surface as import errors there, not in train.py directly.)")
    return ok


def check_cuda() -> bool:
    print("\n=== CUDA / GPU ===")
    import torch
    if not torch.cuda.is_available():
        print("  [!!] torch.cuda.is_available() is False — no GPU visible to this process")
        return False
    print(f"  torch build: {torch.__version__}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  [ok] cuda:{i} {name} (compute {cap[0]}.{cap[1]}, {mem:.0f} GB)")
        if "H200" not in name:
            print(f"       note: expected an H200 — check your SLURM --gres/--gpus request")
    return True


def check_torchvision_ops() -> bool:
    print("\n=== torchvision CUDA ops ===")
    try:
        import torch
        import torchvision
        from torchvision.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
        scores = torch.tensor([0.9, 0.8], device="cuda")
        nms(boxes, scores, 0.5)
        print(f"  [ok] torchvision {torchvision.__version__} CUDA ops working")
        return True
    except ImportError as e:
        print(f"  [!!] torchvision not importable: {e}")
        return False
    except Exception as e:
        print(f"  [!!] torchvision CUDA op failed — likely a torch/torchvision CUDA build "
              f"mismatch (reinstall torchvision from the cu130 wheel index): {e}")
        return False


def check_rtdetr_available() -> bool:
    print("\n=== transformers: RT-DETRv2 ===")
    try:
        from transformers import RTDetrV2ForObjectDetection, AutoImageProcessor  # noqa: F401
        print("  [ok] RTDetrV2ForObjectDetection importable")
        return True
    except ImportError as e:
        print(f"  [!!] RTDetrV2ForObjectDetection not available — upgrade transformers: {e}")
        return False


def check_project_modules() -> bool:
    print("\n=== Project modules ===")
    ok = True
    for mod in ("config", "augmentations", "dataset", "logging_utils"):
        try:
            importlib.import_module(f"training.hugging_face.{mod}")
            print(f"  [ok] training.hugging_face.{mod}")
        except Exception as e:
            print(f"  [!!] training.hugging_face.{mod} failed to import: {e}")
            ok = False
    return ok


def check_data_paths() -> bool:
    print("\n=== Data paths (config.py) ===")
    try:
        from training.hugging_face import config
    except Exception as e:
        print(f"  [??] couldn't import training.hugging_face.config, skipping: {e}")
        return True

    print(f"  DATA_ROOT        = {config.DATA_ROOT}")
    print(f"  SPECTROGRAM_ROOT = {config.SPECTROGRAM_ROOT}")

    ok = True
    targets = [("dataset_manifest.csv", config.MANIFEST_PATH)]
    for split in getattr(config, "SPLITS", ("train", "val", "test")):
        targets.append((f"coco_{split}.json", config.SPECTROGRAM_ROOT / f"coco_{split}.json"))
        targets.append((f"{split}/images/", config.SPECTROGRAM_ROOT / split / "images"))

    for label, path in targets:
        exists = Path(path).exists()
        print(f"  [{'ok' if exists else '!!'}] {label:<20} {path}")
        ok &= exists

    if not ok:
        print("  Missing paths above usually mean FD_DATA_ROOT isn't set (or is wrong) in this")
        print("  environment -- config.py silently falls back to its hardcoded default otherwise.")
    return ok


def check_checkpoint_cached() -> bool:
    print("\n=== Pretrained checkpoint cache ===")
    try:
        from training.hugging_face import config
        checkpoint = config.DEFAULT_CHECKPOINT
    except Exception:
        print("  [??] couldn't read config.DEFAULT_CHECKPOINT, skipping")
        return True

    found = False
    try:
        from huggingface_hub import try_to_load_from_cache
        found = bool(try_to_load_from_cache(checkpoint, "config.json"))
    except Exception:
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        found = (cache_root / ("models--" + checkpoint.replace("/", "--"))).exists()

    if found:
        print(f"  [ok] '{checkpoint}' found in local HF cache")
        return True
    print(f"  [!!] '{checkpoint}' NOT found in local HF cache.")
    print("       Most HPC compute nodes have no internet access — prefetch it from a")
    print("       login node first:")
    print(f"       python -c \"from transformers import AutoModelForObjectDetection, "
          f"AutoImageProcessor; AutoModelForObjectDetection.from_pretrained('{checkpoint}'); "
          f"AutoImageProcessor.from_pretrained('{checkpoint}')\"")
    return False


def main() -> None:
    results = [
        check_packages(),
        check_cuda(),
        check_torchvision_ops(),
        check_rtdetr_available(),
        check_project_modules(),
        check_data_paths(),
        check_checkpoint_cached(),
    ]
    print("\n" + "=" * 40)
    if all(results):
        print("All checks passed.")
        sys.exit(0)
    print("One or more checks failed — see [!!] lines above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
