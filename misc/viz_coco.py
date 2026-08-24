#!/usr/bin/env python3

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def load_coco_annotations(annotation_path: Path):
    """Load COCO data and create annotation/category lookup tables."""
    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    category_names = {
        category["id"]: category.get("name", str(category["id"]))
        for category in categories
    }

    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        bbox = annotation.get("bbox")

        # Keep only annotations with a valid, non-empty bounding box.
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and bbox[2] > 0
            and bbox[3] > 0
        ):
            annotations_by_image[annotation["image_id"]].append(annotation)

    return images, annotations_by_image, category_names


def resolve_image_path(image_dir: Path, file_name: str) -> Path | None:
    """Resolve a COCO file_name against the supplied image directory."""
    normalized_name = file_name.replace("\\", "/")
    relative_path = Path(normalized_name)

    candidates = [
        image_dir / relative_path,
        image_dir / relative_path.name,
        relative_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def visualize_random_images(
    image_dir: Path,
    annotation_path: Path,
    num_images: int,
    seed: int | None = None,
    columns: int = 3,
    show_annotation_ids: bool = False,
    save_path: Path | None = None,
):
    images, annotations_by_image, category_names = load_coco_annotations(
        annotation_path
    )

    # Select only images that have at least one valid annotation.
    annotated_images = [
        image_info
        for image_info in images
        if annotations_by_image.get(image_info["id"])
    ]

    if not annotated_images:
        raise ValueError(
            "No images with valid bounding-box annotations were found."
        )

    rng = random.Random(seed)

    num_images = min(num_images, len(annotated_images))
    selected_images = rng.sample(annotated_images, num_images)

    columns = max(1, min(columns, num_images))
    rows = math.ceil(num_images / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for axis, image_info in zip(axes_flat, selected_images):
        stored_file_name = image_info["file_name"]

        image_path = resolve_image_path(
            image_dir=image_dir,
            file_name=stored_file_name,
        )

        if image_path is None:
            axis.set_title(
                f"Missing: {Path(stored_file_name).name}",
                fontsize=8,
            )
            axis.axis("off")
            print(f"Could not find image: {stored_file_name}")
            continue

        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        image_annotations = annotations_by_image[image_info["id"]]

        for annotation in image_annotations:
            x_min, y_min, width, height = annotation["bbox"]

            category_id = annotation.get("category_id")
            category_name = category_names.get(
                category_id,
                f"class_{category_id}",
            )

            rectangle = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=1.25,
                edgecolor="red",
                facecolor="none",
            )
            axis.add_patch(rectangle)

            label = category_name

            if show_annotation_ids:
                label += f" | {annotation.get('id', '?')}"

            axis.text(
                x_min,
                max(0, y_min - 2),
                label,
                fontsize=6,
                verticalalignment="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )

        # Path(...).name removes train/images/ from the displayed title.
        display_name = Path(stored_file_name.replace("\\", "/")).name

        axis.set_title(
            f"{display_name} ({len(image_annotations)} boxes)",
            fontsize=7,
        )
        axis.axis("off")

    for axis in axes_flat[len(selected_images):]:
        axis.axis("off")

    figure.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
        print(f"Saved visualization to: {save_path}")

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize random COCO images that contain bounding-box "
            "annotations."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Dataset root or image directory.",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to the COCO annotation JSON.",
    )

    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=6,
        help="Number of annotated images to display.",
    )

    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of grid columns.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection.",
    )

    parser.add_argument(
        "--show-annotation-ids",
        action="store_true",
        help="Include annotation IDs in box labels.",
    )

    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.num_images <= 0:
        raise ValueError("--num-images must be greater than zero.")

    if not args.images.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {args.images}"
        )

    if not args.annotations.is_file():
        raise FileNotFoundError(
            f"Annotation file does not exist: {args.annotations}"
        )

    visualize_random_images(
        image_dir=args.images,
        annotation_path=args.annotations,
        num_images=args.num_images,
        seed=args.seed,
        columns=args.columns,
        show_annotation_ids=args.show_annotation_ids,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()