"""Reconstruct the released benchmark images from ImageNet and saved masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def crop_image(image: Image.Image, size: int = 224) -> Image.Image:
    """Match the original bilinear resize and center-crop preprocessing."""
    width, height = image.size
    if min(width, height) != size:
        if width < height:
            resized = (size, int(size * height / width))
        else:
            resized = (int(size * width / height), size)
        image = image.resize(resized, Image.Resampling.BILINEAR)
    left = int(round((image.width - size) / 2))
    top = int(round((image.height - size) / 2))
    return image.crop((left, top, left + size, top + size))


def generate(
    imagenet_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    cropped: bool = False,
    conditions: Optional[list[str]] = None,
) -> int:
    """Write occluded PNGs using the released masks; return the number created."""
    from pycocotools import mask as mask_utils

    with (metadata_dir / "gt.csv").open() as handle:
        labels = {row["filename"]: row for row in csv.DictReader(handle)}
    categories = {filename: row["gt_cat"] for filename, row in labels.items()}
    condition_dirs = sorted(path.parent for path in metadata_dir.glob("*/masks.json"))
    if conditions:
        known = {path.name for path in condition_dirs}
        unknown = set(conditions) - known
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        condition_dirs = [path for path in condition_dirs if path.name in conditions]
    if not condition_dirs:
        raise ValueError(f"No mask conditions found in {metadata_dir}")

    # Fail before exporting if an ImageNet source is missing.
    for filename, category in categories.items():
        source = imagenet_dir / category / filename
        if not source.is_file():
            raise FileNotFoundError(source)

    originals = output_dir / "orig_cropped_images"
    originals.mkdir(parents=True, exist_ok=True)
    pixels_by_name = {}
    for filename, category in categories.items():
        source = imagenet_dir / category / filename
        target = originals / filename
        if not target.exists():
            if cropped:
                shutil.copyfile(source, target)
            else:
                with Image.open(source) as image:
                    crop_image(image).save(target)
        # The original pipeline also saves JPEG crops before applying the masks.
        with Image.open(target) as image:
            if image.size != (224, 224):
                raise ValueError(f"Expected a 224 x 224 crop: {target}")
            pixels_by_name[filename] = np.array(image.convert("RGB"))

    # Same integer-rounded ImageNet mean color as mk_hexp.py.
    color = np.array([round(255 * x) for x in (0.485, 0.456, 0.406)], dtype=np.uint8)
    created = 0
    for condition in condition_dirs:
        output = output_dir / condition.name / "images"
        output.mkdir(parents=True, exist_ok=True)
        for name in ("cfg.json", "masks.json"):
            source = condition / name
            target = output.parent / name
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        masks = json.loads((condition / "masks.json").read_text())
        for filename, rle in masks.items():
            target = output / f"{Path(filename).stem}_{condition.name}.png"
            if target.exists():
                continue
            pixels = pixels_by_name[filename].copy()
            if rle is not None:
                mask = mask_utils.decode(rle).astype(bool)
                if mask.shape != pixels.shape[:2]:
                    raise ValueError(f"Mask dimensions do not match {filename}")
                pixels[mask] = color
            Image.fromarray(pixels).save(target)
            created += 1
        print(f"{condition.name}: {len(masks)} images", flush=True)

    for name in ("gt.csv", "id_to_im_paths.json"):
        source = metadata_dir / name
        target = output_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    # Inference needs one label per image, rather than repeated human trials.
    with (output_dir / "image_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "gt_cat", "gt_idx"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(labels.values())
    return created


def main(raw_args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        required=True,
        help="ImageNet validation directory, with one folder per synset",
    )
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/human_occ_exp/hexp1.1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output experiment directory; defaults to the metadata directory",
    )
    parser.add_argument(
        "--cropped",
        action="store_true",
        help="Source images are already the original 224 x 224 JPEG crops",
    )
    parser.add_argument(
        "--conditions", nargs="+", help="Optional subset of saved condition directory names"
    )
    args = parser.parse_args(raw_args)
    created = generate(
        args.imagenet_dir,
        args.metadata_dir,
        args.output_dir or args.metadata_dir,
        args.cropped,
        args.conditions,
    )
    print(f"Created {created} images. Existing images were skipped.")


if __name__ == "__main__":
    main()
