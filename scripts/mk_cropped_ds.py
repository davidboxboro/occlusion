from __future__ import annotations

# make cropped dataset

from PIL import Image
from torchvision import transforms
import math
from timm.data.transforms import str_to_interp_mode
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import shutil


def main() -> None:
    crop_pct = 1.0
    img_size = (224, 224)
    interpolation = "bilinear"

    scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    assert scale_size[0] == scale_size[1]

    resize_transform = transforms.Resize(
        scale_size[0], interpolation=str_to_interp_mode(interpolation)
    )
    crop_transform = transforms.CenterCrop(img_size)
    transform = transforms.Compose([resize_transform, crop_transform])

    # transform each image and save
    proj_dir = Path.cwd().parent
    img_dir = proj_dir / "data/imagenet/val"
    cropped_img_dir = proj_dir / "data/imagenet/val_cropped"
    if cropped_img_dir.exists():
        shutil.rmtree(cropped_img_dir)
    cropped_img_dir.mkdir()
    cat_dirs = img_dir.glob("*")
    with ProcessPoolExecutor(max_workers=80) as e:
        for cat_dir in cat_dirs:
            e.submit(
                crop_cat, cat_dir=cat_dir, transform=transform, cropped_img_dir=cropped_img_dir
            )
    print("Done.")


def crop_cat(cat_dir: Path, transform: transforms.Compose, cropped_img_dir: Path) -> None:
    cat = cat_dir.name
    for img_loc in cat_dir.glob("*"):
        img_name = img_loc.name
        cropped_img_loc = cropped_img_dir / cat / img_name
        if not cropped_img_loc.parent.exists():
            cropped_img_loc.parent.mkdir()
        img = Image.open(img_loc)
        cropped_img = transform(img)
        cropped_img.save(cropped_img_loc)


if __name__ == "__main__":
    main()
