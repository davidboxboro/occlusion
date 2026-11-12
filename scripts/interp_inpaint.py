from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
import json
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor

import sys

util_path = Path.cwd().parent / "src" / "util"
if str(util_path) not in sys.path:
    sys.path.append(str(util_path))
from mask_util import RLEToMask


def interp(im: np.ndarray, m: np.ndarray, method: str) -> np.ndarray:
    # must be floats
    im = im.astype(float)
    m = m.astype(float)
    # Mask convention: 1 means occluded, 0 means observed.
    inv_m = 1 - m
    # copy the original im to use for the interpolation
    im_interp = np.copy(im)
    # perform the interpolation for each color channel
    for k in range(im.shape[2]):
        im_chan = im[:, :, k]
        non_missing_values = im_chan[inv_m.astype(bool)]
        non_missing_coords = np.array(np.nonzero(inv_m)).T
        missing_coords = np.array(np.nonzero(m)).T
        interpolated_values = griddata(
            non_missing_coords, non_missing_values, missing_coords, method=method
        )
        im_interp[missing_coords[:, 0], missing_coords[:, 1], k] = interpolated_values
    return im_interp


def inpaint_im(im_path: Path, subexp_path: Path, out_subexp_path: Path) -> None:
    # return if already done
    im_name = im_path.name
    if (out_subexp_path / im_name).exists():
        return

    unocc_im_name = "_".join(im_name.split("_")[:3]) + ".JPEG"

    pil_im = Image.open(im_path)
    im = np.array(pil_im)

    # mask image
    with open(subexp_path / "masks.json", "r") as f:
        mask_d = json.load(f)
    rle_m = mask_d[unocc_im_name]
    m = RLEToMask(rle_m)

    # linear interpolation on missing pixels
    im_interp = interp(im, m, method="linear")

    # Nearest-neighbor fill handles pixels outside the linear interpolation hull.
    nan_m = np.isnan(im_interp[:, :, 0])
    im_interp2 = interp(im_interp, nan_m, method="nearest")

    # save
    out_pil_im = Image.fromarray(im_interp2.astype(np.uint8))
    out_pil_im.save(out_subexp_path / im_name)


def main() -> None:
    out_exp_path = Path("../results/hexp1.1/inpaint/interp")
    exp_path = Path("../data/human_occ_exp/hexp1.1")
    subexp_paths = [e for e in exp_path.glob("*") if "simplex_noise" in e.name or "empty" in e.name]

    futures = []
    with ProcessPoolExecutor(max_workers=80) as executor:
        for i, subexp_path in enumerate(subexp_paths):
            subexp_name = subexp_path.name
            out_subexp_path = out_exp_path / subexp_name
            if not out_subexp_path.exists():
                out_subexp_path.mkdir(parents=True)
            im_paths = sorted((subexp_path / "images").glob("*"))
            for im_path in im_paths:
                future = executor.submit(
                    inpaint_im,
                    im_path=im_path,
                    subexp_path=subexp_path,
                    out_subexp_path=out_subexp_path,
                )
                futures.append(future)
                if len(futures) % 100 == 0:
                    print(f"Submitted {len(futures)} tasks.")

    # wait
    results = [future.result() for future in futures]
    print("Done.")


if __name__ == "__main__":
    main()
