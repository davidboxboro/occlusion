from __future__ import annotations
from typing import Optional

# Generate data for human experiment.

# main imports
import numpy as np
import hashlib
import pandas as pd
from PIL import Image
import json
import shutil
from pathlib import Path
from itertools import product
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.transforms import ToNumpy
from timm.data import occ_transforms
from concurrent.futures import ProcessPoolExecutor
import sys
import os

# add src dir to python path
proj_dir = Path.cwd().parent
src_dir = proj_dir / "src"
if src_dir not in sys.path:
    sys.path.append(str(src_dir))

# imports in src
from util import mask_util


def main() -> None:
    # numpy random seed
    seed = 7
    np.random.seed(seed)

    # params
    exp_name = "hexp1.2"
    h, w = 224, 224
    mean_color = tuple(round(255 * e) for e in IMAGENET_DEFAULT_MEAN)
    noise_freq_unique = [2**i for i in range(0, 9)]
    occ_fracs = [0.125, 0.25, 0.5, 0.75, 0.875]
    npix_occ_unique = [int(h * w * occ_frac) for occ_frac in occ_fracs]
    ncat = 50
    nim_per_cat = 8
    nim_per_subj = 100
    ncontrol_repeat = 5
    # derived params
    nim = ncat * nim_per_cat
    assert nim % nim_per_subj == 0
    assert nim_per_subj % ncat == 0
    nocc_type = (
        len(noise_freq_unique) * len(npix_occ_unique) + ncontrol_repeat
    )  # add ncontrol_repeat for unoccluded case
    nsubj = nocc_type * nim // nim_per_subj
    nsubj_group = nim // nim_per_subj
    print(f"Noise freqs: {noise_freq_unique} ({len(noise_freq_unique)} total)")
    print(
        f"Num pixels occluded: {npix_occ_unique} | Occ fracs: {occ_fracs} ({len(npix_occ_unique)} total)"
    )
    print(f"Num images: {nim} | Num images per cat: {nim_per_cat} | Num cat: {ncat}")
    print(f"Num occlusion types: {nocc_type} | Num control repeats: {ncontrol_repeat}")
    print(
        f"Num subjects: {nsubj} | Num images per subject: {nim_per_subj} | Num subject groups: {nsubj_group}"
    )

    # paths
    data_dir = Path("/storage/davidlu/projects/objectnet-oc/data")
    min_viewing_times_loc = data_dir / "min_viewing_times/cropped_images"
    gt_loc = data_dir / "imagenet/ann/gt.csv"
    in_dir = data_dir / "imagenet/val_cropped"
    out_dir = data_dir / "human_occ_exp" / exp_name
    cropped_img_dir = out_dir / "orig_cropped_images"

    # remove old data and create new directory
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    cropped_img_dir.mkdir()

    # get cats
    cat_paths = min_viewing_times_loc.glob("*")
    cat_unique = sorted([cat_path.name for cat_path in cat_paths])
    assert len(cat_unique) == ncat
    print(f"\nCats: {cat_unique}")

    # get ground truth labels
    gt_df = pd.read_csv(gt_loc, index_col=0)

    # store file names and cats in DF
    fns_2d = []
    fns = []
    cats = []
    idxs = []
    for cat in cat_unique:
        in_cat_dir = in_dir / cat
        paths = in_cat_dir.glob("*")
        paths = list(paths)[:nim_per_cat]
        new_fns = sorted([path.name for path in paths])
        for fn in new_fns:
            gt_cat = gt_df.loc[fn].gt_cat
            gt_idx = gt_df.loc[fn].gt_idx
            assert cat == gt_cat
            cats.append(cat)
            idxs.append(gt_idx)
            fns.append(fn)
            # copy image over
            os.symlink(in_dir / cat / fn, cropped_img_dir / fn)
        fns_2d.append(new_fns)
    base_df = pd.DataFrame({"filename": fns, "gt_cat": cats, "gt_idx": idxs})
    base_df = base_df.set_index("filename")
    # fns_2d is a 2D list. compress it to a 1D list where every sublist of size ncat
    # contains 1 image from each cat
    fns_ordered = list(np.array(fns_2d).T.reshape(-1))
    # sort according to fns_ordered so that categories are evenly distributed
    # throughout DF
    base_df = base_df.loc[fns_ordered]
    assert len(base_df) == nim
    print(f"\nBase DF:\n{base_df}")

    # make set of occlusions for each occlusion type subexperiment
    futures = []
    with ProcessPoolExecutor(max_workers=80) as executor:
        # one subexp for empty occlusion case
        # IMPORTANT: empty transform comes first
        occ_trans_str = "empty"
        future = executor.submit(
            mk_subexp,
            base_df=base_df,
            in_dir=in_dir,
            out_dir=out_dir,
            h=h,
            w=w,
            mean_color=mean_color,
            occ_trans_str=occ_trans_str,
        )
        futures.append(future)
        # many subexps for simplex noise
        occ_trans_str = "simplex_noise"
        for noise_freq, npix_occ in product(noise_freq_unique, npix_occ_unique):
            future = executor.submit(
                mk_subexp,
                base_df=base_df,
                in_dir=in_dir,
                out_dir=out_dir,
                h=h,
                w=w,
                mean_color=mean_color,
                occ_trans_str=occ_trans_str,
                noise_freq=noise_freq,
                npix_occ=npix_occ,
            )
            futures.append(future)

    # make mapping of link id to image filenames
    occ_dfs = [future.result() for future in futures]

    # divide base_df into groups and add group-id column
    # IMPORTANT: for each category, the images are ordered alphabetically
    # from lower group indices to higher group indices
    base_df_groups = []
    for subj_group in range(nsubj_group):
        im_idx_offset = subj_group * nim_per_subj
        base_df_group = base_df.iloc[im_idx_offset : im_idx_offset + nim_per_subj]
        # IMPORTANT: shuffle elements of each base_df_group so that 'correlations'
        # are removed between occlusion type and image for a given subject
        base_df_group = base_df_group.sample(frac=1, random_state=seed)
        base_df_groups.append(base_df_group)

    # make sure first DF is unoccluded
    empty_occ_df = occ_dfs.pop(0)
    occ_trans = empty_occ_df.occ_transform.unique()
    assert len(occ_trans) == 1
    assert occ_trans.item() == "empty"
    # add ncontrol_repeat copies of it
    # this is because we may want more controls than 1 instance of each image across all subjects
    # this is hacky, but space out controls in occ_dfs to make all subjects get roughly same num of controls
    interval = len(occ_dfs) // ncontrol_repeat + 1
    for i in range(ncontrol_repeat):
        # make a copy so that they are independently modifiable
        occ_dfs.insert(i * interval, empty_occ_df.copy())

    # for each image, assign it to the right id
    id_to_paths = {}
    for group_idx, base_df_group in enumerate(base_df_groups):
        # base_df_group contains only nim_per_subj images
        for subj_idx in range(nocc_type):
            # generate random subject id
            id_ = str(group_idx) + "-"
            id_ += hashlib.sha256(np.random.bytes(128)).hexdigest()[:10]
            assert id_ not in id_to_paths
            id_paths = []
            # select main images for subject as follows (assuming offset of 0):
            # subject 0: im 0, occ 0; im 1, occ 1; im 2: occ 3 ...
            # subject 1: im 0, occ 1; im 1, occ 2; im 2: occ 4 ...
            for im_idx in range(nim_per_subj):
                occ_idx = (subj_idx + im_idx) % nocc_type
                occ_df = occ_dfs[occ_idx]
                # offset used to differentiate between groups
                fn = base_df_group.index[im_idx]
                im_path = occ_df.loc[fn].im_path
                # add subject id to occ_df
                if "task_id" not in occ_df:
                    occ_df["task_id"] = ""
                assert occ_df.loc[fn, "task_id"] == ""
                occ_df.loc[fn, "task_id"] = id_
                id_paths.append(im_path)
            assert len(id_paths) == nim_per_subj
            assert len(id_paths) == len(set(id_paths))  # no repeats
            # app takes in dictionary form {'image': image_path}
            id_paths = [{"image": path} for path in id_paths]
            # shuffle images so that occlusion types are in a random order
            np.random.shuffle(id_paths)
            id_to_paths[id_] = id_paths

    # save aggregate data
    with open(out_dir / "id_to_im_paths.json", "w") as f:
        json.dump(id_to_paths, f, indent=4)
    # make and save big DF with all instances of images
    big_occ_df = pd.concat(occ_dfs, axis=0)
    # add group index column
    big_occ_df["group"] = big_occ_df.task_id.str.split("-").str[0]
    assert len(big_occ_df) == nim * nocc_type
    # no subject sees same images twice
    subj_value_counts = big_occ_df.reset_index().groupby("task_id").filename.value_counts()
    assert subj_value_counts.max() == subj_value_counts.min() == 1
    # each filename appears nocc_type times
    fn_value_counts = big_occ_df.index.value_counts()
    assert fn_value_counts.max() == fn_value_counts.min() == nocc_type
    # number of controls
    ncontrol_info = (
        big_occ_df[big_occ_df.occ_transform == "empty"]
        .groupby("task_id")
        .im_path.count()
        .describe()
    )
    print(f"\nNum controls:\n{ncontrol_info}")
    print(f"\nBig occlusion DF:\n{big_occ_df}")
    big_occ_df.to_csv(out_dir / "gt.csv")


def mk_subexp(
    base_df: pd.DataFrame,
    in_dir: Path,
    out_dir: Path,
    h: int,
    w: int,
    mean_color: tuple[int, int, int],
    occ_trans_str: str,
    noise_freq: Optional[int] = None,
    npix_occ: Optional[int] = None,
) -> pd.DataFrame:
    """
    Export one mask condition and return its image-level metadata.
    """
    # print(f'Making subexp for ({noise_freq}, {npix_occ})')
    # make subexp dir
    assert occ_trans_str in ["simplex_noise", "empty"]
    if occ_trans_str == "simplex_noise":
        subexp_name = f"{occ_trans_str}_{noise_freq}_{npix_occ}"
        occ_transform = occ_transforms.SimplexNoiseOccTransform(
            noise_freq=noise_freq, npix_occ=npix_occ, color=mean_color, ret_mask=False
        )
        cfg = {
            "occ_transform": occ_trans_str,
            "noise_freq": noise_freq,
            "npix_occ": npix_occ,
            "occ_frac": npix_occ / (h * w),
        }
    elif occ_trans_str == "empty":
        subexp_name = f"{occ_trans_str}"
        occ_transform = occ_transforms.EmptyOccTransform(ret_mask=False)
        cfg = {
            "occ_transform": occ_trans_str,
            "occ_frac": 0,
        }
    else:
        raise ValueError()
    # make new DF
    occ_df = base_df.copy()
    for k, v in cfg.items():
        occ_df[k] = v
    subexp_out_dir = out_dir / subexp_name
    subexp_out_dir.mkdir()
    with open(subexp_out_dir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=4)
    # make and save images and masks for subexp
    occ_ims = {}
    occ_masks = {}
    for i in range(len(occ_df)):
        # make occluded image and mask
        fn, cat = occ_df.index[i], occ_df.iloc[i].gt_cat
        im = Image.open(f"{in_dir}/{cat}/{fn}").convert("RGB")
        np_im = ToNumpy()(im)
        occ_np_im = occ_transform(np_im)
        occ_im = Image.fromarray(occ_np_im.transpose(1, 2, 0).astype("uint8"), "RGB")
        occ_mask = occ_transform.occ_m
        occ_rle = mask_util.maskToRLE(occ_mask)  # occ_rle will be None if empty occlusion used
        # save stuff
        occ_ims[fn] = occ_im
        occ_masks[fn] = occ_rle
    with open(subexp_out_dir / "masks.json", "w") as f:
        json.dump(occ_masks, f, indent=4)
    out_im_dir = subexp_out_dir / "images"
    out_im_dir.mkdir()
    out_im_paths = []
    for fn, occ_im in occ_ims.items():
        fn, old_suff = fn.split(".")
        assert old_suff == "JPEG"
        occ_fn = f"{fn}_{subexp_name}.png"
        out_im_path = out_im_dir / occ_fn
        occ_im.save(out_im_path)
        rel_path = Path(*out_im_path.parts[-4:])
        out_im_paths.append(str(rel_path))
    occ_df["im_path"] = out_im_paths
    return occ_df


if __name__ == "__main__":
    main()
