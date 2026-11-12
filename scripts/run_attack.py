from __future__ import annotations
from typing import Any, Callable

from pathlib import Path
from PIL import Image
import numpy as np
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import attack
import random
import time
import torch
import functools
import timm.data.occ_transforms as occT
from timm.data.constants import IMAGENET_DEFAULT_MEAN

mean_color = tuple(round(255 * e) for e in IMAGENET_DEFAULT_MEAN)
import sys
import argparse
import os


def flush_print(*args: object, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


def run_one_attack(
    cfg: dict[str, Any],
    im_paths: list[Path],
    model: torch.nn.Module,
    out_path: Path,
    dummy_oa: attack.OccAttacker,
    im_idx: int,
) -> None:
    name, occ_frac = cfg["oa_name"], cfg["occ_frac"]
    out_im_path = out_path / f"{name}_{occ_frac}"
    if not out_im_path.exists():
        out_im_path.mkdir(parents=True)
    out_results_path = out_path / "results"
    if not out_results_path.exists():
        out_results_path.mkdir(parents=True)
    save_path = out_results_path / f"{name}_{occ_frac}_{im_idx}.csv"
    if save_path.exists():
        flush_print("Results already exist, exiting.")
        return
    start = time.time()
    results = []
    for i, im_path in enumerate(im_paths):
        fn = im_path.name
        res = run_one_image(
            im_path=im_path,
            run_fn=cfg["run_fn"],
            model=model,
            dummy_oa=dummy_oa,
            out_im_path=out_im_path,
        )
        res |= cfg.copy()
        res["filename"] = fn
        del res["run_fn"]
        results.append(res)
        time_elapsed = time.time() - start
        flush_print(
            f"\tFinished image {i + 1} of {len(im_paths)}, {time_elapsed / 60:.2f} minutes elapsed."
        )
    attack_df = pd.DataFrame(results)
    attack_df.to_csv(save_path)
    flush_print(f"Saved results, done.")


def run_one_image(
    im_path: Path,
    run_fn: Callable[..., np.ndarray],
    model: torch.nn.Module,
    dummy_oa: attack.OccAttacker,
    out_im_path: Path,
) -> dict[str, Any]:
    fn = im_path.name
    gt_idx = gt_df.loc[fn].gt_idx
    im = np.array(Image.open(im_path).convert("RGB"))

    occ_im = run_fn(im, gt_idx=gt_idx)
    occ_im_pil = Image.fromarray(occ_im)
    occ_im_pil.save(out_im_path / fn)

    torch_im = dummy_oa.to_model_fmt(occ_im)
    with torch.no_grad():
        im_probs = model(torch_im.reshape(1, *torch_im.shape)).softmax(-1)
    im_probs = im_probs.squeeze().cpu().numpy()

    targ_prob = im_probs[gt_idx]

    # set prob to negative so it's not chosen
    im_probs[gt_idx] = -1
    best_nontarg_prob = im_probs.max()
    best_nontarg_idx = im_probs.argmax()
    best_nontarg_label = classes_df.loc[best_nontarg_idx].label
    correct = targ_prob > best_nontarg_prob

    return {
        "targ_prob": targ_prob,
        "best_nontarg_prob": best_nontarg_prob,
        "gt_idx": gt_idx,
        "best_nontarg_idx": best_nontarg_idx,
        "correct": correct,
    }


# run functions
def run_oa(im: np.ndarray, oa: attack.OccAttacker, gt_idx: int) -> np.ndarray:
    torch_im = oa.to_model_fmt(im)
    torch_occ_im, _ = oa.run_attack(im=torch_im, gt_idx=gt_idx)
    de_occ_im = oa.from_model_fmt(torch_occ_im)
    return de_occ_im


def run_simplex_noise(im: np.ndarray, noise_freq: int, npix_occ: int, gt_idx: int) -> np.ndarray:
    occ_transform = occT.SimplexNoiseOccTransform(
        noise_freq=noise_freq,
        npix_occ=npix_occ,
        color=mean_color,
        ret_mask=False,
    )
    rand_occ_im = occ_transform(im.transpose(2, 0, 1)).transpose(1, 2, 0)
    return rand_occ_im


def run_patch(
    im: np.ndarray, npatch_occ: int, npatch_h: int, npatch_w: int, gt_idx: int
) -> np.ndarray:
    occ_transform = occT.PatchOccTransform(
        npatch_occ=npatch_occ,
        npatch_h=npatch_h,
        npatch_w=npatch_w,
        color=mean_color,
        ret_mask=False,
    )
    rand_occ_im = occ_transform(im.transpose(2, 0, 1)).transpose(1, 2, 0)
    return rand_occ_im


def run_none(im: np.ndarray, gt_idx: int) -> np.ndarray:
    return im


# global vars
gt_path = Path("../data/imagenet/ann/gt.csv")
classes_path = Path("../data/imagenet/ann/classes.csv")
ims_path = Path("../data/human_occ_exp/hexp1.1/orig_cropped_images")
hgt_path = Path("../data/human_occ_exp/hexp1.1/gt.csv")

gt_df = pd.read_csv(gt_path, index_col=0)
classes_df = pd.read_csv(classes_path, index_col=0)


def main(attack_idx: int, im_idx: int, exp_name: str) -> None:
    # out dir
    out_path = Path(f"../results/{exp_name}")
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # restrict to 200 images humans tested on
    hgt_df = pd.read_csv(hgt_path, index_col=0)
    fns = set(hgt_df[hgt_df.task_id.str[0].isin(["0", "1"])].index.unique().tolist())
    im_paths = list(ims_path.glob("*.JPEG"))
    im_paths = [e for e in im_paths if e.name in fns]
    im_paths = sorted(im_paths)
    assert len(im_paths) == 200
    # get only batch corresponding to image index
    batch_size = 40
    im_paths = im_paths[im_idx * batch_size : (im_idx + 1) * batch_size]

    # model
    model_name = "resnet50"
    # checkpoint_path = '../results/exp1.1/ft/edeb5a5d11e01f30/model_best.pth.tar'
    checkpoint_path = ""
    model = timm.create_model(model_name, pretrained=True, checkpoint_path=checkpoint_path).to(
        "cuda"
    )
    model.eval()

    # create attacker instances
    occ_fracs_unique = [2**i for i in [-5, -4, -3, -2]]

    cfgs = []

    cfg = {
        "oa_name": "none",
        "occ_frac": 0,
        "run_fn": run_none,
    }
    cfgs.append(cfg)

    noise_freqs = [1, 16, 256]
    for noise_freq in noise_freqs:
        for occ_frac in occ_fracs_unique:
            npix_occ = round(224 * 224 * occ_frac)
            run_simplex_noise_partial = functools.partial(
                run_simplex_noise, noise_freq=noise_freq, npix_occ=npix_occ
            )
            cfg = {
                "oa_name": f"rand_simplex_noise_{noise_freq}",
                "occ_frac": occ_frac,
                "run_fn": run_simplex_noise_partial,
                "noise_freq": noise_freq,
            }
            cfgs.append(cfg)

    npatch_h, npatch_w = 8, 8
    for occ_frac in occ_fracs_unique:
        npatch_occ = round(npatch_h * npatch_w * occ_frac)
        run_rand_partial = functools.partial(
            run_patch, npatch_occ=npatch_occ, npatch_h=npatch_h, npatch_w=npatch_w
        )
        cfg = {
            "oa_name": f"rand_patch_{npatch_h}",
            "occ_frac": occ_frac,
            "run_fn": run_rand_partial,
            "npatch_h": npatch_h,
            "npatch_w": npatch_w,
        }
        cfgs.append(cfg)

    npatch_h, npatch_w = 8, 8
    maxiter = 100
    popsize = 32
    mutation = 0.5
    for occ_frac in occ_fracs_unique:
        npatch_occ = round((npatch_h * npatch_w) * occ_frac)
        oa_de = attack.OccAttacker(
            attack_type="diff_ev",
            model_name=model_name,
            npatch_occ=npatch_occ,
            npatch_h=npatch_h,
            npatch_w=npatch_w,
            maxiter=maxiter,
            popsize=popsize,
            diff_ev_obj="prob_diff",
            mutation=mutation,
            verbose=False,
        )
        run_de = functools.partial(run_oa, oa=oa_de)
        cfg = {
            "oa_name": f"diff_ev_{npatch_h}_{mutation}_{popsize}_{maxiter}",
            "occ_frac": occ_frac,
            "run_fn": run_de,
            "npatch_h": npatch_h,
            "npatch_w": npatch_w,
            "maxiter": maxiter,
            "popsize": popsize,
        }
        cfgs.append(cfg)

        # npatch_occ = round((npatch_h * npatch_w) * occ_frac)
        # oa_de = attack.OccAttacker(
        #     attack_type='diff_ev_grad_cam', model_name=model_name,
        #     npatch_occ=npatch_occ, npatch_h=npatch_h, npatch_w=npatch_w,
        #     maxiter=maxiter, popsize=popsize, diff_ev_obj='prob_diff',
        #     mutation=mutation, verbose=False,
        # )
        # run_de = functools.partial(run_oa, oa=oa_de)
        # cfg = {
        #     'oa_name': f'diff_ev_grad_cam_{npatch_h}_{mutation}_{popsize}_{maxiter}',
        #     'occ_frac': occ_frac,
        #     'run_fn': run_de,
        #
        #     'npatch_h': npatch_h,
        #     'npatch_w': npatch_w,
        #     'maxiter': maxiter,
        #     'popsize': popsize,
        # }
        # cfgs.append(cfg)

    npatch_h_gc, npatch_w_gc = 224, 224
    for occ_frac in occ_fracs_unique:
        npatch_occ_gc = round((npatch_h_gc * npatch_w_gc) * occ_frac)
        oa_gc = attack.OccAttacker(
            attack_type="grad_cam",
            model_name=model_name,
            npatch_occ=npatch_occ_gc,
            npatch_h=npatch_h_gc,
            npatch_w=npatch_w_gc,
            verbose=False,
        )
        run_gc = functools.partial(run_oa, oa=oa_gc)
        cfg = {
            "oa_name": "grad_cam",
            "occ_frac": occ_frac,
            "run_fn": run_gc,
            "npatch_h": npatch_h_gc,
            "npatch_w": npatch_h_gc,
        }
        cfgs.append(cfg)

    # run attacks
    dummy_oa = attack.OccAttacker()
    for j, cfg in enumerate(cfgs):
        if j != attack_idx:
            continue
        flush_print(f"\nAttack {j + 1} of {len(cfgs)}.")
        run_one_attack(
            cfg=cfg,
            im_paths=im_paths,
            model=model,
            out_path=out_path,
            dummy_oa=dummy_oa,
            im_idx=im_idx,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--attack-idx", type=int, required=True)
    p.add_argument("--im-idx", type=int, required=True)
    p.add_argument("--exp-name", type=str, required=True)
    args = p.parse_args()

    flush_print(f"Running experiment with index {args.attack_idx} on gpu {args.gpu}.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    main(attack_idx=args.attack_idx, im_idx=args.im_idx, exp_name=args.exp_name)
