from __future__ import annotations

import sys
from pathlib import Path

deepfill_path = Path.cwd().parent / "src" / "deepfill-pytorch"
if str(deepfill_path) not in sys.path:
    sys.path.append(str(deepfill_path))

util_path = Path.cwd().parent / "src" / "util"
if str(util_path) not in sys.path:
    sys.path.append(str(util_path))

import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import shutil
import json
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as torchT
import matplotlib.pyplot as plt
import time

from model.networks import Generator
from utils.tools import (
    get_config,
    random_bbox,
    mask_image,
    is_image_file,
    default_loader,
    normalize,
    get_model_list,
)

from mask_util import RLEToMask


# device
device = "cuda"
device_ids = [0]


def inpaint_im(netG: torch.nn.Module, ims: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
    # [0, 1] to [-1, 1]
    ims = ims * 2.0 - 1.0

    # make masks have same number of dims as images
    ms = ms.unsqueeze(dim=1)

    # zero out occluded pixs
    ims = ims * (1.0 - ms)

    # to cuda
    ims = ims.to(device)
    ms = ms.to(device)

    # inpaint
    with torch.no_grad():
        x1s, x2s, offset_flows = netG(ims, ms)
    inp_ims = x2s.cpu()

    # [-1, 1] to [0, 1]
    inp_ims = (inp_ims + 1.0) / 2.0

    return inp_ims


def main(exp_name: str, subexp_name: str) -> None:
    # in paths
    exp_path = Path("../data/human_occ_exp") / exp_name
    subexp_path = exp_path / subexp_name
    unocc_ims_path = exp_path / "orig_cropped_images"
    mask_path = subexp_path / "masks.json"
    ims_path = subexp_path / "images"

    # get images
    im_paths = sorted(ims_path.glob("*.png"))

    # out paths
    out_ims_path = Path("../results") / exp_name / "inpaint" / "deepfill" / subexp_name
    if out_ims_path.exists():
        shutil.rmtree(out_ims_path)
    out_ims_path.mkdir(parents=True)

    # set random seed
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # get model config
    config_path = deepfill_path / "configs" / "config.yaml"
    config = get_config(config_path)

    # load model
    netG = Generator(config["netG"], use_cuda=True, device_ids=device_ids)
    checkpoint_path = deepfill_path / "checkpoints" / "gen_00430000.pt"
    netG.load_state_dict(torch.load(checkpoint_path))
    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)

    # get masks
    with open(mask_path, "r") as f:
        mask_dict = json.load(f)

    # iterate through images
    pad_size = 16
    batch_size = 128
    nbatch = np.ceil(len(im_paths) / batch_size).astype(int)
    for i in range(nbatch):
        print(f"\tBatch {i + 1} of {nbatch}")

        # collect images into batch
        unocc_np_ims = []
        np_ims = []
        np_ms = []
        torch_ims_pad = []
        torch_ms_pad = []
        for j, im_path in enumerate(im_paths[i * batch_size : (i + 1) * batch_size]):
            # get image
            im_name = im_path.name
            orig_im_name = "_".join(im_name.split("_")[:3]) + ".JPEG"
            im = Image.open(im_path)
            np_im = np.array(im).astype(np.uint8)
            unocc_im = Image.open(unocc_ims_path / orig_im_name)
            unocc_np_im = np.array(unocc_im)
            # get mask
            rle_m = mask_dict[orig_im_name]
            np_m = RLEToMask(rle_m).astype(bool)
            # convert to torch
            torch_im = torchT.ToTensor()(np_im).to(torch.float32)
            torch_m = torch.Tensor(np_m).to(torch.float32)
            # pad to 256
            torch_im_pad = F.pad(torch_im, pad=[pad_size] * 4, value=0)
            torch_m_pad = F.pad(
                torch_m, pad=[pad_size] * 4, value=1
            )  # padding pixels are considered occluded
            # add to batch
            unocc_np_ims.append(unocc_np_im)
            np_ims.append(np_im)
            np_ms.append(np_m)
            torch_ims_pad.append(torch_im_pad)
            torch_ms_pad.append(torch_m_pad)

        # inpaint image
        torch_ims_pad = torch.stack(torch_ims_pad, dim=0)
        torch_ms_pad = torch.stack(torch_ms_pad, dim=0)
        inp_torch_ims_pad = inpaint_im(netG, torch_ims_pad, torch_ms_pad)

        # process output
        for j, im_path in enumerate(im_paths[i * batch_size : (i + 1) * batch_size]):
            # get specific image from batch
            im_name = im_path.name
            unocc_np_im = unocc_np_ims[j]
            np_im = np_ims[j]
            np_m = np_ms[j]
            inp_torch_im_pad = inp_torch_ims_pad[j]
            # crop to 224
            inp_torch_im = inp_torch_im_pad[:, pad_size:-pad_size, pad_size:-pad_size]
            inp_np_im = (inp_torch_im * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
            np_m_3d = np.stack([np_m] * 3, axis=2)
            merged_np_im = (~np_m_3d) * np_im + np_m_3d * inp_np_im
            # save images
            merged_im = Image.fromarray(merged_np_im)
            merged_im.save(out_ims_path / orig_im_name)
            # display images
            if False:
                fig, axs = plt.subplots(1, 4, figsize=(12, 3))
                np_ims = [inp_np_im, merged_np_im, np_im, unocc_np_im]
                names = [
                    "inpainted (all pixels)",
                    "inpainted (occ. pixels only)",
                    "occluded",
                    "unoccluded",
                ]
                for ax, np_im_, name in zip(axs, np_ims, names):
                    ax.imshow(Image.fromarray(np_im_))
                    ax.axis("off")
                    ax.set_title(name)
                plt.show()


if __name__ == "__main__":
    exp_name = "hexp1.1"
    exp_path = Path("../data/human_occ_exp") / exp_name
    subexp_paths = exp_path.glob("*")
    subexp_paths = [e for e in subexp_paths if ("simplex_noise" in str(e) or "empty" in str(e))]
    for i, subexp_path in enumerate(subexp_paths):
        subexp_name = subexp_path.name
        print(f"Sub-exp. {i + 1} of {len(subexp_paths)}: {subexp_name}")
        start = time.time()
        main(exp_name=exp_name, subexp_name=subexp_name)
        end = time.time()
        time_elapsed = end - start
        print(f"Took {time_elapsed:.2f} seconds")
