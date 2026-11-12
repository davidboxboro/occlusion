from __future__ import annotations

import sys
import os
from pathlib import Path

ddnm_path = Path.cwd().parent / "src" / "DDNM"
if str(ddnm_path) not in sys.path:
    sys.path.append(str(ddnm_path))

util_path = Path.cwd().parent / "src" / "util"
if str(util_path) not in sys.path:
    sys.path.append(str(util_path))

from guided_diffusion.diffusion import (
    get_beta_schedule,
    get_schedule_jump,
    compute_alpha,
    get_gaussian_noisy_img,
)
from guided_diffusion.script_util import create_model
from datasets import data_transform, inverse_data_transform
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion
from functions.svd_operators import Inpainting
from main import dict2namespace
from mask_util import RLEToMask
import yaml
import torch
import json
import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as torchT
import matplotlib.pyplot as plt
import random
import pandas as pd
import shutil
import argparse


class DDNMInpainter:
    def __init__(self, eta: float, sigma_y: float, T_sampling: int) -> None:
        # get config
        with open(ddnm_path / "configs" / "imagenet_256.yml", "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)

        # change T_sampling
        config.time_travel.T_sampling = T_sampling

        # device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set up of some param values
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float().to(device)
        num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        assert config.model.var_type == "fixedsmall"
        logvar = posterior_variance.clamp(min=1e-20).log()

        # create model
        assert config.model.type == "openai"
        model_config_dict = vars(config.model)
        model = create_model(**model_config_dict)

        assert config.model.use_fp16
        model.convert_to_fp16()

        assert not config.model.class_cond
        ckpt_path = ddnm_path / "exp" / "logs" / "imagenet" / "256x256_diffusion_uncond.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        model = torch.nn.DataParallel(model)

        # member vars
        self.model = model
        self.config = config
        self.device = device
        self.sigma_y = sigma_y
        self.betas = betas
        self.eta = eta

    def inpaint_im_simp(self, torch_im: torch.Tensor, torch_occ_m: torch.Tensor) -> torch.Tensor:
        # member vars
        model = self.model
        config = self.config
        device = self.device
        sigma_y = self.sigma_y
        betas = self.betas
        eta = self.eta

        mask = 1 - torch_occ_m
        mask = mask.to(device)
        A = lambda z: z * mask
        Ap = A

        sigma_y *= 2  # to account for scaling to [-1, 1]

        subset_start = -1
        idx_init = subset_start
        idx_so_far = subset_start
        avg_psnr = 0.0

        x_orig = torch.unsqueeze(torch_im, 0)  # add dim 0
        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)

        y = A(x_orig)

        if config.sampling.batch_size != 1:
            raise ValueError("Please change the config file to set batch size to 1")

        Apy = Ap(y)

        # init x_T
        x = torch.randn(
            y.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )

        with torch.no_grad():
            skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
            n = x.size(0)
            x0_preds = []
            xs = [x]

            times = get_schedule_jump(
                config.time_travel.T_sampling,
                config.time_travel.travel_length,
                config.time_travel.travel_repeat,
            )
            time_pairs = list(zip(times[:-1], times[1:]))

            # reverse diffusion sampling
            for i, j in tqdm.tqdm(time_pairs):
                i, j = i * skip, j * skip
                if j < 0:
                    j = -1

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(device)
                    next_t = (torch.ones(n) * j).to(device)
                    at = compute_alpha(betas, t.long())
                    at_next = compute_alpha(betas, next_t.long())
                    sigma_t = (1 - at_next**2).sqrt()
                    xt = xs[-1].to(device)

                    et = model(xt, t)

                    if et.size(1) == 6:
                        et = et[:, :3]

                    # eq. 12
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # eq. 19
                    if sigma_t >= at_next * sigma_y:
                        lambda_t = 1.0
                        gamma_t = (sigma_t**2 - (at_next * sigma_y) ** 2).sqrt()
                    else:
                        lambda_t = (sigma_t) / (at_next * sigma_y)
                        gamma_t = 0.0

                    # eq. 17
                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta**2) ** 0.5)

                    # different from the paper, we use DDIM here instead of DDPM
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (
                        c1 * torch.randn_like(x0_t) + c2 * et
                    )

                    x0_preds.append(x0_t.to("cpu"))
                    xs.append(xt_next.to("cpu"))
                else:  # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to(device)

                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to("cpu"))

            x = xs[-1]

        x = [inverse_data_transform(config, xi) for xi in x]

        orig = inverse_data_transform(config, x_orig[0])
        mse = torch.mean((x[0].to(device) - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        avg_psnr += psnr

        idx_so_far += y.shape[0]

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        # print('Total Average PSNR: %.2f' % avg_psnr)
        # print('Number of samples: %d' % (idx_so_far - idx_init))

        inpainted_torch_im = x[0]
        return inpainted_torch_im

    def inpaint_im_svd(self, torch_im: torch.Tensor, torch_occ_m: torch.Tensor) -> torch.Tensor:
        # member vars
        model = self.model
        config = self.config
        device = self.device
        sigma_y = self.sigma_y
        betas = self.betas
        eta = self.eta

        # get degradation matrix
        mask = 1 - torch_occ_m
        mask = mask.to(device)
        mask = mask.reshape(-1)  # flatten mask
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)

        sigma_y *= 2  # to account for scaling to [-1, 1]

        subset_start = -1
        idx_init = subset_start
        idx_so_far = subset_start
        avg_psnr = 0.0

        x_orig = torch.unsqueeze(torch_im, 0)  # add dim 0
        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)

        y = A_funcs.A(x_orig)

        b, hwc = y.size()

        # if args.add_noise:  # for denoising test
        #    y = get_gaussian_noisy_img(y, sigma_y)

        y = y.reshape((b, hwc))

        Apy = A_funcs.A_pinv(y).view(
            y.shape[0], config.data.channels, config.data.image_size, config.data.image_size
        )

        Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

        # start DDIM
        x = torch.randn(
            y.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )

        with torch.no_grad():
            cls_fn = None
            if sigma_y == 0.0:  # noise-free case, turn to ddnm
                x, _ = ddnm_diffusion(x, model, betas, eta, A_funcs, y, config=config)
            else:  # noisy case, turn to ddnm+
                x, _ = ddnm_plus_diffusion(x, model, betas, eta, A_funcs, y, sigma_y, config=config)

        x = [inverse_data_transform(config, xi) for xi in x]
        x = x[0]

        orig = inverse_data_transform(config, x_orig[0])
        mse = torch.mean((x[0].to(device) - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        avg_psnr += psnr

        idx_so_far += y.shape[0]

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        # print('Total Average PSNR: %.2f' % avg_psnr)
        # print('Number of samples: %d' % (idx_so_far - idx_init))

        inpainted_torch_im = x[0]
        return inpainted_torch_im


def main(exp_name: str, subexp_name: str, eta: float, T_sampling: int) -> None:
    """
    Main.
    """
    sigma_y = 0

    # in paths
    exp_path = Path("../data/human_occ_exp") / exp_name
    subexp_path = exp_path / subexp_name
    mask_path = subexp_path / "masks.json"
    ims_path = subexp_path / "images"
    unocc_ims_path = exp_path / "orig_cropped_images"

    # out paths
    out_ims_path = (
        Path("../results") / exp_name / "inpaint" / f"ddnm_{eta:.2f}_{T_sampling}" / subexp_name
    )
    if not out_ims_path.exists():
        print(f"Out directory {out_ims_path} does not exist yet.", flush=True)
        out_ims_path.mkdir(parents=True)
    else:
        print(f"Out directory {out_ims_path} already exists.", flush=True)

    # create inpaint model
    inpaint_model = DDNMInpainter(eta=eta, sigma_y=sigma_y, T_sampling=T_sampling)

    # get masks
    with open(mask_path, "r") as f:
        mask_dict = json.load(f)

    # get images
    im_paths = sorted(ims_path.glob("*.png"))

    print("Loaded model, starting inpainting", flush=True)
    # iterate through images
    for i, im_path in enumerate(im_paths):
        print(f"Image {i + 1} of {len(im_paths)}", flush=True)
        # get image
        im_name = im_path.name
        if (out_ims_path / im_name).exists():
            print(f"\tSkipping image {i + 1} because inpainted image already exists", flush=True)
            continue
        orig_im_name = "_".join(im_name.split("_")[:3]) + ".JPEG"
        im = Image.open(im_path)
        np_im = np.array(im).astype(np.uint8)
        unocc_im = Image.open(unocc_ims_path / orig_im_name)
        unocc_np_im = np.array(unocc_im)
        # get mask
        rle_m = mask_dict[orig_im_name]
        np_m = RLEToMask(rle_m).astype(bool)
        np_m_3d = np.stack([np_m] * 3, axis=2)
        # convert to torch
        torch_im = torchT.ToTensor()(np_im).to(float)
        torch_m = torch.Tensor(np_m).to(float)
        # pad to 256
        pad_size = 16
        torch_im_pad = F.pad(torch_im, pad=[pad_size] * 4, value=0)
        torch_m_pad = F.pad(
            torch_m, pad=[pad_size] * 4, value=1
        )  # padding pixels are considered occluded
        # inpaint image
        inp_torch_im_pad = inpaint_model.inpaint_im_svd(torch_im_pad, torch_m_pad)
        # crop to 224
        inp_torch_im = inp_torch_im_pad[:, pad_size:-pad_size, pad_size:-pad_size]
        inp_np_im = (inp_torch_im * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        merged_np_im = (~np_m_3d) * np_im + np_m_3d * inp_np_im

        # save images
        merged_im = Image.fromarray(merged_np_im)
        merged_im.save(out_ims_path / im_name)

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
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--subexp-name", type=str)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--T_sampling", type=int)
    args = parser.parse_args()

    # run on specified gpu
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # inpaint all images
    main(
        exp_name=args.exp_name,
        subexp_name=args.subexp_name,
        eta=args.eta,
        T_sampling=args.T_sampling,
    )
