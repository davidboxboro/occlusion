from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import math
from pathlib import Path
from PIL import Image
import timm
from scipy.optimize import differential_evolution
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.util import view_as_blocks


class OccAttacker:
    """
    Occlusion attacker that makes images harder to classify by blacking out pixels.
    """

    def __init__(
        self,
        attack_type: str = "diff_ev",
        model_name: str = "resnet18",
        checkpoint: str = "",
        batch_size: int = 256,
        device: str = "cuda",
        c: int = 3,
        h: int = 224,
        w: int = 224,
        npatch_h: Optional[int] = None,
        npatch_w: Optional[int] = None,
        npatch_occ: Optional[int] = None,
        popsize: int = 16,
        maxiter: int = 1000,
        mutation: Union[float, tuple[float, float]] = (0.5, 1),
        tol: float = 0,
        diff_ev_obj: str = "prob_diff",
        workers: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Initialize.

        attack_type: options are 'diff_ev', 'grad_cam'
        model_name: string identifying model for timm to load
        checkpoint: checkpoint of model to load
        batch_size: number of images fed into model at once when performing attack
        device: typically 'cuda' unless GPU not used
        c: number of channels in images fed into attack
        h, w: height, width of images fed into attack
        npatch_h, npatch_w: number of patches in height and width directions
        npatch_occ: number of occluded patches
        popsize, maxiter, tol: params fed into scipy differential evolution algorithm
        diff_ev_obj: differential evolution objective, either 'prob_diff' or 'targ_prob'
        verbose: if true, with print messages
        """
        # model params
        assert attack_type in ["diff_ev", "grad_cam", "diff_ev_grad_cam"]
        self.attack_type = attack_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.checkpoint = checkpoint

        # image and occlusion params
        self.c = c
        self.h, self.w = h, w
        self.npatch_w = npatch_w if npatch_w is not None else w
        self.npatch_h = npatch_h if npatch_h is not None else h
        assert self.h % self.npatch_h == 0 and self.w % self.npatch_w == 0, (
            f"patches must be of same size, ({self.h}, {self.w}) must be divisible by ({self.npatch_h}, {self.npatch_w})"
        )
        self.patch_h = self.h // self.npatch_h
        self.patch_w = self.w // self.npatch_w
        self.npatch = self.npatch_h * self.npatch_w
        self.npatch_occ = npatch_occ if npatch_occ is not None else round(0.5 * self.npatch)
        assert self.npatch_occ <= self.npatch
        self.occ_frac = self.npatch_occ / self.npatch
        if verbose:
            print(f"Height: {self.h} | Width: {self.w} | Channels: {self.c}")
            print(
                f"Num patches: {self.npatch} = {self.npatch_h}*{self.npatch_w} | Num occluded patches: {self.npatch_occ} | Occlusion frac: {self.occ_frac:.6f}"
            )

        # differential evolution params
        self.workers = workers
        self.popsize = popsize
        self.maxiter = maxiter
        self.mutation = mutation
        self.tol = tol
        assert diff_ev_obj in ["targ_prob", "prob_diff"]
        self.diff_ev_obj = diff_ev_obj

        self.verbose = verbose

        # get model mean and std for normalization purposes
        mean = timm.get_pretrained_cfg_value(model_name, "mean")
        std = timm.get_pretrained_cfg_value(model_name, "std")
        assert mean is not None and std is not None
        self.mean = torch.tensor([round(255 * e) for e in mean])  # integer rounded
        self.std = torch.tensor([255 * e for e in std])  # not integer rounded

        # don't create model until it's needed
        self.model = None

    def to_model_fmt(self, im: np.ndarray) -> torch.Tensor:
        """
        Convert from 0-255 HWC nd.array image
        to normalized CHW torch.Tensor on device.
        """
        # normalize np image and convert to tensor
        assert isinstance(im, np.ndarray)
        assert im.shape == (self.h, self.w, self.c)
        im = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW
        im = (im - self.mean.view(self.c, 1, 1)) / self.std.view(self.c, 1, 1)  # CHW
        im = im.to(self.device)
        return im

    def from_model_fmt(self, im: torch.Tensor) -> np.ndarray:
        """
        Opposite of to_model_fmt().

        Convert to 0-255 HWC nd.array image
        from normalized CHW torch.Tensor on device.
        """
        assert isinstance(im, torch.Tensor)
        assert im.shape == (self.c, self.h, self.w)
        im = im.cpu()
        im = (im * self.std.view(self.c, 1, 1)) + self.mean.view(self.c, 1, 1)
        im = im.permute(1, 2, 0)
        im = im.numpy().astype(np.uint8)
        return im

    def run_attack(self, im: torch.Tensor, gt_idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """
        Run occlusion attack on given image.

        im: image in model format (normalized CHW torch.Tensor on device)
        gt_idx: image class index
        """
        assert isinstance(im, torch.Tensor)
        assert im.shape == (self.c, self.h, self.w)
        # create model if not loaded yet
        if self.model is None:
            self.model = timm.create_model(
                self.model_name, pretrained=True, checkpoint_path=self.checkpoint
            )
            self.model = self.model.to(self.device)
            self.model.eval()

        if self.attack_type == "diff_ev":
            # run differential evolution optimization algorithm
            bounds = []
            bounds += [(0, self.npatch_h - 1)] * self.npatch_occ
            bounds += [(0, self.npatch_w - 1)] * self.npatch_occ
            res = differential_evolution(
                func=self.calc_attack_strength,
                args=(im, gt_idx),
                popsize=self.popsize,
                bounds=bounds,
                tol=self.tol,
                maxiter=self.maxiter,
                mutation=self.mutation,
                integrality=True,
                vectorized=True,
                updating="deferred",
                disp=self.verbose,
                workers=self.workers,
            )
            occ_patch_idxs_h = res.x[: self.npatch_occ].astype(int)
            occ_patch_idxs_w = res.x[self.npatch_occ :].astype(int)
            occ_im, occ_m = self.create_occ_im(
                im=im,
                occ_patch_idxs_h=occ_patch_idxs_h,
                occ_patch_idxs_w=occ_patch_idxs_w,
            )
            return occ_im, occ_m

        elif self.attack_type == "grad_cam":
            # faster grad cam gradient-based heuristic

            # only support certain models for now
            if self.model_name in ["resnet18", "resnet50"]:
                target_layers = [self.model.layer4[-1]]
            else:
                raise ValueError("Model not supported by CAM currently.")

            # we get cam for the true class because we want to make
            # that class more difficult
            targets = [ClassifierOutputTarget(gt_idx)]

            # generate cam
            with GradCAM(
                model=self.model, target_layers=target_layers, use_cuda="cuda" in self.device
            ) as cam_generator:
                cam = cam_generator(input_tensor=im.view(1, *im.shape), targets=targets)
                cam = cam[0, :]

            # get cam pixel attribution sum for each patch
            cam_patches = view_as_blocks(cam, (self.patch_h, self.patch_w))
            cam_patch_sums = cam_patches.sum(axis=(2, 3))
            assert cam_patch_sums.shape == (self.npatch_h, self.npatch_w)

            # get occluded patches
            occ_patch_idxs_flattened = np.argsort(cam_patch_sums.reshape(-1))[-self.npatch_occ :]
            occ_patch_idxs_h = occ_patch_idxs_flattened // self.npatch_w
            occ_patch_idxs_w = occ_patch_idxs_flattened % self.npatch_w

            # generate occluded image
            occ_im, occ_m = self.create_occ_im(
                im=im,
                occ_patch_idxs_h=occ_patch_idxs_h,
                occ_patch_idxs_w=occ_patch_idxs_w,
            )
            return occ_im, occ_m

        elif self.attack_type == "diff_ev_grad_cam":
            # only support certain models for now
            if self.model_name in ["resnet18", "resnet50"]:
                target_layers = [self.model.layer4[-1]]
            else:
                raise ValueError("Model not supported by CAM currently.")

            # we get cam for the true class because we want to make
            # that class more difficult
            targets = [ClassifierOutputTarget(gt_idx)]

            # generate cam
            with GradCAM(
                model=self.model, target_layers=target_layers, use_cuda="cuda" in self.device
            ) as cam_generator:
                cam = cam_generator(input_tensor=im.view(1, *im.shape), targets=targets)
                cam = cam[0, :]

            # get cam pixel attribution sum for each patch
            cam_patches = view_as_blocks(cam, (self.patch_h, self.patch_w))
            cam_patch_sums = cam_patches.sum(axis=(2, 3))
            assert cam_patch_sums.shape == (self.npatch_h, self.npatch_w)

            # get occluded patches
            occ_patch_idxs_flattened = np.argsort(cam_patch_sums.reshape(-1))[-self.npatch_occ :]
            occ_patch_idxs_h = occ_patch_idxs_flattened // self.npatch_w
            occ_patch_idxs_w = occ_patch_idxs_flattened % self.npatch_w
            x0 = np.concatenate([occ_patch_idxs_h, occ_patch_idxs_w])

            # run differential evolution optimization algorithm
            # using GC result as starting point
            bounds = []
            bounds += [(0, self.npatch_h - 1)] * self.npatch_occ
            bounds += [(0, self.npatch_w - 1)] * self.npatch_occ
            res = differential_evolution(
                func=self.calc_attack_strength,
                args=(im, gt_idx),
                popsize=self.popsize,
                bounds=bounds,
                tol=self.tol,
                maxiter=self.maxiter,
                integrality=True,
                vectorized=True,
                updating="deferred",
                disp=self.verbose,
                workers=self.workers,
                x0=x0,
            )
            occ_patch_idxs_h = res.x[: self.npatch_occ].astype(int)
            occ_patch_idxs_w = res.x[self.npatch_occ :].astype(int)
            occ_im, occ_m = self.create_occ_im(
                im=im,
                occ_patch_idxs_h=occ_patch_idxs_h,
                occ_patch_idxs_w=occ_patch_idxs_w,
            )
            return occ_im, occ_m

    def calc_attack_strength(
        self, occ_patch_idxs: np.ndarray, im: torch.Tensor, gt_idx: int
    ) -> np.ndarray:
        """
        Calculate attack strength of true label for each of nim input vectors. Lower means stronger attack.

        occ_patch_idxs: array-like of shape (2 * npatch_occ, nim). in each of nim columns,
                        npatch_occ height pixels come first, then npatch_occ width pixels come after
        im: image as normalized CHW torch.Tensor
        gt_idx: integer index corresponding to ground truth class of im
        """
        # params
        assert im.shape == (self.c, self.h, self.w)
        occ_patch_idxs = occ_patch_idxs.astype(int)
        assert occ_patch_idxs.shape[0] == 2 * self.npatch_occ
        nim = occ_patch_idxs.shape[1]
        nbatches = math.ceil(nim / self.batch_size)
        # eval mode
        self.model.eval()
        # make occluded images
        targ_probs = []  # predicted prob of target class
        max_nontarg_probs = []  # predicted prob of most likely class outside target class
        for batch_idx in range(nbatches):
            # make batch of occluded images
            occ_ims = []
            im_idx_lo = self.batch_size * batch_idx
            im_idx_hi = min(im_idx_lo + self.batch_size, nim)
            assert im_idx_lo < im_idx_hi  # batch must contain at least one image
            for im_idx in range(im_idx_lo, im_idx_hi):
                occ_patch_idxs_h = occ_patch_idxs[: self.npatch_occ, im_idx]
                occ_patch_idxs_w = occ_patch_idxs[self.npatch_occ :, im_idx]
                occ_im, _ = self.create_occ_im(
                    im=im,
                    occ_patch_idxs_h=occ_patch_idxs_h,
                    occ_patch_idxs_w=occ_patch_idxs_w,
                )
                occ_ims.append(occ_im)
            occ_ims = torch.stack(occ_ims, dim=0)
            # compute model probs for batch
            with torch.no_grad():
                output = self.model(occ_ims)
            batch_probs = output.softmax(dim=-1)
            batch_probs = batch_probs.cpu().numpy()
            batch_targ_probs = batch_probs[:, gt_idx].tolist()
            ncls = batch_probs.shape[1]
            batch_nontarg_probs = batch_probs[:, np.r_[0:gt_idx, gt_idx + 1 : ncls]]
            batch_max_nontarg_probs = batch_nontarg_probs.max(axis=1).tolist()
            targ_probs += batch_targ_probs
            max_nontarg_probs += batch_max_nontarg_probs
        # return as 1D array with length nim
        targ_probs = np.array(targ_probs)
        max_nontarg_probs = np.array(max_nontarg_probs)
        prob_diffs = targ_probs - max_nontarg_probs
        if self.diff_ev_obj == "targ_prob":
            return targ_probs
        elif self.diff_ev_obj == "prob_diff":
            return prob_diffs

    def create_occ_im(
        self, im: torch.Tensor, occ_patch_idxs_h: np.ndarray, occ_patch_idxs_w: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Create image with occluded pixels at given indices.
        Return occluded image without modifying original image.

        im: unoccluded image in model format
        occ_patch_idxs_h: row indices of occluded patches
        occ_patch_idxs_w: col indices of occluded patches
        return: occluded image in model format and boolean occlusion mask as np.array
        """
        assert isinstance(im, torch.Tensor)
        assert im.shape == (self.c, self.h, self.w)
        # preprocess
        occ_im = im.clone()
        # Expand patch indices into a boolean H x W pixel mask.
        occ_m = np.full(shape=(self.npatch_h, self.npatch_w), fill_value=False, dtype=bool)
        occ_m[occ_patch_idxs_h, occ_patch_idxs_w] = True
        # resize mask to h x w
        occ_m = np.repeat(occ_m, repeats=self.patch_h, axis=0)
        occ_m = np.repeat(occ_m, repeats=self.patch_w, axis=1)
        # zero out occluded pixels
        for i in range(self.c):
            occ_im[i, :, :][occ_m] = 0
        return occ_im, occ_m


def main() -> None:
    """
    Run attack on desired images using OccAttacker.
    """
    # params
    gt_path = Path("../data/imagenet/ann/gt.csv")
    imagenet_path = Path("../data/imagenet/val_cropped")

    # ground truth
    gt_df = pd.read_csv(gt_path, index_col=0)

    # load image
    fn = gt_df.index[0]
    gt_cat = gt_df.loc[fn].gt_cat
    gt_idx = gt_df.loc[fn].gt_idx
    im_path = imagenet_path / gt_cat / fn
    im = np.array(Image.open(im_path))

    # apply attack
    npatch_h, npatch_w = 16, 16
    npatch_occ = round((npatch_h * npatch_w) * 0.1)
    oa = OccAttacker(
        model_name="resnet18",
        npatch_h=npatch_h,
        npatch_w=npatch_w,
        npatch_occ=npatch_occ,
        maxiter=5,
    )
    res = oa.run_attack(im=oa.to_model_fmt(im), gt_idx=gt_idx)
    print(res)


if __name__ == "__main__":
    main()
