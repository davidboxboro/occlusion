from __future__ import annotations
from pathlib import Path
from typing import Union

# Occlusion transforms

import numpy as np
import random
import pyfastnoisesimd as fns
import json
from pycocotools import mask as mask_utils
import torch
import torchvision.transforms as torchT
from PIL import Image

import sys


class OccTransform:
    """
    Base class for occlusion transforms.
    Contains functionalities shared
    across all occlusion transforms.

    Accepts and returns numpy CHW arrays.
    """

    def __init__(self) -> None:
        self.occ_m = None

    def get_3d_mask(self) -> np.ndarray:
        assert self.occ_m is not None
        occ_m_3d = np.stack([self.occ_m] * 3, axis=2)
        return occ_m_3d

    def get_im_occ_frac(self) -> float:
        assert isinstance(self.occ_m, np.ndarray)
        h, w = self.occ_m.shape
        im_occ_frac = self.occ_m.sum() / (h * w)
        return im_occ_frac


class PatchOccTransform(OccTransform):
    """
    Patch occlusion transform.
    """

    def __init__(
        self,
        npatch_h: int,
        npatch_w: int,
        npatch_occ: int,
        color: tuple[int, int, int],
        ret_mask: bool,
    ) -> None:
        self.npatch_h = npatch_h
        self.npatch_w = npatch_w
        self.npatch_occ = npatch_occ
        self.npatch = npatch_h * npatch_w
        self.color = color
        assert self.npatch_occ <= self.npatch
        self.type = "patch"
        self.ret_mask = ret_mask

    def __call__(self, np_img: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels
        h, w = np_img.shape[-2:]  # CHW
        assert h % self.npatch_h == 0 and w % self.npatch_w == 0, (
            f"patches must be of same size, ({h}, {w}) must be divisible by ({self.npatch_h}, {self.npatch_w})"
        )
        patch_h, patch_w = h // self.npatch_h, w // self.npatch_w

        # create occlusion mask
        if self.npatch_occ * 2 <= self.npatch:
            occ_patch_idxs = random.sample(range(self.npatch), self.npatch_occ)
            self.occ_m = np.full(
                shape=(self.npatch_h * self.npatch_w), fill_value=False, dtype=bool
            )
            self.occ_m[occ_patch_idxs] = True
        else:
            occ_patch_idxs = random.sample(range(self.npatch), self.npatch - self.npatch_occ)
            self.occ_m = np.full(shape=(self.npatch_h * self.npatch_w), fill_value=True, dtype=bool)
            self.occ_m[occ_patch_idxs] = False

        # faster method
        self.occ_m = self.occ_m.reshape(self.npatch_h, self.npatch_w)
        self.occ_m = np.repeat(self.occ_m, repeats=patch_h, axis=0)
        self.occ_m = np.repeat(self.occ_m, repeats=patch_w, axis=1)

        # occlude masked pixels
        for c in range(3):
            np_img[c, :, :][self.occ_m] = self.color[c]

        if self.ret_mask:
            return (np_img, self.occ_m)
        return np_img


class BarOccTransform(OccTransform):
    """
    Bar occlusion transform. Bars are horizontal,
    for now.
    """

    def __init__(
        self, nbar: int, nbar_occ: int, color: tuple[int, int, int], ret_mask: bool
    ) -> None:
        self.nbar = nbar
        self.nbar_occ = nbar_occ
        self.color = color
        assert self.nbar_occ <= self.nbar
        self.type = "bar"
        self.ret_mask = ret_mask

    def __call__(self, np_img: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Add horizontal bar to image.
        """
        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels
        h, w = np_img.shape[-2:]  # CHW
        assert h % self.nbar == 0, "height must be divisible by number of bars"
        bar_h = h // self.nbar

        # create occlusion mask
        if self.nbar_occ * 2 <= self.nbar:
            occ_bar_idxs = random.sample(range(self.nbar), self.nbar_occ)
            self.occ_m = np.full(shape=(self.nbar), fill_value=False, dtype=bool)
            self.occ_m[occ_bar_idxs] = True
        else:
            occ_bar_idxs = random.sample(range(self.nbar), self.nbar - self.nbar_occ)
            self.occ_m = np.full(shape=(self.nbar), fill_value=True, dtype=bool)
            self.occ_m[occ_bar_idxs] = False

        # faster method
        self.occ_m = self.occ_m.reshape(self.nbar, 1)
        self.occ_m = np.repeat(
            self.occ_m, repeats=bar_h, axis=0
        )  # bars have some thickness (height)
        self.occ_m = np.repeat(self.occ_m, repeats=w, axis=1)  # bars extend entire width of image

        # occlude masked pixels
        for c in range(3):
            np_img[c, :, :][self.occ_m] = self.color[c]

        if self.ret_mask:
            return (np_img, self.occ_m)
        return np_img


class SimplexNoiseOccTransform(OccTransform):
    """
    Simplex noise.
    """

    def __init__(
        self,
        noise_freq: Union[int, list[int]],
        npix_occ: Union[int, list[int]],
        color: tuple[int, int, int],
        ret_mask: bool,
    ) -> None:
        if isinstance(noise_freq, int):
            noise_freq = [noise_freq]
        if isinstance(npix_occ, int):
            npix_occ = [npix_occ]
        assert isinstance(noise_freq, list)
        assert isinstance(npix_occ, list)
        self.noise_freq_lst = noise_freq
        self.npix_occ_lst = npix_occ
        self.color = color
        self.type = "simplex_noise"
        self.ret_mask = ret_mask

    def __call__(self, np_img: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        # randomly choose noise frequency and number of occluded pixels
        noise_freq = np.random.choice(self.noise_freq_lst)
        npix_occ = np.random.choice(self.npix_occ_lst)

        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels
        h, w = np_img.shape[-2:]  # CHW

        # create simplex noise
        np.random.seed()  # makes sure that numpy itself is reseeded, so that it doesn't generate the same number
        seed = np.random.randint(2**31)
        gen = fns.Noise(seed=seed, numWorkers=1)
        gen.noiseType = fns.NoiseType.Simplex
        gen.frequency = noise_freq / h
        noise = gen.genAsGrid([h, w])

        # generate occlusion mask
        occ_idxs = np.argsort(noise.reshape(-1))[
            :npix_occ
        ]  # select npix_occ smallest values in generated noise to occlude
        self.occ_m = np.full(shape=(h * w,), fill_value=False, dtype=bool)
        self.occ_m[occ_idxs] = True
        self.occ_m = self.occ_m.reshape(h, w)

        # occlude masked pixels
        for c in range(3):
            np_img[c, :, :][self.occ_m] = self.color[c]

        if self.ret_mask:
            return (np_img, self.occ_m)
        return np_img


class PrecomputedMaskOccTransform(OccTransform):
    """
    Computed mask occlusion transform.
    """

    def __init__(
        self, mask_path: Union[str, Path], color: tuple[int, int, int], ret_mask: bool
    ) -> None:
        with open(mask_path, "r") as f:
            self.mask_dict = json.load(f)
        self.color = color
        self.type = "precomputed"
        self.ret_mask = ret_mask

    def __call__(
        self, np_img_and_fn: tuple[np.ndarray, str]
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Takes in tuple (np_img, img_filename).
        Requires filename to look up mask corresponding
        to image.
        """
        assert isinstance(np_img_and_fn, tuple)
        assert len(np_img_and_fn) == 2
        np_img, fn = np_img_and_fn

        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels

        occ_rle = self.mask_dict[fn]
        if occ_rle is None:  # means that mask is empty
            h, w = np_img.shape[-2:]  # CHW
            self.occ_m = np.full(shape=(h, w), fill_value=False, dtype=bool)  # empty mask
        else:
            self.occ_m = mask_utils.decode(occ_rle)
            self.occ_m = self.occ_m.astype(bool)

        # occlude masked pixels
        for c in range(3):
            np_img[c, :, :][self.occ_m] = self.color[c]

        if self.ret_mask:
            return (np_img, self.occ_m)
        return np_img


class EmptyOccTransform(OccTransform):
    """
    Does nothing but makes API consistent for unoccluded images
    (images with 0 occlusion fraction).
    """

    def __init__(self, ret_mask: bool) -> None:
        self.type = "empty"
        self.ret_mask = ret_mask

    def __call__(self, np_img: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels
        h, w = np_img.shape[-2:]  # CHW

        # empty mask
        self.occ_m = np.full(shape=(h, w), fill_value=False, dtype=bool)

        if self.ret_mask:
            return (np_img, self.occ_m)
        return np_img


class FullOccTransform(OccTransform):
    """
    Fully occludes images.
    """

    def __init__(self, color: tuple[int, int, int], ret_mask: bool) -> None:
        self.type = "full"
        self.color = color
        self.ret_mask = ret_mask

    def __call__(
        self, np_img: np.ndarray
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], bool]:
        np_img = np.copy(np_img)  # don't modify input image
        assert isinstance(np_img, np.ndarray), "input to occlusion transform must be np.ndarray"
        assert np_img.shape[0] == 3  # 3 channels
        h, w = np_img.shape[-2:]  # CHW

        # full mask
        self.occ_m = np.full(shape=(h, w), fill_value=True, dtype=bool)

        # apply mask
        for c in range(3):
            np_img[c, :, :][self.occ_m] = self.color[c]

        if self.ret_mask:
            return (np_img, self.occ_m)
        return self.ret_mask


class InpaintTransform:
    """
    Inpaints occluded images given image and mask.
    """

    def __init__(self, inpaint_model_name: str) -> None:
        # imports
        for deepfillv2_path in [
            "/storage/davidlu/projects/objectnet-oc/src/deepfillv2-pytorch",
            # '/home/gridsan/dmayo/dl/objectnet-oc/src/deepfillv2-pytorch',
        ]:
            sys.path.append(deepfillv2_path)
        from model.networks import Generator

        # self.inpaint_model_name = inpaint_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # create inpaint model
        if inpaint_model_name == "deepfillv2":
            checkpt_path = f"{deepfillv2_path}/pretrained/states_pt_places2.pth"
            inpaint_model = Generator(checkpoint=checkpt_path, return_flow=True).to(self.device)
            inpaint_model.eval()
        else:
            raise ValueError("Inpaint type unknown.")
        self.inpaint_model = inpaint_model

    def __call__(self, np_im_and_m: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Accepts tuple (CHW np 0-255 array, HW np boolean array).

        Returns CHW np 0-255 image.
        """
        # tuple of CHW np 0-255 image and 2D boolean mask
        assert isinstance(np_im_and_m, tuple)
        np_im, np_m = np_im_and_m

        # put image in right format
        torch_im = torchT.ToTensor()(np_im.transpose(1, 2, 0)).to(
            self.device
        )  # CHW 0-1 Tensor on device
        # put mask in right format
        np_m_3d = np.stack([np_m] * 3, axis=0)  # CHW boolean np array
        torch_m_3d = torch.Tensor(np_m_3d).to(self.device)  # CHW boolean Tensor on device

        # perform inpainting
        with torch.inference_mode():
            output = self.inpaint_model.infer(torch_im, torch_m_3d, return_vals=["inpainted"])
        np_inpainted_im_raw = output[0].transpose(2, 0, 1)  # CHW 0-255 np image
        # do this because unoccluded pixels may have changed
        np_inpainted_im = np_m_3d * np_inpainted_im_raw + (~np_m_3d) * np_im

        # CHW np 0-255 image
        return np_inpainted_im
