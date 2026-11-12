from __future__ import annotations
from typing import Any, Union

import numpy as np
from pycocotools import mask as maskUtils
import torch


def segToRLE(seg: Union[list[list[float]], dict[str, Any]], h: int, w: int) -> dict[str, Any]:
    """
    Convert segmentation of arbitary format into
    RLE format. Doesn't work for masks.
    """
    if type(seg) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif type(seg["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        # rle
        rle = seg
    return rle


def segToMask(
    seg: Union[np.ndarray, torch.Tensor, list[list[float]], dict[str, Any]], h: int, w: int
) -> np.ndarray:
    """
    Convert segmentation of arbitary format into
    mask format. Return 2D numpy array.
    """
    if type(seg) == np.ndarray:
        # if already in mask form, don't touch
        assert seg.shape == (h, w)
        m = seg
    elif type(seg) == torch.Tensor:
        # convert to numpy first
        seg = seg.cpu().numpy()
        assert seg.shape == (h, w)
        m = seg
    else:
        # conver to rle, and then decode to mask
        rle = segToRLE(seg, h, w)
        m = maskUtils.decode(rle)
    return m


def maskToRLE(mask: np.ndarray) -> dict[str, Any]:
    rle = maskUtils.encode(np.asarray(mask, order="F"))
    rle["counts"] = rle["counts"].decode("ascii")  # to ensure JSON serializability
    return rle


def RLEToMask(rle: dict[str, Any]) -> np.ndarray:
    m = maskUtils.decode(rle)
    return m
