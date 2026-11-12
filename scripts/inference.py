"""
Inference script.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

from __future__ import annotations
from typing import Any, Optional


import argparse
import json
import logging
import os
import time
from contextlib import suppress
from functools import partial
import hashlib
import shutil

import numpy as np
import pandas as pd
import torch

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs
from timm.data import occ_transforms as occT
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def parse_args(raw_args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse and return args.
    """
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Inference inference")
    parser.add_argument(
        "data",
        nargs="?",
        metavar="DIR",
        const=None,
        help="path to dataset (*deprecated*, use --data-dir)",
    )
    parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        default="",
        help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)',
    )
    parser.add_argument(
        "--split", metavar="NAME", default="validation", help="dataset split (default: validation)"
    )
    parser.add_argument(
        "--model",
        "-m",
        metavar="MODEL",
        default="resnet50",
        help="model architecture (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--img-size",
        default=None,
        type=int,
        metavar="N",
        help="Input image dimension, uses model default if empty",
    )
    parser.add_argument(
        "--in-chans",
        type=int,
        default=None,
        metavar="N",
        help="Image input channels (default: None => 3)",
    )
    parser.add_argument(
        "--input-size",
        default=None,
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
    )
    parser.add_argument(
        "--use-train-size",
        action="store_true",
        default=False,
        help="force use of train input size, even when test size is specified in pretrained cfg",
    )

    parser.add_argument(
        "--crop-pct", default=None, type=float, metavar="N", help="Input image center crop pct"
    )
    parser.add_argument(
        "--crop-mode",
        default=None,
        type=str,
        metavar="N",
        help="Input image crop mode (squash, border, center). Model default if None.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override mean pixel value of dataset",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override std deviation of of dataset",
    )
    parser.add_argument(
        "--interpolation",
        default="",
        type=str,
        metavar="NAME",
        help="Image resize interpolation type (overrides model)",
    )
    parser.add_argument("--num-classes", type=int, default=1000, help="Number classes in dataset")
    parser.add_argument(
        "--class-map",
        default="",
        type=str,
        metavar="FILENAME",
        help='path to class to idx mapping file (default: "")',
    )
    parser.add_argument(
        "--log-freq",
        default=10,
        type=int,
        metavar="N",
        help="batch logging frequency (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="Use channels_last memory layout",
    )
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="use Native AMP for mixed precision training",
    )
    parser.add_argument(
        "--amp-dtype",
        default="float16",
        type=str,
        help="lower precision AMP dtype (default: float16)",
    )
    parser.add_argument(
        "--fuser",
        default="",
        type=str,
        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
    )
    parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument(
        "--no-prefetcher", action="store_true", default=False, help="disable fast prefetcher"
    )

    scripting_group = parser.add_mutually_exclusive_group()
    scripting_group.add_argument(
        "--torchscript", default=False, action="store_true", help="torch.jit.script the full model"
    )
    scripting_group.add_argument(
        "--torchcompile",
        nargs="?",
        type=str,
        default=None,
        const="inductor",
        help="Enable compilation w/ specified backend (default: inductor).",
    )
    scripting_group.add_argument(
        "--aot-autograd", default=False, action="store_true", help="Enable AOT Autograd support."
    )

    parser.add_argument(
        "--results-separate-col",
        action="store_true",
        default=False,
        help="separate output columns per result index.",
    )
    parser.add_argument("--topk", default=1, type=int, metavar="N", help="Top-k to output to CSV")
    parser.add_argument(
        "--fullname",
        action="store_true",
        default=False,
        help="use full sample name in output (not just basename).",
    )
    parser.add_argument(
        "--filename-col", default="filename", help="name for filename / sample name column"
    )
    parser.add_argument("--index-col", default="pd_idx", help="name for output indices column(s)")
    parser.add_argument("--output-col", default=None, help="name for logit/probs output column(s)")
    parser.add_argument(
        "--output-type",
        default="prob",
        help='output type colum ("prob" for probabilities, "logit" for raw logits)',
    )
    parser.add_argument(
        "--exclude-output",
        action="store_true",
        default=False,
        help="exclude logits/probs from results, just indices. topk must be set !=0.",
    )

    # new args (not in original script)
    parser.add_argument("--results-dir", type=str, default=None, help="folder for output results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="pred.csv",
        help="results filename (relative to results-dir)",
    )
    parser.add_argument(
        "--cfg-file", type=str, default="cfg.json", help="cfg filename (relative to results-dir)"
    )
    parser.add_argument(
        "--stat-file", type=str, default="stat.json", help="stat filename (relative to results-dir)"
    )
    parser.add_argument("--gt-loc", help="path to csv of ground truth labels")
    parser.add_argument(
        "--delete-old",
        action="store_true",
        default=False,
        help="delete old experiment results, if they exist (default: False)",
    )
    parser.add_argument(
        "--all-class-eval",
        action="store_true",
        default=False,
        help="keep predicted classes not in ground truth set when evaluating accuracy (default: False)",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        default=False,
        help="do not resize & crop images; assumes input images have already been resized & cropped",
    )

    # occlusion transform
    group = parser.add_argument_group("Occlusion transform parameters")
    group.add_argument(
        "--occ-mean-sub",
        action="store_true",
        default=False,
        help="perform mean subtraction on occluded pixels as well",
    )
    group.add_argument(
        "--occ-transform", type=str, default="", help="occlusion transform type (default: )"
    )
    group.add_argument(
        "--mask-path",
        type=str,
        help="path to json containing dictionary mapping image names to masks.",
    )
    group.add_argument(
        "--npatch-h", type=int, help="number of patches in height direction for patch occlusion"
    )
    group.add_argument(
        "--npatch-w", type=int, help="number of patches in width direction for patch occlusion"
    )
    group.add_argument(
        "--npatch-occ", type=int, help="number of occluded patches for patch occlusion"
    )
    group.add_argument("--nbar", type=int, help="number of bars for bar occlusion")
    group.add_argument("--nbar-occ", type=int, help="number of occluded bars for bar occlusion")
    group.add_argument(
        "--noise-freq", type=int, nargs="+", help="frequency used to generate simplex noise"
    )
    group.add_argument(
        "--npix-occ",
        type=int,
        nargs="+",
        help="number of occluded pixels in simplex noise occlusion",
    )

    # inpainting
    parser.add_argument(
        "--inpaint-model-precomp",
        type=str,
        default=None,
        help="name of inpainting model used in precomputed images",
    )
    parser.add_argument(
        "--inpaint-model", type=str, default=None, help="inpaint model architecture (default: None)"
    )

    # parse and return args
    args = parser.parse_args(args=raw_args)
    return args


class InferenceEngine:
    """
    Performs inference experiments.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialization.
        """
        # logger
        self.logger = logging.getLogger("inference")
        setup_default_logging()

        # parser args
        self.args = args

        # run on specified gpu
        if args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def flatten(lst: list[Any]) -> Any:
        """
        If list of size one, return first and only
        element of list.
        """
        assert isinstance(lst, list)
        if len(lst) == 1:
            return lst[0]
        return lst

    def setup(self) -> bool:
        """
        Create results dir.
        Returns False if results already exist, True otherwise.
        """
        args = self.args

        # create cfg for experiment
        self.cfg = {
            "model": args.model,
            "occ_mean_sub": args.occ_mean_sub,
            "inpaint_model_precomp": args.inpaint_model_precomp,
            "data_dir": args.data_dir,
        }
        assert args.pretrained or args.checkpoint
        # basically args.checkpoint overrides args.pretrained
        if args.checkpoint:
            self.cfg["checkpoint"] = args.checkpoint
        else:
            self.cfg["checkpoint"] = "pretrained"
        # transform
        if args.occ_transform:
            assert args.occ_transform in ["patch", "bar", "simplex_noise", "precomputed"]
            self.cfg["occ_transform"] = args.occ_transform
            if args.occ_transform == "patch":
                self.cfg.update(
                    {
                        "npatch_h": args.npatch_h,
                        "npatch_w": args.npatch_w,
                        "npatch_occ": args.npatch_occ,
                    }
                )
            elif args.occ_transform == "bar":
                self.cfg.update(
                    {
                        "nbar": args.nbar,
                        "nbar_occ": args.nbar_occ,
                    }
                )
            elif args.occ_transform == "simplex_noise":
                self.cfg.update(
                    {
                        "noise_freq": InferenceEngine.flatten(args.noise_freq),
                        "npix_occ": InferenceEngine.flatten(args.npix_occ),
                    }
                )
            elif args.occ_transform == "precomputed":
                self.cfg.update(
                    {
                        "mask_path": args.mask_path,
                    }
                )
        # inpaint
        if args.inpaint_model:
            self.cfg["inpaint_model"] = args.inpaint_model
        self.logger.info(self.cfg)

        # get cfg hash
        cfg_str = json.dumps(self.cfg, sort_keys=True)
        cfg_bytes = cfg_str.encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

        # add experiment name to results dir
        args.results_dir = os.path.join(args.results_dir, cfg_hash)
        self.logger.info(f"Results dir: {args.results_dir}")

        # create results dir
        stat_path = os.path.join(args.results_dir, args.stat_file)
        if os.path.exists(stat_path) and not args.delete_old:
            # return False if experiment results already exist
            return False

        if os.path.exists(args.results_dir):
            shutil.rmtree(args.results_dir)
        os.makedirs(args.results_dir)
        return True

    def create_model(self) -> None:
        """
        Sets some config variables and creates model. Sets value of self.model,
        self.data_config, self.amp_autocast, self.device.
        """
        args = self.args

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.device = torch.device(args.device)

        # number of channels
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        elif args.input_size is not None:
            in_chans = args.input_size[0]

        # create model
        if args.model.split("_")[0] == "mae":
            # mae
            self.model_repo = "mae"

            from pathlib import Path
            import sys

            mae_path = Path.cwd().parent / "src" / "mae"
            sys.path.append(str(mae_path))

            import models_vit

            vit_model = "_".join(args.model.split("_")[1:])
            model = models_vit.__dict__[vit_model](num_classes=args.num_classes, global_pool=True)

            if args.checkpoint == "":
                assert args.pretrained == True
                checkpoint_path_dict = {
                    "mae_vit_base_patch16": "mae_finetuned_vit_base.pth",
                    "mae_vit_large_patch16": "mae_finetuned_vit_large.pth",
                    "mae_vit_huge_patch14": "mae_finetuned_vit_huge.pth",
                }
                checkpoint_path = str(mae_path / "checkpoints" / checkpoint_path_dict[args.model])
            else:
                checkpoint_path = args.checkpoint
            self.logger.info(f"MAE checkpoint path: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            msg = model.load_state_dict(checkpoint)

            data_config = {
                "mean": IMAGENET_DEFAULT_MEAN,
                "std": IMAGENET_DEFAULT_STD,
            }
        else:
            # timm
            self.model_repo = "timm"

            args.pretrained = args.pretrained or not args.checkpoint
            model = create_model(
                args.model,
                num_classes=args.num_classes,
                in_chans=in_chans,
                pretrained=args.pretrained,
                checkpoint_path=args.checkpoint,
                **args.model_kwargs,
            )
            self.logger.info(
                f"timm model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}"
            )

        # resolving values like input size, crop, etc.
        data_config = resolve_data_config(vars(args), model=model)
        self.logger.info(data_config)

        # modify model format based on args
        model = model.to(self.device)
        model.eval()

        # data parallelism across gpus
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

        self.data_config = data_config
        self.model = model

    def create_loader(self) -> None:
        """
        Create data loader based on args.
        """
        args = self.args

        # create dataset
        pass_fn = args.occ_transform == "precomputed"
        root_dir = args.data or args.data_dir
        dataset = create_dataset(
            root=root_dir,
            name=args.dataset,
            split=args.split,
            class_map=args.class_map,
            pass_fn=pass_fn,
        )

        # create inpaint transform
        if args.inpaint_model:
            inpaint_transform = occT.InpaintTransform(
                args.inpaint_model,
            )
        else:
            inpaint_transform = None

        # create occlusion transform
        if args.occ_transform:
            # in the raw image, occluded pixels will be black if mean subtraction is
            # not specified; otherwise, they will be the mean color used in the model,
            # which is usually a greyish shade
            mean_color = tuple(round(255 * e) for e in self.data_config["mean"])
            if args.occ_mean_sub:
                color = (0, 0, 0)
            else:
                # mean subtraction comes after occlusion transform, so we add
                # mean in occlusion transform if we wish to avoid mean subtraction
                # of occluded pixels
                color = mean_color
            # create transform
            ret_mask = inpaint_transform is not None
            if args.occ_transform == "patch":
                occ_transform = occT.PatchOccTransform(
                    npatch_h=args.npatch_h,
                    npatch_w=args.npatch_w,
                    npatch_occ=args.npatch_occ,
                    color=color,
                    ret_mask=ret_mask,
                )
            elif args.occ_transform == "bar":
                occ_transform = occT.BarOccTransform(
                    nbar=args.nbar,
                    nbar_occ=args.nbar_occ,
                    color=color,
                    ret_mask=ret_mask,
                )
            elif args.occ_transform == "simplex_noise":
                occ_transform = occT.SimplexNoiseOccTransform(
                    noise_freq=args.noise_freq,
                    npix_occ=args.npix_occ,
                    color=color,
                    ret_mask=ret_mask,
                )
            elif args.occ_transform == "precomputed":
                occ_transform = occT.PrecomputedMaskOccTransform(
                    mask_path=args.mask_path,
                    color=color,
                    ret_mask=ret_mask,
                )
            else:
                raise ValueError()
        else:
            occ_transform = None

        # create loader
        args.prefetcher = not args.no_prefetcher
        workers = 1 if "tfds" in args.dataset or "wds" in args.dataset else args.workers
        self.loader = create_loader(
            dataset,
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            num_workers=workers,
            **self.data_config,
            occ_transform=occ_transform,
            inpaint_transform=inpaint_transform,
            skip_crop=args.skip_crop,
        )

    def save_results(self) -> None:
        """
        Save results, cfg, stat files.
        """
        args = self.args

        # get full paths
        results_path = os.path.join(args.results_dir, args.results_file)
        cfg_path = os.path.join(args.results_dir, args.cfg_file)
        stat_path = os.path.join(args.results_dir, args.stat_file)

        # save
        self.pd_df.to_csv(results_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)
        with open(stat_path, "w") as f:
            json.dump(self.stat, f)
        self.logger.info("Saved results.")

    def run_inference(self) -> None:
        """
        Run main inference loop and compute results and stats.
        """
        args = self.args

        # read ground truth DF (where each row corresponds to an image)
        gt_df = pd.read_csv(args.gt_loc, index_col="filename")
        # change filenames to original JPEG name if png
        filenames = self.loader.dataset.filenames(basename=not args.fullname)
        new_filenames = []
        for fn in filenames:
            suff = fn.split(".")[-1]
            if suff == "png":
                new_fn = "_".join(fn.split("_")[:3]) + ".JPEG"
            elif suff == "JPEG":
                new_fn = fn
            else:
                raise Exception(f"Invalid image suffix: {suff}")
            new_filenames.append(new_fn)
        filenames = new_filenames
        # only keep subset of ground truth rows that correspond to validation images
        if len(filenames) != len(gt_df):
            self.logger.warning("Ground truth and prediction DFs have different lengths")
        gt_df = gt_df.loc[filenames]

        # make mask of valid indices (i.e., indices in ground truth)
        # will be used if invalid indices are excluded from model output
        valid_idxs = gt_df.gt_idx.unique()
        valid_idx_mask = np.full(shape=args.num_classes, dtype=bool, fill_value=False)
        valid_idx_mask[valid_idxs] = True

        # main inference loop
        inference_start = time.time()
        top_k = min(args.topk, args.num_classes)
        batch_time = AverageMeter()
        end = time.time()
        all_indices = []
        all_outputs = []
        use_probs = args.output_type == "prob"
        self.model.eval()  # should have been called previously, but just in case
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.loader):
                if not args.prefetcher:
                    input = input.to(self.device)

                output = self.model(input)

                if use_probs:
                    output = output.softmax(-1)

                if top_k:
                    if not args.all_class_eval:
                        # first get rid of indices that don't appear in ground truth
                        # setting invalid class indices to negative probability
                        # ensures class won't get picked
                        output[:, ~valid_idx_mask] = -1

                    output, indices = output.topk(top_k)
                    all_indices.append(indices.cpu().numpy())

                all_outputs.append(output.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    self.logger.info(
                        "Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                            batch_idx, len(self.loader), batch_time=batch_time
                        )
                    )

        # total inference time
        inference_end = time.time()
        inference_min_elapsed = (inference_end - inference_start) / 60
        inference_min_elapsed = round(inference_min_elapsed, 3)

        # collect inference results into one DF
        all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
        all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)

        output_col = args.output_col or ("prob" if use_probs else "logit")
        data_dict = {args.filename_col: filenames}
        if args.results_separate_col and all_outputs.shape[-1] > 1:
            if all_indices is not None:
                for i in range(all_indices.shape[-1]):
                    data_dict[f"{args.index_col}_{i}"] = all_indices[:, i]
            for i in range(all_outputs.shape[-1]):
                data_dict[f"{output_col}_{i}"] = all_outputs[:, i]
        else:
            if all_indices is not None:
                if all_indices.shape[-1] == 1:
                    all_indices = all_indices.squeeze(-1)
                data_dict[args.index_col] = list(all_indices)
            if all_outputs.shape[-1] == 1:
                all_outputs = all_outputs.squeeze(-1)
            data_dict[output_col] = list(all_outputs)

        pd_df = pd.DataFrame(data=data_dict)
        pd_df = pd_df.set_index(args.filename_col)

        # Align predictions by filename before computing top-1 accuracy.
        res_df = pd.DataFrame({"pred": pd_df[args.index_col], "true": gt_df["gt_idx"]})
        top1 = (res_df.pred == res_df.true).mean()

        # make stats
        num_param = sum([m.numel() for m in self.model.parameters()])  # may be useful in analysis
        self.stat = {
            "top1": top1,
            "inference_time": inference_min_elapsed,
            "num_param": num_param,
        }
        self.logger.info(self.stat)

        self.pd_df = pd_df

        self.logger.info("Finished inference and creating stats.")

    def run_exp(self) -> None:
        """
        Main inference function. Perform inference
        and save results.
        """
        # make new experiment, return if results already exit
        is_new_exp = self.setup()
        if not is_new_exp:
            self.logger.info("Experiment results already exist, exiting now.")
            return
        self.logger.info(f"Experiment results do not exist yet, continuing.")

        # create model and data config
        self.create_model()

        # create loader
        self.create_loader()

        # perform inference
        self.run_inference()

        # save results
        self.save_results()

        self.logger.info("Experiment complete.")


def main() -> None:
    torch.multiprocessing.set_start_method("spawn")
    # start time
    start = time.time()
    # parse args
    args = parse_args()
    # run experiment
    ie = InferenceEngine(args)
    ie.run_exp()
    # stop time
    end = time.time()
    min_elapsed = (end - start) / 60
    print(f"Minutes elapsed: {min_elapsed:.3f}.\n\n")


if __name__ == "__main__":
    main()
