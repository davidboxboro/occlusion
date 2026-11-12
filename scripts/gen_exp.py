from __future__ import annotations

# Generate script with commands

import argparse
import json
import stat
from pathlib import Path
import shutil
import math
import time
import subprocess

sample_model_fams = [
    "VGG (IN1k)",
    "ResNet (IN1k)",
    "ConvNeXt (IN1k)",
    "ConvNeXt-v2 (IN1k)",
    "ViT (IN21k)",
    "MAE (IN1k)",
    "DeiT (IN1k)",
    "Swin (IN1k)",
    "Swin (IN21k)",
    "CLIP (WIT)",
    "ConvNeXt (IN21k)",
    "ConvNeXt-v2 (IN21k)",
    "BEiT (IN21k)",
    "BEiT-v2 (IN21k)",
    "DenseNet (IN1k)",
    "ResNeXt (IN1k)",
]


def mk_cmds_precomp(
    exp_name: str,
    ngpu: int,
) -> list[str]:
    """
    Make list of slurm commands to run.
    Occlusion is precomputed.
    """
    proj_dir = Path("..")
    print(f"project dir: {proj_dir}")

    # get models
    with open("models.json", "r") as f:
        models_dict = json.load(f)
    models = []
    for model_fam in models_dict:
        # if model_fam not in sample_model_fams:
        #     continue
        for model, _ in models_dict[model_fam]:
            models.append(model)
    print(f"Number of models: {len(models)}")

    # make commands
    results_dir = proj_dir / "results" / exp_name / "eval"
    exp_dir = proj_dir / "data/human_occ_exp" / exp_name
    subexp_dirs = [e for e in exp_dir.glob("*") if ("simplex_noise" in str(e) or "empty" in str(e))]
    inpaint_dir = proj_dir / "results" / exp_name / "inpaint"
    inpaint_models = ["interp"]  # 'ddnm_0.90_1000', 'ddnm_0.90_100', 'deepfill']
    occ_data_dir = exp_dir / "orig_cropped_images"
    gt_loc = proj_dir / "data/imagenet/ann/gt.csv"
    batch_size = 128
    w, h = 224, 224

    workers = 10
    # note: crop-mode, crop-pct, interpolation aren't actually used when skip-crop is set
    # but they are included for clarity
    base_cmd_lines = [
        "python inference.py",
        # dataset
        "--dataset imagenet",
        # paths
        f"--gt-loc {gt_loc}",
        f"--results-dir {results_dir}",
        f"--input-size 3 {w} {h}",
        f"--batch-size {batch_size}",
        f"--num-gpu {ngpu}",
        # all of these settings are for validation only
        # the last three are not necessary when skip-crop flag is used
        "--skip-crop",
        "--interpolation bilinear",
        "--crop-mode center",
        "--crop-pct 1.0",
        # both finetuned and pretrained models get this flag
        "--pretrained",
        # optimizations
        "--amp",
        f"--workers {workers}",
    ]

    # generate commands for each model
    cmds = []
    # for each model, and each list of masks, compute model performance
    for model in models:
        # make cmds for pretrained, finetuned, and inpaint model
        for subexp_dir in subexp_dirs:
            subexp_name = subexp_dir.name
            mask_path = subexp_dir / "masks.json"

            ## inpaint
            for inpaint_model in inpaint_models:
                inpaint_data_dir = inpaint_dir / inpaint_model / subexp_name
                cmd_lines = base_cmd_lines + [
                    # data
                    f"--data-dir {inpaint_data_dir}",
                    f"--inpaint-model-precomp {inpaint_model}",
                    # no occlusion because of inpainting
                    # model
                    f"--model {model}",
                ]
                cmd = " \\\n".join(cmd_lines)
                cmds.append(cmd)

    return cmds


def mk_cmds(
    exp_name: str,
    ngpu: int,
) -> list[str]:
    """
    Make list of slurm commands to run.
    """
    proj_dir = Path("..")
    print(f"project dir: {proj_dir}")

    # get models
    with open("models.json", "r") as f:
        models_dict = json.load(f)
    models = []
    for model_fam in sample_model_fams:
        for model, _ in models_dict[model_fam]:
            models.append(model)
    print(models)
    print(f"Number of models: {len(models)}")

    # make commands
    results_dir = proj_dir / "results" / exp_name / "eval"
    data_dir = proj_dir / "data/imagenet/val_cropped"
    gt_loc = proj_dir / "data/imagenet/ann/gt.csv"
    batch_size = 128
    w, h = 224, 224

    workers = 10
    # note: crop-mode, crop-pct, interpolation aren't actually used when skip-crop is set
    # but they are included for clarity
    base_cmd_lines = [
        "python inference.py",
        # dataset
        "--dataset imagenet",
        f"--data-dir {data_dir}",
        # paths
        f"--gt-loc {gt_loc}",
        f"--results-dir {results_dir}",
        f"--input-size 3 {w} {h}",
        f"--batch-size {batch_size}",
        f"--num-gpu {ngpu}",
        # all of these settings are for validation only
        # the last three are not necessary when skip-crop flag is used
        "--skip-crop",
        "--interpolation bilinear",
        "--crop-mode center",
        "--crop-pct 1.0",
        "--pretrained",
        # optimizations
        "--amp",
        f"--workers {workers}",
    ]

    # generate commands for each model
    cmds = []

    # finetuned eval
    occ_transform = "simplex_noise"
    ft_results_dir = proj_dir / "results" / exp_name / "ft"
    checkpoint_paths = [str(path) for path in ft_results_dir.glob("*/model_best.pth.tar")]
    assert len(checkpoint_paths) > 0  # make sure path is not misspecified
    noise_freqs = [2**i for i in range(0, 9)]
    npix_occs = [(w * h * i) // 8 for i in range(0, 8)]
    print("Finetuned model experiments:")
    for model in models:
        # look for a valid finetuned checkpoint corresponding to model
        for checkpoint_path in checkpoint_paths:
            ft_cfg_path = Path(checkpoint_path).parent / "cfg.json"
            if not ft_cfg_path.exists():
                continue
            with open(ft_cfg_path, "r") as f:
                ft_cfg = json.load(f)
            if not (
                "noise_freq" in ft_cfg
                and ft_cfg["noise_freq"] == noise_freqs
                and ft_cfg["npix_occ"] == npix_occs
                and ft_cfg["model"] == model
            ):
                continue
            model = ft_cfg["model"]
            print(f"\t{model}")
            for noise_freq in noise_freqs:
                for npix_occ in npix_occs:
                    cmd_lines = base_cmd_lines + [
                        f"--model {model}",
                        f"--occ-transform {occ_transform}",
                        f"--noise-freq {noise_freq}",
                        f"--npix-occ {npix_occ}",
                    ]
                    # don't use checkpoint flag for non-finetuned version
                    if checkpoint_path != "":
                        cmd_lines += [
                            f"--checkpoint {checkpoint_path}",
                        ]
                    cmd = " \\\n".join(cmd_lines)
                    cmds.append(cmd)

    return cmds


def mk_cmds_ft(
    exp_name: str,
    ngpu: int,
) -> list[str]:
    """
    Make list of slurm commands to run.
    """
    proj_dir = Path("..")
    print(f"project dir: {proj_dir}")

    # get models
    with open("models.json", "r") as f:
        models_dict = json.load(f)
    models = []
    model_fams = []
    for model_fam in sample_model_fams:
        for model, _ in models_dict[model_fam]:
            models.append(model)
            model_fams.append(model_fam)
    print(models)
    print(f"Number of models: {len(models)}")

    # make commands
    results_dir = proj_dir / "results" / exp_name / "ft"
    data_dir = proj_dir / "data/imagenet"
    train_split = "train"
    val_split = "val_cropped"
    gt_loc = proj_dir / "data/imagenet/ann/gt.csv"

    w, h = 224, 224
    batch_size = 128
    epochs = 20

    lr_base = 0.001
    decay_rate = 0.9
    decay_epochs = 1
    warmup_epochs = 0

    workers = 10

    # note: crop-mode, crop-pct, interpolation aren't actually used when skip-crop is set
    # but they are included for clarity
    base_cmd_lines = [
        "train.py",
        # dataset stuff
        "--dataset imagenet",
        f"--data-dir {data_dir}",
        f"--train-split {train_split}",
        f"--val-split {val_split}",
        # paths
        f"--gt-loc {gt_loc}",
        f"--results-dir {results_dir}",
        f"--input-size 3 {w} {h}",
        f"--batch-size {batch_size}",
        f"--epochs {epochs}",
        # lr scheduler
        "--sched step",
        f"--lr-base {lr_base}",
        f"--lr-base-size {batch_size}",
        "--lr-base-scale linear",
        f"--decay-rate {decay_rate}",
        f"--decay-epochs {decay_epochs}",
        f"--warmup-epochs {warmup_epochs}",
        # all of these settings are for validation only
        # the last three are not necessary when skip-crop flag is used
        "--skip-crop",
        "--interpolation bilinear",
        "--crop-mode center",
        "--crop-pct 1.0",
        "--pretrained",
        # optimizations
        "--amp",
        f"--workers {workers}",
    ]

    # generate commands for each model
    cmds = []
    for i, model in enumerate(models):
        # use either torchrun or python depending on
        # whether it's a distributed process
        assert ngpu >= 1
        if ngpu > 1:
            port = 4400 + i  # each DDP needs separate port
            init_cmd_lines = [
                "torchrun",
                f"--master_port {port}",
                f"--nproc_per_node {ngpu}",
            ]
        else:
            init_cmd_lines = [
                "python",
            ]

        # simplex noise occlusion, mult. frequencies and occlusion fracs
        occ_transform = "simplex_noise"
        noise_freqs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        npix_occs = [(w * h * i) // 8 for i in range(0, 8)]
        noise_freq_str = " ".join(map(str, noise_freqs))
        npix_occ_str = " ".join(map(str, npix_occs))

        cmd_lines = (
            init_cmd_lines
            + base_cmd_lines
            + [
                # model
                f"--model {model}",
                # occlusion params
                f"--occ-transform {occ_transform}",
                f"--noise-freq {noise_freq_str}",
                f"--npix-occ {npix_occ_str}",
            ]
        )

        cmd = " \\\n".join(cmd_lines)
        cmds.append(cmd)
        # end for

    return cmds


def main() -> None:
    """
    Generate slurm file to run experiment.
    """
    # params
    p = argparse.ArgumentParser()
    p.add_argument("--slurm-machine", type=str)
    p.add_argument("--exp-name", type=str)
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--free-gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--attack", action="store_true", default=False)
    p.add_argument("--ft", action="store_true", default=False)
    p.add_argument("--precomputed", action="store_true", default=False)
    p.add_argument("--no-slurm", action="store_true", default=False)
    args = p.parse_args()

    mit_machines = ["manitoulin", "somerset", "coats", "akimiski"]
    assert args.slurm_machine in ["llsc"] + mit_machines

    free_gpus = [int(e) for e in args.free_gpus.split(",")]

    # get commands
    if args.attack:
        cmds = []
        for attack_idx in range(50):
            for im_idx in range(5):
                cmds += [
                    f"python run_attack.py --attack-idx {attack_idx} --im-idx {im_idx} --exp-name {args.exp_name}"
                ]
    elif args.ft:
        cmds = mk_cmds_ft(args.exp_name, args.ngpu)
    else:
        if args.precomputed:
            cmds = mk_cmds_precomp(args.exp_name, args.ngpu)
        else:
            cmds = mk_cmds(args.exp_name, args.ngpu)

    if not args.no_slurm:
        # make slurm script dirs
        print(f"Generating slurm script for exp: {args.exp_name}")
        ft_or_eval = "ft" if args.ft else "eval"
        slurm_dir = Path("slurm_scripts") / args.exp_name / ft_or_eval
        slurm_logdir = slurm_dir / "logs"
        slurm_subdir = slurm_dir / "subscripts"
        if slurm_dir.exists():
            shutil.rmtree(slurm_dir)
        slurm_dir.mkdir(parents=True)
        slurm_subdir.mkdir()
        slurm_logdir.mkdir()
        subscript_paths = []

        # make scripts
        n = math.ceil(len(cmds) / 50)
        idx = 0
        for i in range(0, len(cmds), n):
            # create script for given n commands.
            # script header is different depending on
            # which machine to run on
            if args.slurm_machine == "llsc":
                slurm_lines = [
                    "#!/bin/bash\n",
                    "#SBATCH --cpus-per-task 20",
                    f"#SBATCH --gres gpu:volta:{args.ngpu}",
                    f"#SBATCH --ntasks {args.ngpu}",
                    f"#SBATCH --output {slurm_logdir}/{idx}_{args.exp_name}.out\n",
                    "source /etc/profile",
                    "module load anaconda/2022b",
                    "module load cuda/11.6\n",
                ]
            else:
                slurm_lines = [
                    "#!/bin/zsh\n",
                    "#SBATCH --cpus-per-task 10",
                    f"#SBATCH --gres gpu:{args.ngpu}",
                    f"#SBATCH --ntasks {args.ngpu}",
                    f"#SBATCH -w {args.slurm_machine}",
                    "#SBATCH -p v2",
                    f"#SBATCH --output {slurm_logdir}/{idx}_{args.exp_name}.out\n",
                ]
            # write subscript to storage
            subscript_path = slurm_subdir / f"{idx}_{args.exp_name}.sh"
            subscript_paths.append(subscript_path)
            with open(subscript_path, "w") as f:
                # write all header lines
                for line in slurm_lines:
                    f.write(line + "\n")
                # write all n commands
                cmd_subset = cmds[i : i + n]
                for cmd in cmd_subset:
                    f.write(cmd + "\n\n")
            subscript_path.chmod(subscript_path.stat().st_mode | stat.S_IXUSR)
            # increment
            idx += 1

        # create main script to run all subscripts
        main_path = slurm_dir / "main.sh"
        with open(main_path, "w") as f:
            # activate occ env for infolab machines
            if args.slurm_machine in mit_machines:
                lines = [
                    "#!/bin/zsh",
                    "source ~/.zshrc",
                    "conda activate occ",
                ]
                for line in lines:
                    f.write(line + "\n")
            for subscript_path in subscript_paths:
                f.write(f"sbatch {subscript_path}\n")
        main_path.chmod(main_path.stat().st_mode | stat.S_IXUSR)

    else:
        # don't use slurm, run directly

        # log dir
        ft_or_eval = "ft" if args.ft else "eval"
        nonslurm_log_dir = Path("nonslurm_logs") / args.exp_name / ft_or_eval
        if nonslurm_log_dir.exists():
            shutil.rmtree(nonslurm_log_dir)
        nonslurm_log_dir.mkdir(parents=True)

        ps_lst = []
        cmd_idx = 0
        while len(cmds) > 0:
            time.sleep(0.01)
            # remove finished processes and relinquish
            # control of those gpus
            for i, (ps, gpu) in enumerate(ps_lst):
                if ps.poll() is not None:
                    ps_lst.pop(i)
                    free_gpus.append(gpu)

            # start new cmd if there is a free gpu
            if len(free_gpus) > 0 and len(cmds) > 0:
                gpu = free_gpus.pop(0)
                cmd = cmds.pop(0)
                # start cmd on now-free gpu
                cmd += f" --gpu {gpu}"
                cmd = cmd.replace("\\", "")
                cmd = cmd.replace("\n", "")
                print(f"\nStarting command on gpu {gpu} ({len(cmds)} commands left)")
                print(f"\tCommand: {cmd}")
                with open(nonslurm_log_dir / f"{cmd_idx}_{args.exp_name}.out", "w") as f:
                    ps = subprocess.Popen(cmd.split(), stdout=f, stderr=f)
                ps_lst.append((ps, gpu))
                cmd_idx += 1


if __name__ == "__main__":
    main()
