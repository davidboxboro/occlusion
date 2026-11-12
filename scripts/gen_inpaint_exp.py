from pathlib import Path
import stat
import shutil

# params
T_sampling = 1000
eta = 0.9
slurm_machine = "manitoulin"
mit_machines = ["manitoulin", "coats"]
exp_name = "hexp1.1"
exp_path = Path("../data/human_occ_exp") / exp_name
subexp_paths = [e for e in exp_path.glob("*") if ("simplex_noise" in str(e) or "empty" in str(e))]

# make commands
cmds = []
for subexp_path in subexp_paths:
    subexp_name = subexp_path.name
    cmd = f"python ddnm_inpaint.py --exp-name {exp_name} --subexp-name {subexp_name} --eta {eta} --T_sampling {T_sampling}"
    cmds.append(cmd)

# slurm paths
slurm_dir = Path("slurm_scripts") / exp_name / "inpaint"
slurm_subdir = slurm_dir / "subscripts"
slurm_logdir = slurm_dir / "logs"
if slurm_dir.exists():
    shutil.rmtree(slurm_dir)
slurm_dir.mkdir(parents=True)
slurm_subdir.mkdir()
slurm_logdir.mkdir()

# write bash script for each command
subscript_paths = []
for idx, cmd in enumerate(cmds):
    if slurm_machine == "llsc":
        slurm_lines = [
            "#!/bin/bash\n",
            "#SBATCH --cpus-per-task 10",
            f"#SBATCH --gres gpu:volta:1",
            f"#SBATCH --output {slurm_logdir}/{idx}_{exp_name}.out\n",
            "source /etc/profile",
            "module load anaconda/2021b",
            "module load cuda/11.6\n",
        ]
    else:
        slurm_lines = [
            "#!/bin/zsh\n",
            "#SBATCH --cpus-per-task 10",
            f"#SBATCH --gres gpu:1",
            f"#SBATCH -w {slurm_machine}",
            "#SBATCH -p v2",
            f"#SBATCH --output {slurm_logdir}/{idx}_{exp_name}.out\n",
        ]

    # write subscript to storage
    subscript_path = slurm_subdir / f"{idx}_{exp_name}.sh"
    subscript_paths.append(subscript_path)
    with open(subscript_path, "w") as f:
        # write all header lines and cmd
        for line in slurm_lines:
            f.write(line + "\n")
        f.write(cmd + "\n\n")
    subscript_path.chmod(subscript_path.stat().st_mode | stat.S_IXUSR)

    # create main script to run all subscripts
    main_path = slurm_dir / "main.sh"
    with open(main_path, "w") as f:
        if slurm_machine in mit_machines:
            # activate occ env for infolab machines
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
