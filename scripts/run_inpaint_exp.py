from pathlib import Path
import stat
import shutil
import time
import subprocess

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
cmds = sorted(cmds)
# cmds.reverse()

# log dir
log_dir = Path.cwd() / "nonslurm_logs" / exp_name / "inpaint" / f"ddnm_{eta:.2f}_{T_sampling}"
if not log_dir.exists():
    log_dir.mkdir(parents=True)

free_gpus = [1, 2, 3, 4, 5, 6]
ps_lst = []
cmd_idx = 0
while len(cmds) > 0 or len(ps_lst) > 0:
    time.sleep(0.1)
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
        print(f"Starting command on gpu {gpu} ({len(cmds)} commands left)")
        with open(log_dir / f"{cmd_idx}_{exp_name}.out", "w") as f:
            ps = subprocess.Popen(cmd.split(), stdout=f, stderr=f)
        ps_lst.append((ps, gpu))
        cmd_idx += 1
