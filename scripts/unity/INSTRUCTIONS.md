# Training TRM Base on the Unity HPC Cluster

Step-by-step guide for running the TinyLLM TRM training pipeline on Unity.

---

## Prerequisites

- You have a Unity account and can SSH into the cluster.
- You know your Unity username (e.g., `jdoe`).

---

## Step 0 — Connect to Unity

```bash
ssh <your-username>@unity.rc.umass.edu
```

You land on a **login node**. Login nodes are shared — never run heavy
computation here.  Everything heavy goes through SLURM job scripts.

---

## Step 1 — Upload your code

From your **local machine** (not on Unity), push your branch to GitHub
and clone on Unity:

```bash
# On Unity login node
cd ~
git clone https://github.com/<your-org>/TinyLLM.git
cd TinyLLM
git checkout Experimental-TRM      # or whichever branch has the TRM code
```

Or use `scp` / `rsync` / `sftp` to copy the repo folder directly.

---

## Step 2 — Create the virtual environment (one-time)

This installs all Python packages **inside your home directory** so it
doesn't affect other users on the cluster.

```bash
cd ~/TinyLLM
bash scripts/unity/setup_venv.sh
```

What this does:
1. `module load python/3.11.7` and `module load cuda/12.6`
2. Creates `~/venvs/tinyllm` with `python -m venv`
3. Installs PyTorch (CUDA 12.6), your `requirements.txt`, and the extra
   packages the training code needs (`wandb`, `coolname`, `pydantic`,
   `argdantic`, `huggingface_hub`).

This takes a few minutes. You only need to do it **once**.

> **Tip:** If `module load python/3.11.7` fails, run `module spider python`
> to see available versions and update the script accordingly.

---

## Step 3 — Set up Weights & Biases

The training script logs to W&B.  On the login node:

```bash
source ~/venvs/tinyllm/bin/activate
wandb login
# Paste your API key when prompted (get it from https://wandb.ai/authorize)
deactivate
```

If compute nodes **cannot reach the internet**, add this line to
`train_trm.sh` (it's already there, commented out):

```bash
export WANDB_MODE=offline
```

You can sync the offline run later with `wandb sync`.

---

## Step 4 — Prepare the Sudoku dataset

Submit the data-prep job:

```bash
cd ~/TinyLLM
mkdir -p logs
sbatch scripts/unity/prepare_data.sh
```

Check progress:

```bash
squeue --me                         # see your job in the queue
cat logs/prepare_data_<JOBID>.out   # read output once it starts
```

When finished, you should see files under `data/sudoku-extreme-full/train/`
and `data/sudoku-extreme-full/test/`.

---

## Step 5 — Launch training

```bash
cd ~/TinyLLM
sbatch scripts/unity/train_trm.sh
```

Monitor:

```bash
squeue --me                         # job status (PD=pending, R=running)
tail -f logs/train_trm_<JOBID>.out  # live output
```

### What the training job does

| Setting | Value |
|---------|-------|
| Partition | `gpu-preempt` (max 2 hours, preemptible) |
| GPU | 1x NVIDIA A100 |
| Memory | 32 GB RAM |
| Config | `trm_base/config_pretrain_unity.yml` |
| Batch size | 16 |
| Epochs | 100 (eval every 10) |
| Learning rate | 1e-4 with cosine warmup (2000 steps) |
| `torch.compile` | Enabled (A100 Ampere) |

Checkpoints are saved to `checkpoints/<project>/<run>/step_<N>.pt`.

---

## Step 6 — Monitor and manage jobs

```bash
squeue --me              # list your running/pending jobs
scancel <JOBID>          # cancel a job
sacct -j <JOBID>         # see resource usage after completion
```

To SSH into the compute node while your job runs (for `nvidia-smi`, etc.):

```bash
srun --jobid=<JOBID> --pty bash
nvidia-smi               # check GPU utilisation
```

---

## Common adjustments

### Need more than 2 hours?

Switch to the non-preemptible GPU partition (up to 48 hours):

```bash
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
```

For runs longer than 48 hours, also add:

```bash
#SBATCH --qos=long
```

### Want a different GPU?

Change the constraint (see available GPUs at
https://docs.unity.rc.umass.edu/documentation/tools/gpus/):

```bash
#SBATCH --constraint=l40s       # 48 GB VRAM, Ada Lovelace
#SBATCH --constraint=v100       # 16-32 GB VRAM, Volta
#SBATCH --constraint=2080ti     # 11 GB VRAM, Turing (cheaper/faster to get)
```

For the 2080ti, you may need to reduce `global_batch_size` or use
`forward_dtype: float32` since older GPUs have limited float16 support.

### Resume from checkpoint

Edit `config_pretrain_unity.yml` and add:

```yaml
load_checkpoint: 'checkpoints/<project>/<run>/step_<N>.pt'
```

### Multi-GPU training

Request multiple GPUs and use `torchrun`:

```bash
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2

torchrun --nproc_per_node=2 trm_base/pretrain.py --config trm_base/config_pretrain_unity.yml
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'puzzle_dataset'` | Make sure `PYTHONPATH` includes `trm_base/`. The SLURM scripts set this automatically. |
| `CUDA_ERROR_OUT_OF_MEMORY` | Reduce `global_batch_size` in the YAML config or pick a GPU with more VRAM. |
| Job stuck in `PD` (pending) forever | A100s are popular. Try `--constraint=l40s` or `--constraint=a100-40g` or a less busy partition. |
| `wandb` login fails on compute node | Use `WANDB_MODE=offline` and `wandb sync` afterwards. |
| `module load python/3.11.7` not found | Run `module spider python` and pick an available version, then update the scripts. |
