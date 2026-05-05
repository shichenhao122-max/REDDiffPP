# Reproducibility Notes

This repository is an overlay for InverseBench. It intentionally excludes datasets, pretrained checkpoints, full experiment logs, W&B state, and raw reconstruction tensors.

## Setup

1. Prepare an InverseBench checkout and install its environment.
2. Copy `src/algo/*.py` into `InverseBench/algo/`.
3. Copy `configs/algorithm/*.yaml` into `InverseBench/configs/algorithm/`.
4. Place pretrained checkpoints under the checkpoint path expected by the local InverseBench checkout.
5. Replace all `/path/to/...` placeholders in the commands below with local paths.

## Example Commands

FFHQ super-resolution:

```bash
python main.py \
  problem=ffhq256_sr_x8 \
  algorithm=reddiffpp-ffhq \
  pretrain=ffhq256 \
  problem.prior=/path/to/checkpoints/ffhq256.pt \
  problem.data.root=/path/to/ffhq256/test \
  problem.data.id_list=0-99 \
  wandb=False \
  seed=0 \
  deterministic=True \
  exp_name=reddiffpp_ffhq_sr_x8
```

MRI raw x4:

```bash
python main.py \
  problem=multi-coil-mri-knee-lmdb \
  algorithm=reddiffpp-mri-raw-x4 \
  pretrain=mri-knee-mvue \
  problem.prior=/path/to/checkpoints/MRI-knee.pt \
  problem.data.root=/path/to/knee_test_lmdb \
  problem.data.simulated_kspace=false \
  problem.model.acceleration_ratio=4 \
  problem.model.mask_seed=0 \
  wandb=False \
  seed=0 \
  deterministic=True \
  exp_name=reddiffpp_mri_raw_x4
```

MRI raw x8 with RSS initialization:

```bash
python main.py \
  problem=multi-coil-mri-knee-lmdb \
  algorithm=reddiffpp-mri-raw-x8-rssinit \
  pretrain=mri-knee-mvue \
  problem.prior=/path/to/checkpoints/MRI-knee.pt \
  problem.data.root=/path/to/knee_test_lmdb \
  problem.data.simulated_kspace=false \
  problem.model.acceleration_ratio=8 \
  problem.model.mask_seed=0 \
  wandb=False \
  seed=0 \
  deterministic=True \
  exp_name=reddiffpp_mri_raw_x8_rssinit
```

Black-hole imaging:

```bash
python main.py \
  problem=blackhole \
  algorithm=reddiffpp-blackhole \
  pretrain=blackhole \
  problem.prior=/path/to/checkpoints/blackhole.pt \
  problem.model.root=/path/to/blackhole/measure \
  problem.data.root=/path/to/blackhole/test \
  problem.data.id_list=0-99 \
  wandb=False \
  seed=0 \
  deterministic=True \
  exp_name=reddiffpp_blackhole
```

Inverse scattering:

```bash
python main.py \
  problem=inv-scatter \
  algorithm=reddiffpp-inv-scatter-360-2000 \
  pretrain=inv-scatter \
  problem.prior=/path/to/checkpoints/inv-scatter-5m.pt \
  problem.data.root=/path/to/inv-scatter/test \
  problem.data.id_list=0-99 \
  wandb=False \
  seed=0 \
  deterministic=True \
  exp_name=reddiffpp_inv_scatter
```

## Notes

- The package is designed for private review rather than public one-command reproduction.
- The selected configs are final tuned configs, while full sweeps and intermediate ablations are intentionally excluded.
- Datasets, checkpoints, W&B runs, logs, and raw reconstruction tensors are not included.
