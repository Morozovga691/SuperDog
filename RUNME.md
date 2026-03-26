# RUNME

## Install

```bash
cd last_project/SuperDog

conda create --name superdog python=3.8
conda activate superdog

pip install -e .
```

Optional extras:

```bash
pip install -e ".[mjx]"
pip install -e ".[ros2]"
pip install onnxruntime
```

## Standard Training

```bash
python scripts/train.py configs/a1.yaml --train --headless
```

This starts SAC training with the default robot config, curriculum, reward shaping, and checkpoint/logging behavior.

## Useful Commands

Train with explicit overrides:

```bash
python scripts/train.py configs/a1.yaml \
  --train \
  --headless \
  --episodes 1000 \
  --sac_decimation 5 \
  --save_every_n 100 \
  --spawn_clearance 0.7 \
  --grid_step 0.1
```

MJX parallel training:

```bash
python scripts/train.py configs/a1.yaml \
  --train \
  --headless \
  --use_mjx \
  --batch_size 128
```

Resume from the latest checkpoint:

```bash
python scripts/train.py configs/a1.yaml \
  --train \
  --headless \
  --load_pretrained
```

Fine-tune from the latest checkpoint:

```bash
python scripts/train.py configs/a1.yaml \
  --train \
  --headless \
  --fine_tune
```

Resume from checkpoint with a fresh replay buffer:

```bash
python scripts/train.py configs/a1.yaml \
  --train \
  --headless \
  --load_pretrained \
  --fresh_buffer
```

Run evaluation:

```bash
python scripts/train.py configs/a1.yaml
```

Inspect the checked-in TensorBoard run:

```bash
tensorboard --logdir runs/Mar26_04-55-56_griga-Katana-GF76-12UGSO
```

Inspect newly created default training runs:

```bash
tensorboard --logdir data/runs
```

Export the trained actor to ONNX:

```bash
python scripts/export_to_onnx.py \
  --model_path data/models/<run_id>/sac_actor.pth \
  --config_path configs/a1.yaml \
  --output_path sac_actor.onnx
```

Run ONNX inference:

```bash
python scripts/inference_onnx.py \
  --model_path sac_actor.onnx \
  --config_path configs/a1.yaml
```

## Output Paths

- Checkpoints: `data/models/<timestamp>/`
- Replay buffer snapshots: `data/buffer/`
- Default TensorBoard logs for new runs: `data/runs/<run_name>/`
- Checked-in sample TensorBoard run: `runs/Mar26_04-55-56_griga-Katana-GF76-12UGSO/`
- Walking policy checkpoint: `data/walking_policy/model_4999.pt`
- Main documentation: `README.md`

## Runtime Notes

- `--train` enables training mode. Without it, `scripts/train.py` runs evaluation.
- In evaluation mode, the script automatically loads the latest checkpoint from `data/models/`.
- The current `a1.yaml` sets `sac_decimation: 10`, so the effective default SAC rate is lower than the parser's bare `--sac_decimation 5` default unless you override it explicitly.
- Headless training without MJX is possible but slow.
- If MJX initialization fails, the script falls back to sequential MuJoCo simulation.

## Troubleshooting

- If `--use_mjx` is requested but MJX or JAX is not installed, install the optional `.[mjx]` extra and rerun.
- If TensorBoard does not show new logs, check whether the run was created under `data/runs/` instead of the checked-in `runs/` directory.
- If evaluation does not find a checkpoint, train once first or confirm that `data/models/` contains a saved run directory.
- For project overview, architecture, reward design, and training plots, use `README.md` rather than this file.
