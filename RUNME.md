# RUNME — SuperDog Path Planning

## Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | 2.4.1 |
| MuJoCo | 3.2.3 |
| NumPy | 1.24.3 |
| TensorBoard | ≥ 2.7.0 |
| PyYAML | ≥ 5.4.1 |
| rsl-rl-lib | 2.3.3 |

Optional:

| Package | Purpose |
|---|---|
| `mujoco-mjx`, `jax`, `jaxlib` | MJX GPU/TPU parallel simulation |
| `onnxruntime` | ONNX model verification and real-robot inference |
| ROS2 packages | Real-robot deployment |

## Installation

```bash
# Create and activate conda environment
conda create --name superdog python=3.8
conda activate superdog

# Install package in development mode
pip install -e .

# Optional: MJX (GPU/TPU parallel simulation, 10-100x speedup)
pip install -e ".[mjx]"

# Optional: ROS2 interface
pip install -e ".[ros2]"
```

---

## Training

### Standard training (CPU/GPU, sequential)

```bash
python scripts/train.py configs/a1.yaml --train --headless
```

### Training with all flags

```bash
python scripts/train.py configs/a1.yaml \
    --train \
    --headless \
    --episodes 1000 \
    --sac_decimation 5 \
    --log_dir runs \
    --save_every_n 100 \
    --spawn_clearance 0.7 \
    --grid_step 0.1
```

| Flag | Default | Description |
|---|---|---|
| `config_file` | — | Path to YAML config (required) |
| `--train` | off | Enable training mode (omit for evaluation) |
| `--headless` | off | Disable MuJoCo viewer (faster) |
| `--episodes` | 1000 | Number of training episodes |
| `--sac_decimation` | 5 | Run SAC every N control cycles |
| `--log_dir` | `runs` | TensorBoard log directory |
| `--save_every_n` | 100 | Save checkpoint every N episodes |
| `--spawn_clearance` | 0.7 | Obstacle clearance radius for spawn (m) |
| `--grid_step` | 0.1 | Grid resolution for free space sampling (m) |

### MJX parallel training (GPU/TPU)

```bash
python scripts/train.py configs/a1.yaml \
    --train \
    --headless \
    --use_mjx \
    --batch_size 128
```

| Flag | Default | Description |
|---|---|---|
| `--use_mjx` | off | Enable MJX batched parallel simulation |
| `--batch_size` | 128 | Number of parallel environments (MJX only) |

---

## Fine-tuning

Load a saved checkpoint and continue training with reset entropy and optimizer state:

```bash
python scripts/train.py configs/a1.yaml \
    --train \
    --headless \
    --fine_tune
```

Load checkpoint and continue training **without** resetting buffer or optimizer:

```bash
python scripts/train.py configs/a1.yaml \
    --train \
    --headless \
    --load_pretrained
```

Start training from a checkpoint but with a **fresh replay buffer**:

```bash
python scripts/train.py configs/a1.yaml \
    --train \
    --headless \
    --load_pretrained \
    --fresh_buffer
```

---

## Evaluation

Run evaluation (no `--train` flag — loads latest checkpoint automatically):

```bash
python scripts/train.py configs/a1.yaml
```

Run with visualization:

```bash
python scripts/train.py configs/a1.yaml
# (omit --headless to open the MuJoCo viewer)
```

---

## TensorBoard

```bash
tensorboard --logdir data/runs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

Track a specific run:

```bash
tensorboard --logdir data/runs/Mar26_04-27-43_griga-Katana-GF76-12UGSO
```

---

## Export to ONNX

Export a trained SAC actor for deployment on a real robot:

```bash
python scripts/export_to_onnx.py \
    --model_path data/models/<run_id>/sac_actor.pth \
    --config_path configs/a1.yaml \
    --output_path sac_actor.onnx
```

With model verification (requires `onnxruntime`):

```bash
pip install onnxruntime
python scripts/export_to_onnx.py \
    --model_path data/models/<run_id>/sac_actor.pth \
    --config_path configs/a1.yaml \
    --output_path sac_actor.onnx \
    --verify
```

---

## ONNX Inference

Run inference with an exported ONNX model:

```bash
python scripts/inference_onnx.py \
    --model_path sac_actor.onnx \
    --config_path configs/a1.yaml
```

---

## Project Layout

```
SuperDog/
├── configs/
│   ├── a1.yaml            # Main robot and SAC config
│   └── curriculum.yaml    # Curriculum learning levels
├── data/
│   ├── models/            # Saved checkpoints (gitignored)
│   ├── runs/              # TensorBoard logs (gitignored)
│   └── walking_policy/    # Pretrained walking policy weights
├── scripts/
│   ├── train.py           # Main training + evaluation entry point
│   ├── export_to_onnx.py  # Export SAC actor to ONNX
│   └── inference_onnx.py  # ONNX inference for real robot
├── src/
│   ├── policy/
│   │   ├── SAC/           # SAC actor, critic, utils
│   │   ├── replay_buffer.py
│   │   └── walking_policy.py
│   └── utils/
│       ├── curriculum.py
│       ├── observation.py
│       ├── reward.py
│       ├── scene_generator.py
│       └── target_generator.py
└── assets/
    └── unitree_a1/        # MuJoCo robot model and meshes
```
