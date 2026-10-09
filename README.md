# fan_grasping_preprocessing

Turns a raw `ros2_data_collector` HDF5 recording into the LeRobot dataset the fan-grasping policy is
trained on. The pipeline and its documentation are in
[`raw_data_preprocessing/`](raw_data_preprocessing/README.md).

## Data

Both datasets are too large for GitHub and live in Hugging Face dataset repos:

| Repo | Contents |
|---|---|
| [ArthurGoldFishKing/fan_grasping_raw](https://huggingface.co/datasets/ArthurGoldFishKing/fan_grasping_raw) | input: `real_world_dataset.hdf5`, the raw 2026-09 recording (107 demos, 220,426 frames, 14 GB) |
| [ArthurGoldFishKing/fan_grasping_lerobot](https://huggingface.co/datasets/ArthurGoldFishKing/fan_grasping_lerobot) | output: the LeRobot v3.0 training set this pipeline produced from it (97 episodes, 99,720 frames, 30 fps, 472 MB) |

Download into `data/` at the repo root, where `raw_data_preprocessing/paths.py` looks for it:

```bash
cd fan_grasping_preprocessing
hf download ArthurGoldFishKing/fan_grasping_raw real_world_dataset.hdf5 --repo-type dataset --local-dir data/raw_data
hf download ArthurGoldFishKing/fan_grasping_lerobot --repo-type dataset --local-dir data/lerobot_data
```

Then follow the [Quick start](raw_data_preprocessing/README.md#quick-start).
