# fan_grasping_preprocessing

## Data

Both datasets are too large for GitHub and live in Hugging Face dataset repos:

| Repo | Contents |
|---|---|
| [ArthurGoldFishKing/fan_grasping_raw](https://huggingface.co/datasets/ArthurGoldFishKing/fan_grasping_raw) | input: `real_world_dataset.hdf5`, the raw 2026-09 recording (107 demos, 220,426 frames, 14 GB) |
| [ArthurGoldFishKing/fan_grasping_lerobot](https://huggingface.co/datasets/ArthurGoldFishKing/fan_grasping_lerobot) | output: the LeRobot v3.0 training set this pipeline produced from it (97 episodes, 99,720 frames, 30 fps, 472 MB) |

```bash
hf download ArthurGoldFishKing/fan_grasping_raw real_world_dataset.hdf5 --repo-type dataset --local-dir data/raw_data
hf download ArthurGoldFishKing/fan_grasping_lerobot --repo-type dataset --local-dir data/lerobot_data
```

## Quick start

```bash
conda activate lerobot                     # the only env with numpy + h5py + lerobot
cd fan_grasping_preprocessing

# 1. check first: read-only, writes nothing
python data_preprocessing_main.py --raw /path/to/recording.hdf5 --task "pick up the fan by its handle" --fps 30 --dry-run

# 2. build the dataset
python data_preprocessing_main.py --raw /path/to/recording.hdf5 --task "pick up the fan by its handle" --fps 30
```

`--fps` is required because the current `ros2_data_collector` does not store `obs/real_stamp`. Without
it the runner exits after step 0, before any check. If a recording does have the stamps, leave
`--fps` out and the rate is measured from them.

Output, next to the raw file (which is never modified):

- `<recording>_prepared.hdf5`: the cleaned HDF5
- `lerobot_data/`: the LeRobot dataset to train on (choose a different location with `--lerobot-out`)

If it stops:

| Message | Fix |
|---|---|
| `No camera rate` | the file has no `obs/real_stamp`; add `--fps 30` (the rate you recorded at) |
| `AssertionError` in step 1 | `<recording>_prepared.hdf5` already exists; delete it or pass `--work new.hdf5` |
| `already exists; refusing to overwrite` | the LeRobot folder exists; delete it or pass `--lerobot-out new_dir` |
| drop_jumps refuses (> 40% dropped) | read the step-size distribution it prints and adjust `--pos-thr` / `--joint-thr` |
| `No module named h5py` | wrong env; use `lerobot`, not `lerobot312` |

The rest of this README explains what each step does and why.

## Overview

The pipeline that turns a raw recording from `ros2_data_collector` into the LeRobot dataset a policy
is trained on. Six step scripts, two helpers (`paths.py`, `common.py`) and a runner
(`data_preprocessing_main.py`) that chains them; the step scripts that edit a file in place are
read-only by default and need `--apply` to change anything.

**What the 2026-09 recording went through**

```
ros2_data_collector  ->  real_world_dataset.hdf5      107 demos, 220,426 frames  (never modified)
                             |  dedup.py                        -> 107 / 132,590
                             |  drop_jumps.py                   ->  97 / 119,947
                             |  fix_quat_signs.py               ->  97 / 119,947 (signs only)
                             |  dedup_rgb.py                    ->  97 /  99,817
                             v
                         real_world_dataset_dedup.hdf5
                             |  convert_to_lerobot.py
                             v
                         data/lerobot_data            97 episodes, 99,720 frames, fps 30
```

A new recording will not necessarily need every step — each one measures the data first and tells
you whether it applies.

## The runner in detail

`data_preprocessing_main.py` (see Quick start) runs steps 0–6 in order.

**Step 0 (inspect)** runs first and only reads the raw file. It reports the episode and frame
counts, the `obs` and per-demo keys, and the fraction of duplicate RGB frames (which decides whether
steps 1 and 4 do anything). It measures the camera rate from `obs/real_stamp`. If that field is
missing and `--fps` is not given, the runner stops: the policy executes one action per camera
period, so a wrong fps changes the robot's speed.

| Option | Default | Meaning |
|---|---|---|
| `--raw` | required | raw HDF5 from `ros2_data_collector`; never modified |
| `--task` | required | language task string stored with every episode |
| `--work` | `<raw>_prepared.hdf5` | working copy written by step 1 and edited in place by steps 2–4 (must differ from `--raw`) |
| `--lerobot-out` | `<work dir>/lerobot_data` | LeRobot dataset directory |
| `--repo-id` | `arthur/fan_grasp` | LeRobot repo id |
| `--fps` | measured | override the camera rate measured from `obs/real_stamp`; a note is printed if it differs by more than 1 Hz |
| `--pos-thr` / `--joint-thr` | 0.005 m / 0.02 rad | passed to `drop_jumps.py` |
| `--min-dup-frac` | 0.005 | passed to `dedup_rgb.py` |
| `--dry-run` | off | step 0, then the step 2 and step 4 checks read-only on the raw file; nothing is written |
| `--keep-going` | off | continue after a failing step instead of stopping there |

At the end it prints a summary: prepared HDF5 episode and frame counts, LeRobot episodes, frames and
fps, and which steps failed, if any.

`dedup.py` also writes `report.json` into this folder: one `[demo, frames before, frames after]`
entry per episode from its last run. Each run overwrites it, so it only describes the most recent
input.

## The scripts

| # | Script | Does | Skips itself when |
|---|---|---|---|
| 0 | `paths.py` | finds the project root by walking up for `data/raw_data`; override with `FAN_GRASP_ROOT`. Run it alone to print what it resolves | — |
| 0 | `common.py` | the action formula every step uses: `dpos = pos[t+1]-pos[t]`, `drot = (R[t+1]·R[t]⁻¹).as_rotvec()`, `gripper = gripper_cmd[t+1]`, last row zeros | imported, not run |
| 1 | `dedup.py` | pass 1: keep frame 0, then a frame only if `real_rgb`, `real_ee_pos` or `real_joint_pos` changed; recompute actions; one image per chunk | takes input/output as arguments |
| 2 | `drop_jumps.py` | measures every episode's worst single-step motion and drops those above the thresholds | nothing exceeds them |
| 3 | `fix_quat_signs.py` | makes `real_ee_quat` sign-continuous (first frame w ≥ 0, no flips after). Only signs change | "0 demos need changes" |
| 4 | `dedup_rgb.py` | pass 2: keep a frame only if the **image** changed, so every frame carries a new camera frame; then `h5repack` | duplicate fraction below `--min-dup-frac` |
| 5 | `convert_to_lerobot.py` | builds the LeRobot v3.0 dataset: wrist video, 7-D state, 7-D action, task string, fps | refuses if the output exists |
| 6 | `verify_lerobot.py` | checks the LeRobot dataset against the HDF5 | read-only |

## Order matters: 2 before 4

`drop_jumps.py` compares **per-step** motion, and `dedup_rgb.py` changes what a step is worth: after
it, a step can span 2–4 original camera frames, so the same real motion produces a larger delta.

Running the current (already deduplicated) dataset through `drop_jumps.py` flags `demo_76` at
5.2 mm — but that step spans 4 original frames where the camera stalled, with neighbours at
3.1–3.5 mm. It is smooth motion over a longer interval, not the 190 mm snap-and-return the threshold
exists to catch. **Run step 2 on evenly spaced data, before step 4.**

## Each script measures before it acts

**`drop_jumps.py`** — no episode list is hard-coded. It reports the step-size distribution
(p50/p99/p99.9/max) and the worst episodes, then drops whatever exceeds `--pos-thr` (default
0.005 m) or `--joint-thr` (default 0.02 rad):

```
worst 5 episodes:
  episode     max|d pos| (mm)  max|d joint| (rad)   verdict
  demo_1              190.058             0.00163   DROP
  demo_3                1.237             0.58043   DROP
  demo_4                1.281             0.00196   keep
```

Safety rails, because thresholds that suit one recording may not suit the next: it refuses to drop
*every* episode, refuses above `--max-drop-frac` (40%) without `--force`, and after applying it
re-measures the survivors and asserts none still exceed the thresholds. Re-running is safe —
`dropped_demos` accumulates and `orig_demo_name` keeps the episode's first name.

**`dedup_rgb.py`** — first counts duplicate frames, then decides:

```
duplicate RGB frames: 0.000% of consecutive steps
Below --min-dup-frac 0.500%: the camera did not republish frames, so every frame is
already unique. SKIPPING this pass; the file is unchanged.
```

If duplicates are present it keeps one frame per unique image and checks each episode ends near
`1 − (its duplicate fraction)` of its current length, within `--ratio-tol` (0.05). That works for
any duplication pattern, not just the 2× of the 2026-09 recording. `orig_indices` is optional, so it
also runs on a raw file that never went through pass 1.

Steps 1 and 4 exist **only** because the RGB topic republished every frame. If that is fixed at the
driver, both will report nothing to do and skip.

## What to re-check for a new recording

- **`convert_to_lerobot.py`**: `FPS`, `TASK`, `REPO_ID`, and the `FEATURES` dict (currently wrist
  image + 7-D state + 7-D action — no depth, no `joint_vel`, no EE pose). Measure the real frame rate
  from `obs/real_stamp` if the recording has it; do not assume 30.
- **Thresholds** in steps 2 and 4: the defaults are printed alongside the measured distribution so
  you can see whether they still make sense.

## Environment

Steps 0–4 (`data_preprocessing_main.py`, `dedup.py`, `drop_jumps.py`, `fix_quat_signs.py`,
`dedup_rgb.py`) need only `numpy` and `h5py`; `dedup_rgb.py` also needs the `h5repack` binary.
`common.py` does the rotation maths in plain numpy, with no scipy; `python3 common.py` checks its
actions against the scipy-computed ones stored in the dataset. `convert_to_lerobot.py` and
`verify_lerobot.py` need **lerobot and h5py in the same environment**.

On this machine, use the `lerobot` conda env (lerobot 0.4.4, h5py, `h5repack` on the path) for the
whole pipeline. `lerobot312` (0.6.2) has no h5py, so it cannot run any step.

## Not included here

Deliberately left in their own places: `claude_utility/final.py` (verification for the old 107-demo
numbering, outdated), `claude_utility/make_obs_only.py` (builds an observation-only replay set),
`analysis/check_joint_vel_lag.py` (the read-only study behind `fps = 30`), and
`baselines/make_splits.py` (the train/validation episode split — that belongs to training).
