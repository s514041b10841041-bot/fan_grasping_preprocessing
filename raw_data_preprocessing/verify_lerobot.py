"""Verify ./data/lerobot_data against real_world_dataset_dedup.hdf5 (read-only on both).

    ~/miniconda3/envs/lerobot/bin/python claude_utility/verify_lerobot.py
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from paths import lerobot_data, raw

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser(description='Verify a LeRobot dataset against the HDF5 it came from.')
_ap.add_argument('--hdf5', default=raw('real_world_dataset_dedup.hdf5'))
_ap.add_argument('--lerobot', default=lerobot_data())
_ap.add_argument('--repo-id', default='arthur/fan_grasp')
_args = _ap.parse_args()
H5, OUT, REPO_ID = _args.hdf5, os.path.abspath(_args.lerobot), _args.repo_id
np.set_printoptions(precision=6, suppress=True, linewidth=150)

src = json.load(open(os.path.join(OUT, 'meta', 'episode_source.json')))
starts = np.concatenate([[0], np.cumsum([e['num_frames'] for e in src])])

with h5py.File(H5, 'r') as f:
    d = f['data']
    names = sorted([k for k in d.keys() if k.startswith('demo_')], key=lambda s: int(s.split('_')[1]))
    h5_total = int(d.attrs['total'])
    assert [e['hdf5_demo'] for e in src] == names

    ds = LeRobotDataset(REPO_ID, root=OUT)
    print(f"episodes: {ds.num_episodes} (HDF5 demos {len(names)})")
    print(f"frames:   {ds.num_frames} (HDF5 total {h5_total} - {len(names)} dropped last frames = {h5_total - len(names)})")
    print(f"fps: {ds.fps}   tasks: {list(ds.meta.tasks.index)}")

    item = ds[0]
    print("\nds[0]:")
    for k, v in item.items():
        print(f"  {k:28s} {tuple(v.shape) if hasattr(v, 'shape') else ''} {getattr(v, 'dtype', type(v).__name__)}"
              + (f"  = {v}" if not hasattr(v, 'shape') or v.numel() <= 1 else ""))

    rng = np.random.default_rng(42)
    print("\n3 random (demo, frame) checks:")
    for _ in range(3):
        ep = int(rng.integers(0, len(names))); t = int(rng.integers(0, src[ep]['num_frames']))
        it = ds[int(starts[ep] + t)]
        assert int(it['episode_index']) == ep and int(it['frame_index']) == t
        g = d[names[ep]]
        h_state = np.concatenate([g['obs/real_joint_pos'][t], [g['obs/gripper_cmd'][t]]]).astype(np.float32)
        h_act = g['actions'][t].astype(np.float32)
        h_img = g['obs/real_rgb'][t].astype(np.float32) / 255.0
        img = it['observation.images.wrist'].permute(1, 2, 0).numpy()          # CHW float [0,1] -> HWC
        print(f"  {names[ep]} ({src[ep]['orig_demo_name']}) frame {t}: timestamp {float(it['timestamp']):.4f} "
              f"(= {t}/30: {abs(float(it['timestamp']) - t / 30) < 1e-4})")
        print(f"     max|state diff| {np.abs(it['observation.state'].numpy() - h_state).max():.2e}   "
              f"max|action diff| {np.abs(it['action'].numpy() - h_act).max():.2e}   "
              f"image mean|diff| {np.abs(img - h_img).mean() * 255:.2f}/255 (video is lossy)")

stats = json.load(open(os.path.join(OUT, 'meta', 'stats.json')))
for key in ['action', 'observation.state']:
    names_ = ds.features[key]['names']
    print(f"\n{key} stats:")
    print(f"  {'dim':8s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
    for i, nm in enumerate(names_):
        s = {k: np.asarray(stats[key][k]).ravel()[i] for k in ['mean', 'std', 'min', 'max']}
        print(f"  {nm:8s} {s['mean']:12.6f} {s['std']:12.6f} {s['min']:12.6f} {s['max']:12.6f}")
