"""Run the whole pipeline: a raw recording from ros2_data_collector -> a LeRobot dataset.

    python data_preprocessing_main.py --raw datasets/dataset.hdf5 --task "pick up the fan by its handle"
    python data_preprocessing_main.py --raw datasets/dataset.hdf5 --dry-run     # measure only, change nothing

Steps, in this order (each one measures the data first and skips itself when it does not apply):

    0. inspect      raw file: episodes, frames, fields, duplicate frames, camera rate
    1. dedup.py     pass 1: drop frames where rgb, ee_pos and joint_pos are all unchanged
    2. drop_jumps   drop episodes containing a pose/joint jump
    3. fix_quat     make quaternion signs continuous
    4. dedup_rgb    pass 2: keep only frames carrying a new camera image, then h5repack
    5. convert      build the LeRobot dataset
    6. verify       check the dataset against the HDF5 it came from

Order matters: step 2 compares per-step motion, and step 4 changes how much time a step covers, so
jump detection has to happen while the frames are still evenly spaced.

The raw file is never modified. Step 1 writes a working copy (--work) and every later step edits
that copy in place.

Environment: numpy + h5py for steps 0-4, and lerobot in the same environment for steps 5-6. On this
machine that is the `lerobot` conda env.
"""
import argparse
import os
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sh(cmd, label):
    """Run a step, streaming its output. Returns the exit code."""
    print(f"\n{'=' * 78}\n== {label}\n== {' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd[1:3])}"
          f"\n{'=' * 78}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"-- {label}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
    return rc


def changed(x):
    return (x[1:] != x[:-1]).reshape(len(x) - 1, -1).any(1)


def inspect(path):
    """Report what the raw file contains and what the later steps will find."""
    with h5py.File(path, 'r') as f:
        d = f['data']
        names = sorted([k for k in d if k.startswith('demo_')], key=lambda s: int(s.split('_')[1]))
        if not names:
            sys.exit(f"{path} has no demo_* groups under /data")
        total = sum(int(d[n].attrs['num_samples']) for n in names)
        obs_keys = sorted(d[names[0]]['obs'].keys())
        top_keys = sorted(k for k in d[names[0]].keys() if k != 'obs')
        print(f"episodes {len(names)}, frames {total}")
        print(f"obs keys : {', '.join(obs_keys)}")
        print(f"per demo : {', '.join(top_keys)}")

        # duplicate camera frames (what decides whether steps 1 and 4 do anything)
        dup = steps = 0
        for n in names:
            ds = d[n]['obs']['real_rgb']
            prev = None
            for s0 in range(0, ds.shape[0], 256):
                blk = ds[s0:s0 + 256]
                if prev is not None and np.array_equal(blk[0], prev):
                    dup += 1
                dup += int((~changed(blk)).sum())
                prev = blk[-1]
            steps += ds.shape[0] - 1
        dup_frac = dup / steps if steps else 0.0
        print(f"duplicate RGB frames: {dup_frac:.2%} of consecutive steps")

        # camera rate, if the recorder stored the image timestamps
        fps = None
        if 'real_stamp' in d[names[0]]['obs']:
            per = []
            for n in names:
                st = d[n]['obs']['real_stamp'][:].astype(np.float64)
                dt = np.diff(st)
                dt = dt[dt > 0]                       # ignore repeated stamps from republished frames
                if len(dt):
                    per.append(1.0 / np.median(dt))
            if per:
                fps = float(np.median(per))
                print(f"camera rate from obs/real_stamp: {fps:.2f} Hz (median over {len(per)} episodes)"
                      f"  <- measured, not assumed")
        else:
            print("obs/real_stamp absent: the camera rate cannot be measured from this file "
                  "(--fps must be supplied)")
    return {'episodes': len(names), 'frames': total, 'dup_frac': dup_frac, 'fps': fps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='raw HDF5 from ros2_data_collector (never modified)')
    ap.add_argument('--work', help='working copy to create (default: <raw>_prepared.hdf5)')
    ap.add_argument('--lerobot-out', help='LeRobot dataset directory (default: <work dir>/lerobot_data)')
    ap.add_argument('--task', required=True, help='language task string stored with every episode')
    ap.add_argument('--repo-id', default='arthur/fan_grasp')
    ap.add_argument('--fps', type=int, help='override the measured camera rate')
    ap.add_argument('--pos-thr', type=float, default=0.005, help='jump threshold, m per step')
    ap.add_argument('--joint-thr', type=float, default=0.02, help='jump threshold, rad per step')
    ap.add_argument('--min-dup-frac', type=float, default=0.005, help='skip pass 2 below this duplicate fraction')
    ap.add_argument('--dry-run', action='store_true', help='inspect and run every check read-only')
    ap.add_argument('--keep-going', action='store_true', help='continue after a failing step')
    args = ap.parse_args()

    raw_path = os.path.abspath(args.raw)
    if not os.path.exists(raw_path):
        sys.exit(f"{raw_path} not found")
    work = os.path.abspath(args.work or raw_path.replace('.hdf5', '') + '_prepared.hdf5')
    out = os.path.abspath(args.lerobot_out or os.path.join(os.path.dirname(work), 'lerobot_data'))
    if work == raw_path:
        sys.exit("--work must differ from --raw: the raw recording is never modified")

    t_start = time.time()
    print(f"raw          {raw_path}\nworking copy {work}\nlerobot out  {out}")
    print(f"\n{'=' * 78}\n== step 0: inspect the raw recording\n{'=' * 78}")
    info = inspect(raw_path)

    fps = args.fps or (round(info['fps']) if info['fps'] else None)
    if fps is None:
        sys.exit("\nNo camera rate: this file has no obs/real_stamp, so pass --fps explicitly. "
                 "It must be the rate the demonstrations were recorded at - the policy executes one "
                 "action per camera period, so a wrong value scales the robot's speed.")
    if args.fps and info['fps'] and abs(args.fps - info['fps']) > 1:
        print(f"\nNOTE: --fps {args.fps} differs from the measured {info['fps']:.2f} Hz; using {args.fps}.")
    print(f"\nusing fps = {fps}")

    if args.dry_run:
        print(f"\n{'=' * 78}\n== dry run: read-only checks, nothing will be written\n{'=' * 78}")
        for label, cmd in [
            ('step 2 check: jumps', [PY, os.path.join(HERE, 'drop_jumps.py'), '--file', raw_path,
                                     '--pos-thr', str(args.pos_thr), '--joint-thr', str(args.joint_thr)]),
            ('step 4 check: duplicate frames', [PY, os.path.join(HERE, 'dedup_rgb.py'), '--file', raw_path,
                                                '--min-dup-frac', str(args.min_dup_frac)]),
        ]:
            sh(cmd, label)
        print(f"\ndry run finished in {time.time() - t_start:.0f}s. Re-run without --dry-run to build the dataset.")
        return 0

    steps = [
        ('step 1: dedup pass 1 (drop held frames)',
         [PY, os.path.join(HERE, 'dedup.py'), raw_path, work]),
        ('step 2: drop episodes with pose/joint jumps',
         [PY, os.path.join(HERE, 'drop_jumps.py'), '--file', work,
          '--pos-thr', str(args.pos_thr), '--joint-thr', str(args.joint_thr), '--apply']),
        ('step 3: quaternion sign continuity',
         [PY, os.path.join(HERE, 'fix_quat_signs.py'), '--file', work, '--apply']),
        ('step 4: dedup pass 2 (one frame per camera image)',
         [PY, os.path.join(HERE, 'dedup_rgb.py'), '--file', work,
          '--min-dup-frac', str(args.min_dup_frac), '--apply']),
        ('step 5: convert to a LeRobot dataset',
         [PY, os.path.join(HERE, 'convert_to_lerobot.py'), '--input', work, '--out', out,
          '--fps', str(fps), '--task', args.task, '--repo-id', args.repo_id]),
        ('step 6: verify the dataset against the HDF5',
         [PY, os.path.join(HERE, 'verify_lerobot.py'), '--hdf5', work, '--lerobot', out,
          '--repo-id', args.repo_id]),
    ]
    failed = []
    for label, cmd in steps:
        rc = sh(cmd, label)
        if rc != 0:
            failed.append(label)
            if not args.keep_going:
                print(f"\nSTOPPED at: {label}. Nothing after it has run. "
                      f"Fix the cause and re-run, or pass --keep-going.")
                return 1

    print(f"\n{'=' * 78}\n== summary\n{'=' * 78}")
    if os.path.exists(work):
        with h5py.File(work, 'r') as f:
            d = f['data']
            names = [k for k in d if k.startswith('demo_')]
            print(f"prepared HDF5 : {work}\n                {len(names)} episodes, "
                  f"{sum(int(d[n].attrs['num_samples']) for n in names)} frames")
    if os.path.exists(os.path.join(out, 'meta', 'info.json')):
        import json
        i = json.load(open(os.path.join(out, 'meta', 'info.json')))
        print(f"lerobot set   : {out}\n                {i['total_episodes']} episodes, "
              f"{i['total_frames']} frames, fps {i['fps']}")
    print(f"raw untouched : {raw_path}")
    print(f"\n{'FAILED steps: ' + ', '.join(failed) if failed else 'All steps completed'} "
          f"in {time.time() - t_start:.0f}s")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
