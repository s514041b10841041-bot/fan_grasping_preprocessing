"""Canonicalize obs/real_ee_quat signs in real_world_dataset_dedup.hdf5, in place.

q and -q are the same rotation (quaternions are xyzw, so w = q[:, 3]). Rule, per demo:
  1. the first frame has w >= 0 (negate it if w < 0);
  2. sign continuity: q[t] is negated if dot(q[t], q[t-1]) < 0.
Only signs change; actions are untouched (they are rotation vectors, sign-invariant).
Idempotent: running it on already-canonical data changes nothing.

Usage:
    python fix_quat_signs.py            # read-only: report what would change
    python fix_quat_signs.py --apply    # rewrite real_ee_quat in place

Never opens the original real_world_dataset.hdf5.
"""
import argparse
import os
import sys
import h5py
import numpy as np

from paths import raw

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = raw('real_world_dataset_dedup.hdf5')
KEY = 'obs/real_ee_quat'
RULE = ("obs/real_ee_quat (xyzw) sign-canonicalized per demo: first frame negated if w < 0 so that w >= 0, "
        "then q[t] negated when dot(q[t], q[t-1]) < 0 (sign continuity). Only signs changed; actions unchanged.")


def demo_names(d):
    return sorted(d.keys(), key=lambda s: int(s.split('_')[1]))


def canonicalize(q):
    q = q.copy()
    if q[0, 3] < 0:
        q[0] = -q[0]
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def count_flips(q):
    return int((np.sum(q[1:] * q[:-1], axis=1) < 0).sum())


def main():
    global PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=PATH)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    PATH = args.file
    do_apply = args.apply
    print(f"file: {PATH}\n")
    with h5py.File(PATH, 'r') as f:
        d = f['data']
        plan = {}
        for n in demo_names(d):
            q0 = d[n][KEY][:]
            neg = int((canonicalize(q0) == -q0).all(1).sum())
            if neg:
                plan[n] = (neg, count_flips(q0), bool(q0[0, 3] < 0))
    print(f"{len(plan)} demos need changes:")
    for n, (neg, flips, w0neg) in plan.items():
        print(f"  {n:9s} rows to negate {neg:5d}  flips {flips:3d}  first-frame w<0: {w0neg}")
    if not do_apply:
        print("Read-only check done; rerun with --apply to modify the file.")
        return

    with h5py.File(PATH, 'r+') as f:
        d = f['data']
        for n in plan:
            ds = d[n][KEY]
            q0 = ds[:]
            q1 = canonicalize(q0)
            assert ((q1 == q0).all(1) | (q1 == -q0).all(1)).all()   # only whole-row sign changes
            ds[...] = q1
        f.attrs['quat_sign_fixed'] = RULE

    with h5py.File(PATH, 'r') as f:
        d = f['data']
        qs = {n: d[n][KEY][:].astype(np.float64) for n in demo_names(d)}
    ref = qs['demo_0'][0]
    assert all(q[0, 3] >= 0 for q in qs.values())
    assert sum(count_flips(q) for q in qs.values()) == 0
    dots = {n: (q @ ref).min() for n, q in qs.items()}
    worst = min(dots, key=dots.get)
    print(f"Done. All first frames have w >= 0; 0 flips remain.")
    print(f"Min dot(q, demo_0 frame 0) over all {sum(len(q) for q in qs.values())} frames: "
          f"{dots[worst]:.6f} ({worst}, = {np.degrees(2 * np.arccos(min(1, dots[worst]))):.2f} deg)")


if __name__ == '__main__':
    main()
