"""Drop episodes containing an arm position/joint jump, in place. Works on any dataset.

    python drop_jumps.py                     # read-only: statistics + which episodes would go
    python drop_jumps.py --apply             # drop them, renumber the rest
    python drop_jumps.py --file other.hdf5 --pos-thr 0.008 --joint-thr 0.03 --apply

What a "jump" is: within one episode, a single step where the end-effector moves more than
--pos-thr metres or any joint moves more than --joint-thr radians. In the 2026-09 recording this
caught a glitch where the logged pose leapt away for ~25-30 steps and then snapped back exactly,
while the camera showed no corresponding motion - i.e. a recording fault, not real movement.

Nothing about which episodes are affected is hard-coded: every episode is measured and dropped on
its own merits, so a new recording is handled on its own terms. Re-running is safe - episodes
already dropped stay dropped, and anything newly flagged is dropped too.

Safety rails, because a threshold that is wrong for a new dataset could silently delete everything:
  * read-only by default; --apply is required to modify anything,
  * refuses to drop more than --max-drop-frac of the episodes (default 40%) unless --force,
  * refuses to drop every episode,
  * prints the distribution so the thresholds can be sanity-checked against the data.
"""
import argparse
import os
import sys

import h5py
import numpy as np

from paths import raw

POS_THR = 0.005   # m per step. Reference: in the 2026-09 data the 99th percentile step was ~1.2 mm
JOINT_THR = 0.02  # rad per step. Reference: 99th percentile ~0.003 rad; worst clean episode 0.0034


def demo_names(d):
    return sorted([k for k in d.keys() if k.startswith('demo_')], key=lambda s: int(s.split('_')[1]))


def summarize(d):
    names = demo_names(d)
    total = sum(int(d[n].attrs['num_samples']) for n in names)
    maxa = np.max([np.abs(d[n]['actions'][:]).max(0) for n in names], axis=0) if names else np.zeros(7)
    days = {}
    for n in names:
        day = str(d[n].attrs.get('timestamp', ''))[:10]
        days[day] = days.get(day, 0) + 1
    return len(names), total, maxa, days


def measure(d):
    """Per-episode worst single-step motion: (max |d ee_pos| in m, max |d joint_pos| in rad)."""
    out = {}
    for n in demo_names(d):
        o = d[n]['obs']
        dp = np.linalg.norm(np.diff(o['real_ee_pos'][:].astype(np.float64), axis=0), axis=1)
        dj = np.abs(np.diff(o['real_joint_pos'][:].astype(np.float64), axis=0)).max(1)
        out[n] = (float(dp.max()) if len(dp) else 0.0, float(dj.max()) if len(dj) else 0.0,
                  dp, dj)
    return out


def check(d, pos_thr, joint_thr, top):
    m = measure(d)
    dp_all = np.concatenate([v[2] for v in m.values()]) if m else np.zeros(1)
    dj_all = np.concatenate([v[3] for v in m.values()]) if m else np.zeros(1)
    print(f"{len(m)} episodes, {len(dp_all)} steps")
    print(f"step size percentiles   |d ee_pos| [mm]:  p50 {np.percentile(dp_all, 50)*1000:7.3f}  "
          f"p99 {np.percentile(dp_all, 99)*1000:7.3f}  p99.9 {np.percentile(dp_all, 99.9)*1000:7.3f}  "
          f"max {dp_all.max()*1000:8.3f}")
    print(f"                    max|d joint| [rad]:  p50 {np.percentile(dj_all, 50):7.5f}  "
          f"p99 {np.percentile(dj_all, 99):7.5f}  p99.9 {np.percentile(dj_all, 99.9):7.5f}  "
          f"max {dj_all.max():8.5f}")
    print(f"\nthresholds: pos > {pos_thr} m  or  joint > {joint_thr} rad")
    print(f"\nworst {top} episodes:")
    print(f"  {'episode':10s} {'max|d pos| (mm)':>16s} {'max|d joint| (rad)':>19s}   verdict")
    order = sorted(m, key=lambda n: max(m[n][0] / pos_thr, m[n][1] / joint_thr), reverse=True)
    flagged = [n for n in order if m[n][0] > pos_thr or m[n][1] > joint_thr]
    for n in order[:top]:
        p, j = m[n][0], m[n][1]
        print(f"  {n:10s} {p*1000:16.3f} {j:19.5f}   {'DROP' if n in flagged else 'keep'}")
    return flagged, m


def apply(f, drop):
    d = f['data']
    for n in drop:
        del d[n]
    remaining = demo_names(d)
    mapping = {old: f'demo_{i}' for i, old in enumerate(remaining)}
    # Two-phase move so a new name never collides with an existing one.
    for old in remaining:
        d.move(old, f'tmp_{old}')
    for old, new in mapping.items():
        d.move(f'tmp_{old}', new)
        # Keep the FIRST name this episode ever had: on a re-run, orig_demo_name is already set
        # and must not be overwritten with the intermediate name from the previous pass.
        if 'orig_demo_name' not in d[new].attrs:
            d[new].attrs['orig_demo_name'] = old
    d.attrs['total'] = np.int64(sum(int(d[n].attrs['num_samples']) for n in mapping.values()))
    if 'mask' in f:
        for key in list(f['mask'].keys()):
            vals = [v.decode() if isinstance(v, bytes) else str(v) for v in f['mask'][key][:]]
            new = [mapping[v] for v in vals if v in mapping]
            del f['mask'][key]
            f['mask'].create_dataset(key, data=np.array(new, dtype='S'))
            print(f"mask/{key}: {len(vals)} -> {len(new)}")
    previous = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs.get('dropped_demos', [])]
    f.attrs['dropped_demos'] = np.array(previous + list(drop), dtype=h5py.string_dtype())
    f.attrs['drop_reason'] = 'arm position/joint jumps'
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=raw('real_world_dataset_dedup.hdf5'))
    ap.add_argument('--pos-thr', type=float, default=POS_THR, help='m per step (default %(default)s)')
    ap.add_argument('--joint-thr', type=float, default=JOINT_THR, help='rad per step (default %(default)s)')
    ap.add_argument('--max-drop-frac', type=float, default=0.4,
                    help='refuse to drop more than this fraction of episodes (default %(default)s)')
    ap.add_argument('--top', type=int, default=15, help='how many episodes to list (default %(default)s)')
    ap.add_argument('--force', action='store_true', help='drop even if that exceeds --max-drop-frac')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"{args.file} not found")
    print(f"file: {args.file}\n")
    with h5py.File(args.file, 'r') as f:
        d = f['data']
        already = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs.get('dropped_demos', [])]
        if already:
            print(f"note: {len(already)} episode(s) were dropped by a previous run: {already}\n")
        before = summarize(d)
        flagged, _ = check(d, args.pos_thr, args.joint_thr, args.top)

    n_before = before[0]
    if not flagged:
        print("\nNothing flagged: every episode is clean at these thresholds. No changes needed.")
        return 0
    frac = len(flagged) / max(n_before, 1)
    print(f"\n{len(flagged)} of {n_before} episodes flagged ({frac:.0%}): {flagged}")
    if len(flagged) == n_before:
        sys.exit("Refusing: that is every episode in the file. Check the thresholds against the "
                 "percentiles above - they may be far too tight for this dataset.")
    if frac > args.max_drop_frac and not args.force:
        sys.exit(f"Refusing: {frac:.0%} exceeds --max-drop-frac {args.max_drop_frac:.0%}. If the "
                 f"thresholds are right for this data, re-run with --force.")
    if not args.apply:
        print("\nRead-only check done; rerun with --apply to drop them.")
        return 0

    with h5py.File(args.file, 'r+') as f:
        apply(f, flagged)
    with h5py.File(args.file, 'r') as f:
        d = f['data']
        after = summarize(d)
        names = demo_names(d)
        assert names == [f'demo_{i}' for i in range(len(names))], "renumbering is not contiguous"
        assert not any(k.startswith('tmp_') for k in d.keys()), "leftover tmp_ groups"
        assert d.attrs['total'] == after[1]
        origs = [str(d[n].attrs['orig_demo_name']) for n in names]
        assert len(set(origs)) == len(origs), "duplicate orig_demo_name"
        # everything that survived must now be clean at these thresholds
        still, _ = [], None
        for n in names:
            o = d[n]['obs']
            dp = np.linalg.norm(np.diff(o['real_ee_pos'][:].astype(np.float64), axis=0), axis=1).max()
            dj = np.abs(np.diff(o['real_joint_pos'][:].astype(np.float64), axis=0)).max()
            if dp > args.pos_thr or dj > args.joint_thr:
                still.append(n)
        assert not still, f"episodes still exceeding the thresholds after the drop: {still}"

    np.set_printoptions(precision=5, suppress=True)
    print(f"\n{'':8s}{'episodes':>9s} {'steps':>8s}")
    print(f"before  {before[0]:9d} {before[1]:8d}\nafter   {after[0]:9d} {after[1]:8d}")
    print("max|action| before", before[2])
    print("max|action| after ", after[2])
    print("recording dates before", before[3], " after", after[3])
    print(f"dropped: {flagged}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
