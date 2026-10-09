"""Second, stricter dedup pass on real_world_dataset_dedup.hdf5, in place.

Pass 1 (dedup.py) kept a frame if real_rgb, real_ee_pos or real_joint_pos changed, so some kept
frames repeat the previous camera image. This pass keeps frame 0, then frame t only if real_rgb[t]
differs (exact equality) from real_rgb of the last kept frame. Every obs dataset, rewards and
orig_indices are subset; actions are recomputed with common.compute_actions; dones = 1 on the last
frame only; num_samples, data.attrs['total'], dedup_method and approx_hz are updated.

Usage (needs scipy for common.compute_actions):
    python dedup_rgb.py            # read-only check: per-demo lengths, spacing, depth/rgb mismatch
    python dedup_rgb.py --apply    # check, apply in place, verify, then h5repack and replace the file

Each demo is rebuilt in a temporary group data/_new_<demo>, verified against the old group, and only
then swapped in, so an interrupted run never leaves a half-written demo. Never opens the original
real_world_dataset.hdf5.
"""
import argparse
import os
import sys
import time
import subprocess
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # common.py sits next to this file
from common import compute_actions  # noqa: E402
from paths import raw  # noqa: E402

PATH = raw('real_world_dataset_dedup.hdf5')
TMP = PATH + '.repack.tmp'
# This pass only makes sense when the camera topic actually delivered duplicate frames. A fixed camera
# (or a driver that no longer republishes) produces none, and then there is nothing to remove: the run
# is skipped rather than rewriting the file for no reason.
MIN_DUP_FRAC = 0.005     # skip entirely below this fraction of duplicate frames (0.5%)
# Sanity guard, replacing the old "must end at 45-55% of the original length" rule, which assumed
# exactly 2x duplication. The expected new length is (1 - duplicate fraction) x current length, per
# episode; flag any episode deviating by more than this.
RATIO_TOL = 0.05
TMP_PREFIX = '_new_'
DEDUP_METHOD = (
    "Pass 1 (dedup.py, from real_world_dataset.hdf5): per demo, keep frame 0; keep frame t only if real_rgb, "
    "real_ee_pos or real_joint_pos differs (exact equality) from the last kept frame. "
    "Pass 2 (dedup_rgb.py, after dropping the jump demos and fixing quaternion signs): keep frame 0; keep frame t "
    "only if real_rgb differs (exact equality) from the last kept frame. After each pass all obs, rewards and "
    "orig_indices are subset with the same indices; actions recomputed from the subset obs (dpos world, rotvec of "
    "R(q[t+1])*R(q[t]).inv() with xyzw quats, gripper_cmd[t+1], last row zeros); dones = 1 on last frame only. "
    "orig_indices index into the original file's demo named by orig_demo_name.")


def demo_names(d):
    return sorted([k for k in d.keys() if k.startswith('demo_')], key=lambda s: int(s.split('_')[1]))


def scan_duplicates(d):
    """Fraction of consecutive frames whose RGB is byte-identical, per episode and overall.

    Cheap first pass: this is what decides whether the dedup is needed at all.
    """
    per, dup_total, step_total = {}, 0, 0
    for n in demo_names(d):
        ds = d[n]['obs']['real_rgb']
        dup = 0
        prev = None
        for s0 in range(0, ds.shape[0], 256):        # bounded memory
            blk = ds[s0:s0 + 256]
            if prev is not None and np.array_equal(blk[0], prev):
                dup += 1
            dup += int((~changed(blk)).sum())
            prev = blk[-1]
        steps = ds.shape[0] - 1
        per[n] = dup / steps if steps else 0.0
        dup_total += dup
        step_total += steps
    return per, (dup_total / step_total if step_total else 0.0)


def rgb_keep(rgb):
    keep = [0]
    for t in range(1, len(rgb)):
        if not np.array_equal(rgb[t], rgb[keep[-1]]):
            keep.append(t)
    return np.asarray(keep, np.int64)


def changed(x):
    return (x[1:] != x[:-1]).reshape(len(x) - 1, -1).any(1)


def spacing_counts(orig_idx):
    s = np.diff(orig_idx)
    return np.array([(s == 1).sum(), (s == 2).sum(), (s == 3).sum(), (s > 3).sum()])


def estimate_hz(j, v):
    """Fit joint_vel = hz * (joint_pos[i+1] - joint_pos[i]) over consecutive frames (trapezoid velocity)."""
    dj = np.diff(j.astype(np.float64), axis=0).ravel()
    vv = ((v[1:] + v[:-1]) / 2).astype(np.float64).ravel()
    m = np.abs(dj) > 1e-4
    return float(vv[m] @ vv[m] / (vv[m] @ dj[m]))


def action_stats(d):
    zero = n = 0
    maxa = np.zeros(7)
    for name in demo_names(d):
        a = d[name]['actions'][:]
        zero += int((np.abs(a[:-1, :6]).max(1) == 0).sum()); n += len(a) - 1
        maxa = np.maximum(maxa, np.abs(a).max(0))
    return zero, n, maxa


def quat_ok(q):
    return q[0, 3] >= 0 and not (np.sum(q[1:] * q[:-1], axis=1) < 0).any()


def check(d, dup_per, ratio_tol):
    rows, keeps = [], {}
    sp_old, sp_new = np.zeros(4, int), np.zeros(4, int)
    d_not_r = r_not_d = 0
    for n in demo_names(d):
        o = d[n]['obs']
        rgb = o['real_rgb'][:]
        keep = rgb_keep(rgb)
        cr, cd = changed(rgb), changed(o['real_depth'][:])
        del rgb
        d_not_r += int((cd & ~cr).sum()); r_not_d += int((cr & ~cd).sum())
        has_oi = 'orig_indices' in d[n]
        if has_oi:
            oi = d[n]['orig_indices'][:]
            so, sn = spacing_counts(oi), spacing_counts(oi[keep])
            sp_old += so; sp_new += sn
            cur = len(oi)
        else:                      # a raw file that never went through pass 1
            sn = np.zeros(4, int)
            cur = int(d[n].attrs['num_samples'])
        keeps[n] = keep
        rows.append((n, cur, len(keep), sn, dup_per[n]))
    print(f"{'demo':9s} {'cur':>6s} {'new':>6s} {'new/cur':>8s} {'expected':>9s} {'dup frac':>9s}   spacing 1/2/3/>3")
    for n, L0, L1, sn, dup in rows:
        exp = 1.0 - dup
        flag = f'  <-- expected ~{exp:.3f}' if abs(L1 / L0 - exp) > ratio_tol else ''
        print(f"{n:9s} {L0:6d} {L1:6d} {L1 / L0:8.3f} {exp:9.3f} {dup:9.3f}   "
              f"{sn[0]:5d} {sn[1]:5d} {sn[2]:4d} {sn[3]:4d}{flag}")
    r = np.array([L1 / L0 for _, L0, L1, _, _ in rows])
    print(f"\nTotal steps {sum(x[1] for x in rows)} -> {sum(x[2] for x in rows)};  kept ratio min/median/max "
          f"{r.min():.3f}/{np.median(r):.3f}/{r.max():.3f} of current")
    if sp_old.sum():        # only meaningful when orig_indices exists (i.e. after pass 1)
        fmt = lambda s: ', '.join(f"{k}: {v} ({v / s.sum():.1%})" for k, v in zip(['1', '2', '3', '>3'], s))
        print(f"Spacing now:   {fmt(sp_old)}")
        print(f"Spacing after: {fmt(sp_new)}")
    print(f"Consecutive frames (current file): depth changes but rgb doesn't: {d_not_r}; "
          f"rgb changes but depth doesn't: {r_not_d}")
    over = [n for n, L0, L1, _, dup in rows if abs(L1 / L0 - (1.0 - dup)) > ratio_tol]
    return keeps, over


def new_like(group, name, old, data):
    ds = group.create_dataset(name, data=data, dtype=old.dtype, chunks=old.chunks, compression=old.compression,
                              compression_opts=old.compression_opts, shuffle=old.shuffle, fletcher32=old.fletcher32)
    for k, v in old.attrs.items():
        ds.attrs[k] = v
    return ds


def rebuild_demo(d, n, keep, rng):
    old = d[n]
    expected = {'obs', 'actions', 'rewards', 'dones'}
    assert expected <= set(old.keys()) <= expected | {'orig_indices'}, (n, set(old.keys()))
    has_oi = 'orig_indices' in old
    if TMP_PREFIX + n in d:
        del d[TMP_PREFIX + n]
    new = d.create_group(TMP_PREFIX + n)
    for k, v in old.attrs.items():
        new.attrs[k] = v
    L = len(keep)
    new.attrs['num_samples'] = np.int64(L)
    oo, no = old['obs'], new.create_group('obs')
    for k, v in oo.attrs.items():
        no.attrs[k] = v
    spots = np.unique(np.concatenate([[0, L // 2, L - 1], rng.integers(0, L, 5)]))
    lowdim = {}
    for k in oo.keys():
        data = oo[k][:][keep]
        if k == 'real_rgb':
            assert changed(data).all(), (n, 'repeated rgb after dedup')
        if data.ndim <= 2:
            lowdim[k] = data
        new_like(no, k, oo[k], data)
        del data
    acts = compute_actions(lowdim['real_ee_pos'], lowdim['real_ee_quat'], lowdim['gripper_cmd'])
    new_like(new, 'actions', old['actions'], acts)
    new_like(new, 'rewards', old['rewards'], old['rewards'][:][keep])
    dn = np.zeros(L, old['dones'].dtype); dn[-1] = 1
    new_like(new, 'dones', old['dones'], dn)
    if has_oi:
        new_like(new, 'orig_indices', old['orig_indices'], old['orig_indices'][:][keep])

    # verify new group against the old one before swapping
    for k in oo.keys():
        if oo[k].ndim <= 2:
            assert np.array_equal(no[k][:], oo[k][:][keep]), (n, k)
        else:
            for i in spots:
                assert np.array_equal(no[k][i], oo[k][keep[i]]), (n, k, i)
    if has_oi:
        assert np.array_equal(new['orig_indices'][:], old['orig_indices'][:][keep])
    assert np.array_equal(new['actions'][:], compute_actions(no['real_ee_pos'][:], no['real_ee_quat'][:],
                                                             no['gripper_cmd'][:]))
    assert quat_ok(no['real_ee_quat'][:]), (n, 'quaternion sign')
    for k in list(no.keys()) + ['actions', 'rewards', 'dones'] + (['orig_indices'] if has_oi else []):
        ds = no[k] if k in no else new[k]
        ref = oo[k] if k in oo else old[k]
        assert ds.shape[0] == L and (ds.chunks, ds.compression, ds.compression_opts) == \
            (ref.chunks, ref.compression, ref.compression_opts), (n, k)
    hz = estimate_hz(lowdim['real_joint_pos'], lowdim['real_joint_vel'])
    del d[n]
    d.move(TMP_PREFIX + n, n)
    return hz


def verify_file(path, full_rgb=True):
    with h5py.File(path, 'r') as f:
        d = f['data']
        names = demo_names(d)
        assert names == [f'demo_{i}' for i in range(len(names))]
        assert not any(k.startswith(TMP_PREFIX) for k in d.keys())
        total = 0
        for n in names:
            g = d[n]; o = g['obs']; L = int(g.attrs['num_samples']); total += L
            keys = ['actions', 'rewards', 'dones'] + (['orig_indices'] if 'orig_indices' in g else [])
            assert all(o[k].shape[0] == L for k in o) and all(g[k].shape[0] == L for k in keys)
            assert g['dones'][:].sum() == 1 and g['dones'][-1] == 1
            assert np.array_equal(g['actions'][:], compute_actions(o['real_ee_pos'][:], o['real_ee_quat'][:],
                                                                   o['gripper_cmd'][:])), n
            assert quat_ok(o['real_ee_quat'][:]), n
            if 'orig_indices' in g:
                assert (np.diff(g['orig_indices'][:]) > 0).all(), n
            if full_rgb:
                assert changed(o['real_rgb'][:]).all(), (n, 'repeated rgb')
        assert d.attrs['total'] == total
        return len(names), total


def repack_and_replace():
    if os.path.exists(TMP):
        os.remove(TMP)
    t = time.time()
    subprocess.run(['h5repack', PATH, TMP], check=True)
    print(f"h5repack done in {time.time() - t:.0f}s", flush=True)
    # quick verification of the repacked copy against the in-place file
    rng = np.random.default_rng(1)
    with h5py.File(PATH, 'r') as a, h5py.File(TMP, 'r') as b:
        assert dict(a.attrs) .keys() == dict(b.attrs).keys()
        def _same(x, y):
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype.kind == 'f' and y.dtype.kind == 'f':
                return np.array_equal(x, y, equal_nan=True)      # NaN == NaN for this purpose
            return np.array_equal(x, y)
        assert all(_same(a.attrs[k], b.attrs[k]) for k in a.attrs), \
            [k for k in a.attrs if not _same(a.attrs[k], b.attrs[k])]
        da, db = a['data'], b['data']
        assert demo_names(da) == demo_names(db) and da.attrs['total'] == db.attrs['total']
        for n in demo_names(da):
            ga, gb = da[n], db[n]
            assert set(ga.attrs.keys()) == set(gb.attrs.keys())
            for k in (['actions', 'rewards', 'dones'] + (['orig_indices'] if 'orig_indices' in ga else [])
                      + [f'obs/{k}' for k in ga['obs']]):
                x, y = ga[k], gb[k]
                assert (x.shape, x.dtype, x.chunks, x.compression, x.compression_opts) == \
                    (y.shape, y.dtype, y.chunks, y.compression, y.compression_opts), (n, k)
                if x.ndim <= 2:
                    assert np.array_equal(x[:], y[:]), (n, k)
                else:
                    for i in rng.integers(0, x.shape[0], 3):
                        assert np.array_equal(x[i], y[i]), (n, k, i)
    verify_file(TMP, full_rgb=False)
    size_before, size_after = os.path.getsize(PATH), os.path.getsize(TMP)
    os.replace(TMP, PATH)
    return size_before, size_after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=raw('real_world_dataset_dedup.hdf5'))
    ap.add_argument('--min-dup-frac', type=float, default=MIN_DUP_FRAC,
                    help='skip the whole pass below this duplicate fraction (default %(default)s)')
    ap.add_argument('--ratio-tol', type=float, default=RATIO_TOL,
                    help='per-episode tolerance on the expected kept ratio (default %(default)s)')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    global PATH, TMP
    PATH, TMP = args.file, args.file + '.repack.tmp'
    do_apply = args.apply
    size0 = os.path.getsize(PATH)
    print(f"file: {PATH}\n")

    # Step 1: is there anything to do? A camera that does not republish frames produces no duplicates.
    with h5py.File(PATH, 'r') as f:
        d = f['data']
        dup_per, dup_all = scan_duplicates(d)
    worst = sorted(dup_per.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"duplicate RGB frames: {dup_all:.3%} of consecutive steps "
          f"(worst episodes: {', '.join(f'{n} {v:.1%}' for n, v in worst)})")
    if dup_all < args.min_dup_frac:
        print(f"\nBelow --min-dup-frac {args.min_dup_frac:.3%}: the camera did not republish frames, so every "
              f"frame is already unique. SKIPPING this pass; the file is unchanged.")
        return 0
    print(f"At or above --min-dup-frac {args.min_dup_frac:.3%}: duplicates present, continuing.\n")

    with h5py.File(PATH, 'r') as f:
        d = f['data']
        zero0, n0, max0 = action_stats(d)
        keeps, over = check(d, dup_per, args.ratio_tol)
    if over:
        sys.exit(f"\n{len(over)} episode(s) would not end up near (1 - duplicate fraction) of their current "
                 f"length: {over}. That suggests the duplicates are not a simple repeat pattern. "
                 f"Not modifying anything.")
    if not do_apply:
        print("\nRead-only check done; rerun with --apply to modify the file.")
        return

    t0 = time.time()
    rng = np.random.default_rng(0)
    hz = {}
    with h5py.File(PATH, 'r+') as f:
        d = f['data']
        for k in [k for k in d.keys() if k.startswith(TMP_PREFIX)]:
            del d[k]                      # leftovers from an interrupted run
        for n in demo_names(d):
            hz[n] = rebuild_demo(d, n, keeps[n], rng)
            print(f"{n:9s} rebuilt  [{time.time() - t0:.0f}s]", flush=True)
        d.attrs['total'] = np.int64(sum(int(d[n].attrs['num_samples']) for n in demo_names(d)))
        h = np.array(list(hz.values()))
        f.attrs['dedup_method'] = DEDUP_METHOD
        h = h[np.isfinite(h)]
        if len(h):
            f.attrs['approx_hz'] = float(round(np.median(h), 1))
        else:
            # joint_vel is all zeros or missing, so d(joint_pos)/joint_vel says nothing about the rate
            print("WARNING: could not estimate the frame rate from joint_vel; approx_hz not written")
    if len(h):
        print(f"approx_hz per demo: median {np.median(h):.2f}, IQR {np.percentile(h, 25):.2f}-{np.percentile(h, 75):.2f}, "
              f"min {h.min():.2f}, max {h.max():.2f}", flush=True)

    ndemo, total = verify_file(PATH)
    size1 = os.path.getsize(PATH)
    with h5py.File(PATH, 'r') as f:
        zero1, n1, max1 = action_stats(f['data'])
    print(f"Verified in-place file: {ndemo} demos, {total} steps. [{time.time() - t0:.0f}s]", flush=True)

    s_before, s_after = repack_and_replace()
    np.set_printoptions(precision=5, suppress=True)
    print(f"\nall-zero 6D actions (excl last row): before {zero0}/{n0} = {zero0 / n0:.3%}   "
          f"after {zero1}/{n1} = {zero1 / n1:.3%}")
    print("max|a| before", max0)
    print("max|a| after ", max1)
    print(f"file size: at start {size0:,}  after in-place pass {size1:,}  after h5repack {s_after:,} bytes")
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
