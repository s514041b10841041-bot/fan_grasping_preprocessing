"""The action formula shared by every preprocessing step.

action[t] = [ pos[t+1] - pos[t],  rotvec(R[t+1] . R[t]^-1),  gripper_cmd[t+1] ]  (last row zeros)

Position delta in metres, world frame. Rotation delta as a rotation vector (axis * angle) in
radians, composed in the world frame (left multiplication). Quaternions are xyzw.

numpy only - deliberately no scipy, so the whole pipeline runs anywhere numpy and h5py are
installed. The quaternion maths below reproduces
`scipy.spatial.transform.Rotation.from_quat(...).as_rotvec()`; `python3 common.py` checks it against
the actions stored in the dataset, which were computed with scipy.
"""
import numpy as np


def _qmul(a, b):
    """Hamilton product of xyzw quaternions, row-wise."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


def _qinv(q):
    """Inverse of a unit xyzw quaternion (its conjugate)."""
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _as_rotvec(q):
    """xyzw quaternion -> rotation vector, matching scipy's as_rotvec()."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = q * np.where(q[:, 3:4] < 0, -1.0, 1.0)          # shortest rotation: keep w >= 0
    v, w = q[:, :3], np.clip(q[:, 3], -1.0, 1.0)
    sin_half = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(sin_half, w)
    # scale = angle / sin(angle/2), with the small-angle limit 2 + angle^2/12 to avoid 0/0
    small = sin_half < 1e-8
    scale = np.empty_like(angle)
    scale[~small] = angle[~small] / sin_half[~small]
    scale[small] = 2.0 + angle[small] ** 2 / 12.0
    return v * scale[:, None]


def demo_names(d):
    """demo_0, demo_1, ... in numeric order (demo_10 after demo_9, not after demo_1)."""
    return sorted([k for k in d.keys() if k.startswith('demo_')], key=lambda s: int(s.split('_')[1]))


def compute_actions(p, q, gc):
    """Positions (T,3), quaternions xyzw (T,4), gripper (T,) -> actions (T,7) float32."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    L = len(p)
    a = np.zeros((L, 7), np.float64)
    if L > 1:
        a[:-1, :3] = p[1:] - p[:-1]
        a[:-1, 3:6] = _as_rotvec(_qmul(q[1:], _qinv(q[:-1])))
        a[:-1, 6] = np.asarray(gc)[1:]
    return a.astype(np.float32)


if __name__ == '__main__':
    # Regression check against actions computed with scipy and stored in the dataset.
    import h5py
    from paths import raw

    path = raw('real_world_dataset_dedup.hdf5')
    worst = 0.0
    with h5py.File(path, 'r') as f:
        d = f['data']
        names = sorted(d.keys(), key=lambda s: int(s.split('_')[1]))
        for n in names:
            g = d[n]
            got = compute_actions(g['obs/real_ee_pos'][:], g['obs/real_ee_quat'][:], g['obs/gripper_cmd'][:])
            worst = max(worst, float(np.abs(got - g['actions'][:]).max()))
    print(f"checked {len(names)} episodes in {path}")
    print(f"max |numpy actions - stored (scipy) actions| = {worst:.3e}")
    print("OK - identical within float32 precision" if worst < 1e-6 else "MISMATCH - do not use")
