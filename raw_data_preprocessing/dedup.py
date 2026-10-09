import h5py, numpy as np, sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # common.py sits next to this file
from common import compute_actions, demo_names
SRC, DST = sys.argv[1], sys.argv[2]
assert os.path.abspath(SRC) != os.path.abspath(DST) and not os.path.exists(DST)
TMP = DST + '.partial'
if os.path.exists(TMP): os.remove(TMP)
IMG_KEYS = ('real_rgb', 'real_depth')   # stored one frame per chunk, whatever the image size is
KEYS = ['real_rgb', 'real_ee_pos', 'real_joint_pos']
t0 = time.time(); report = []
with h5py.File(SRC, 'r') as fi, h5py.File(TMP, 'w') as fo:
    for k, v in fi.attrs.items(): fo.attrs[k] = v
    for k in fi.keys():
        if k != 'data': fi.copy(fi[k], fo, name=k)
    di = fi['data']; do = fo.create_group('data')
    for k, v in di.attrs.items(): do.attrs[k] = v
    total = 0
    for n in demo_names(di):
        gi = di[n]; oi = gi['obs']; L = gi['actions'].shape[0]
        ref = {k: oi[k][:] for k in KEYS}
        keep = [0]
        for t in range(1, L):
            j = keep[-1]
            if any(not np.array_equal(ref[k][t], ref[k][j]) for k in KEYS): keep.append(t)
        keep = np.asarray(keep, np.int64); del ref
        go = do.create_group(n); oo = go.create_group('obs')
        for k, v in gi.attrs.items(): go.attrs[k] = v
        go.attrs['num_samples'] = np.int64(len(keep))
        for k in oi.keys():
            ds = oi[k]
            if k in IMG_KEYS:
                out = oo.create_dataset(k, shape=(len(keep),) + ds.shape[1:], dtype=ds.dtype,
                                        chunks=(1,) + ds.shape[1:], compression='gzip', compression_opts=4)
                for s in range(0, len(keep), 256):     # bounded memory
                    idx = keep[s:s+256]
                    blk = ds[idx[0]:idx[-1]+1]
                    out[s:s+len(idx)] = blk[idx - idx[0]]
            else:
                oo.create_dataset(k, data=ds[:][keep])
            for a, v in ds.attrs.items(): oo[k].attrs[a] = v
        o = oo
        go.create_dataset('actions', data=compute_actions(o['real_ee_pos'][:], o['real_ee_quat'][:], o['gripper_cmd'][:]))
        go.create_dataset('rewards', data=gi['rewards'][:][keep])
        dn = np.zeros(len(keep), gi['dones'].dtype); dn[-1] = 1
        go.create_dataset('dones', data=dn)
        go.create_dataset('orig_indices', data=keep)
        for k in gi.keys():
            if k not in ('obs', 'actions', 'rewards', 'dones'):
                print("WARNING unhandled key", n, k, flush=True)
        total += len(keep); report.append((n, L, len(keep)))
        print(f"{n:9s} {L:5d} -> {len(keep):5d}  ratio {len(keep)/L:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    do.attrs['total'] = np.int64(total)
    fo.attrs['source_file'] = os.path.basename(SRC)
    fo.attrs['dedup_method'] = ("Per demo: keep frame 0; keep frame t only if real_rgb, real_ee_pos or real_joint_pos "
        "differs (exact equality) from the last kept frame. All obs indexed with the same keep indices; actions "
        "recomputed from deduplicated obs (dpos world, rotvec of R(q[t+1])*R(q[t]).inv() with xyzw quats, "
        "gripper_cmd[t+1], last row zeros); rewards subsampled; dones = 1 on last frame only; "
        "orig_indices stores kept source indices.")
    fo.attrs['approx_hz'] = 'TBD'
json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report.json'), 'w'))
os.rename(TMP, DST)
print("DONE", time.time() - t0)
