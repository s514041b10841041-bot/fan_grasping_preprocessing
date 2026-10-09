"""Locate the project's data directories, wherever this folder is checked out.

The scripts here were written when they lived next to the data; they now sit two levels down in
delta/. Rather than hard-coding either depth, find the project root by walking up from this file
looking for a `data/raw_data` directory, so moving this folder does not break anything.

Override with the FAN_GRASP_ROOT environment variable:

    FAN_GRASP_ROOT=/path/to/Robot_Friday python convert_to_lerobot.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = (os.path.join('data', 'raw_data'), os.path.join('data', 'lerobot_data'))


def project_root() -> str:
    """Directory containing data/raw_data (or data/lerobot_data). Env var wins."""
    env = os.environ.get('FAN_GRASP_ROOT')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    d = HERE
    while True:
        if any(os.path.isdir(os.path.join(d, m)) for m in MARKERS):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(
                "could not find the project root: no data/raw_data or data/lerobot_data in any parent "
                f"of {HERE}. Set FAN_GRASP_ROOT to the directory that contains data/."
            )
        d = parent


def raw(*parts) -> str:
    """Path inside data/raw_data, e.g. raw('real_world_dataset_dedup.hdf5')."""
    return os.path.join(project_root(), 'data', 'raw_data', *parts)


def lerobot_data(*parts) -> str:
    """Path inside data/lerobot_data."""
    return os.path.join(project_root(), 'data', 'lerobot_data', *parts)


if __name__ == '__main__':
    print(f"project root : {project_root()}")
    for p in (raw('real_world_dataset.hdf5'), raw('real_world_dataset_dedup.hdf5'), lerobot_data()):
        print(f"  {'OK     ' if os.path.exists(p) else 'MISSING'} {p}")
