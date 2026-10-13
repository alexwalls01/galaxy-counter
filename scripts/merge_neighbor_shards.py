"""
Combine the HDF5 shard files produced by save_neighbor_shards.py into HDF5
Virtual Dataset (VDS) files — one for training, one for validation.

A VDS is a thin index file (no data is copied).  Once created, h5py and PyTorch
read through it exactly like a regular HDF5 file, but data is served from the
underlying shards.

Usage
-----
    python scripts/merge_neighbor_shards.py --input-dir /path/to/neighbor_batches

Outputs
-------
    <output-dir>/train_neighbors.vds   — all shards except the last
    <output-dir>/val_neighbors.vds     — last shard only
(--output-dir defaults to --input-dir.)

Recovery shards (*_recovery.h5) left by a crashed save_neighbor_shards.py run are
skipped: rerun that script instead, so no samples are duplicated.

The VDS files point at the shards by path, so keep the shards where they are.
Point GALAXY_COUNTER_PRECOMPUTED_H5 at the train VDS for training.
"""

import argparse
import glob
import os

import h5py
from tqdm import tqdm

SHARD_GLOB = "neighbors_shard_*.h5"
COMPLETE_MARKER = "shards_complete.json"   # written by save_neighbor_shards.py when it finishes


def create_vds(files, output_path):
    """Create a Virtual Dataset that concatenates *files* along the row axis."""
    print(f"Creating VDS {output_path} from {len(files)} files ...")

    with h5py.File(files[0], "r") as f0:
        keys   = list(f0.keys())
        shapes = {k: f0[k].shape[1:] for k in keys}
        dtypes = {k: f0[k].dtype     for k in keys}

    total_rows = 0
    sources = []
    for fn in tqdm(files, desc="Scanning shards"):
        with h5py.File(fn, "r") as f:
            rows = f["targets"].shape[0]
            sources.append((fn, rows))
            total_rows += rows

    print(f"  Total samples: {total_rows}")

    layouts = {
        k: h5py.VirtualLayout(shape=(total_rows,) + shapes[k], dtype=dtypes[k])
        for k in keys
    }

    current_idx = 0
    for fn, rows in tqdm(sources, desc="Mapping VDS"):
        for k in keys:
            vsource = h5py.VirtualSource(fn, k, shape=(rows,) + shapes[k])
            layouts[k][current_idx : current_idx + rows] = vsource
        current_idx += rows

    with h5py.File(output_path, "w", libver="latest") as f:
        for k, layout in layouts.items():
            f.create_virtual_dataset(k, layout)

    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory with the shard files")
    parser.add_argument("--output-dir", default=None, help="Where to write the VDS files (default: --input-dir)")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    files = sorted(glob.glob(os.path.join(args.input_dir, SHARD_GLOB)))
    recovery = [f for f in files if f.endswith("_recovery.h5")]
    files = [f for f in files if not f.endswith("_recovery.h5")]
    if recovery:
        print(f"WARNING: skipping {len(recovery)} recovery shard(s) from a crashed run: {recovery}")
    if not files:
        raise FileNotFoundError(f"No shard files found in {args.input_dir}")
    if not os.path.exists(os.path.join(args.input_dir, COMPLETE_MARKER)):
        print(f"WARNING: no {COMPLETE_MARKER} in {args.input_dir}; save_neighbor_shards.py may not have finished")

    print(f"Found {len(files)} shards.")
    if len(files) < 2:
        raise SystemExit("Need at least 2 shards (all but the last for training, the last for validation); "
                         "rerun save_neighbor_shards.py with a smaller --chunks-per-file.")

    train_files = files[:-1]
    val_files   = files[-1:]

    os.makedirs(output_dir, exist_ok=True)
    create_vds(train_files, os.path.join(output_dir, "train_neighbors.vds"))
    create_vds(val_files,   os.path.join(output_dir, "val_neighbors.vds"))


if __name__ == "__main__":
    main()
