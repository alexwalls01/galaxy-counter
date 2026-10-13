"""
Iterate through a NeighborsDataset and write pre-baked, pre-padded batches to
compressed HDF5 shard files.  Each shard stores ~CHUNKS_PER_FILE batches so
that later reads are purely sequential (no random-access penalty).

After running this script, combine the shards with merge_neighbor_shards.py.

Usage
-----
    python scripts/save_neighbor_shards.py \
        --source-h5 /path/to/neighbours.h5 --output-dir /path/to/neighbor_batches

The script will create the output directory if it does not exist and write files named
    neighbors_shard_0000.h5, neighbors_shard_0001.h5, …
plus shards_complete.json once every shard has been written. It refuses to write into a
directory that already holds shards unless --overwrite is given (which deletes them first),
so shards from different runs are never mixed.

If the run crashes mid-way the current buffer is saved as
    neighbors_shard_XXXX_recovery.h5
so you do not lose progress.
"""

import argparse
import glob
import json
import os
import sys
import traceback
from datetime import datetime

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxy_counter.neighbors import NeighborsDataset, collate_neighbors

SHARD_GLOB = "neighbors_shard_*.h5"
COMPLETE_MARKER = "shards_complete.json"


def save_shard(shard_idx, buffer, save_dir, global_max_neighbors):
    """Write a buffered list of batches to a single HDF5 shard file."""
    if not buffer:
        return

    filename = os.path.join(save_dir, f"neighbors_shard_{shard_idx:04d}.h5")

    all_targets  = torch.cat([b[0] for b in buffer], dim=0).numpy()
    all_samegals = torch.cat([b[1] for b in buffer], dim=0).numpy()

    # Pad each batch's neighbor dimension to the global maximum before concat.
    all_sameins_list = []
    all_masks_list   = []
    for b in buffer:
        neigh_tensor = b[2]  # (B, N_loc, C, H, W)
        mask_tensor  = b[3]  # (B, N_loc)
        B, N_loc, C, H, W = neigh_tensor.shape

        pad_n = global_max_neighbors - N_loc
        if pad_n > 0:
            new_neigh = torch.zeros((B, global_max_neighbors, C, H, W), dtype=neigh_tensor.dtype)
            new_neigh[:, :N_loc] = neigh_tensor
            new_mask = torch.zeros((B, global_max_neighbors), dtype=mask_tensor.dtype)
            new_mask[:, :N_loc] = mask_tensor
            all_sameins_list.append(new_neigh)
            all_masks_list.append(new_mask)
        else:
            all_sameins_list.append(neigh_tensor)
            all_masks_list.append(mask_tensor)

    all_sameins = torch.cat(all_sameins_list, dim=0).numpy()
    all_masks   = torch.cat(all_masks_list,   dim=0).numpy()

    flat_metadata = [item for b in buffer for item in b[4]]
    meta_idxs     = np.array([m["idx"]                for m in flat_metadata])
    meta_num_same = np.array([m["num_same_instrument"] for m in flat_metadata], dtype=np.int32)
    meta_survey   = np.array([m["anchor_survey"].encode("utf-8") for m in flat_metadata])

    print(f"Saving {filename} | shape: {all_targets.shape} ...")
    with h5py.File(filename, "w") as f:
        f.create_dataset("targets",        data=all_targets,  compression="lzf")
        f.create_dataset("samegals",       data=all_samegals, compression="lzf")
        f.create_dataset("sameins",        data=all_sameins,  compression="lzf")
        f.create_dataset("neighbor_masks", data=all_masks,    compression="lzf")
        f.create_dataset("meta_idx",                   data=meta_idxs)
        f.create_dataset("meta_num_same_instrument",   data=meta_num_same)
        f.create_dataset("meta_survey",                data=meta_survey)


def prepare_output_dir(output_dir, overwrite):
    """Create output_dir; refuse (or with overwrite, clear) shards left by a previous run."""
    os.makedirs(output_dir, exist_ok=True)
    old = sorted(glob.glob(os.path.join(output_dir, SHARD_GLOB)))
    marker = os.path.join(output_dir, COMPLETE_MARKER)
    if not old and not os.path.exists(marker):
        return
    if not overwrite:
        sys.exit(f"{output_dir} already contains {len(old)} shard file(s) from a previous run; "
                 "use --overwrite to replace them, or choose another --output-dir.")
    print(f"--overwrite: deleting {len(old)} existing shard file(s) in {output_dir}")
    for path in old + ([marker] if os.path.exists(marker) else []):
        os.remove(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-h5", required=True,
                        help="Neighbours HDF5 read by NeighborsDataset (images_*, neighbor_idx_*, source_type)")
    parser.add_argument("--output-dir", required=True, help="Directory for the output shards")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8, help="Increase for faster random-read throughput")
    parser.add_argument("--chunks-per-file", type=int, default=50,
                        help="Batches per shard (~3200 samples at batch size 64)")
    parser.add_argument("--max-neighbors", type=int, default=5,
                        help="Must match the number of neighbours stored in --source-h5")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete shards already in --output-dir before writing")
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)

    dataset = NeighborsDataset(hdf5_path=args.source_h5, max_neighbors=args.max_neighbors)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_neighbors,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    buffer        = []
    shard_counter = 0
    n_samples     = 0

    print(f"Starting at {datetime.now().isoformat()}")
    print(f"Output: {args.output_dir} | total batches: {len(dataloader)}")

    try:
        for batch in tqdm(dataloader):
            buffer.append(batch)
            n_samples += len(batch[0])
            if len(buffer) >= args.chunks_per_file:
                save_shard(shard_counter, buffer, args.output_dir, dataset.max_neighbors)
                buffer = []
                shard_counter += 1

        if buffer:
            save_shard(shard_counter, buffer, args.output_dir, dataset.max_neighbors)

        total_shards = shard_counter + (1 if buffer else 0)
        print(f"Done at {datetime.now().isoformat()} | wrote {total_shards} shards.")

        # Written last, so its presence means every shard is complete
        with open(os.path.join(args.output_dir, COMPLETE_MARKER), "w") as f:
            json.dump({"num_shards": total_shards, "num_samples": n_samples,
                       "finished": datetime.now().isoformat(), **vars(args)}, f, indent=2)

    except Exception as e:
        if buffer:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("Saving buffered batches to recovery shard ...", file=sys.stderr)
            save_shard(shard_counter, buffer, args.output_dir, dataset.max_neighbors)
            normal_path   = os.path.join(args.output_dir, f"neighbors_shard_{shard_counter:04d}.h5")
            recovery_path = os.path.join(args.output_dir, f"neighbors_shard_{shard_counter:04d}_recovery.h5")
            os.rename(normal_path, recovery_path)
            print(f"Recovery shard saved: {recovery_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
