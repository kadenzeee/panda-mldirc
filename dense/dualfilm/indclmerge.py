#!/usr/bin/env python3

import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser(description="Merge chunk files into final memmaps")
parser.add_argument('-i', '--input', required=True)
args = parser.parse_args()

nbins = 10
npmt  = 8
max_photons = 128

hist_size = nbins * npmt
angle_size = 7

hist_files  = sorted(glob.glob(f"{args.input}/*_HISTS.dat"))
time_files  = sorted(glob.glob(f"{args.input}/*_TIMES.dat"))
angle_files = sorted(glob.glob(f"{args.input}/*_ANGLES.dat"))
label_files = sorted(glob.glob(f"{args.input}/*_LABELS.dat"))

assert len(hist_files) > 0
assert len(hist_files) == len(time_files) == len(angle_files) == len(label_files)

print(f"[INFO] Found {len(hist_files)} chunks")

chunk_sizes = []
for f in hist_files:
    size = os.path.getsize(f)
    entries = size // hist_size
    chunk_sizes.append(entries)

# ------------------------------------------------------------
# PASS 1: count valid events
# ------------------------------------------------------------

valid_counts = []
total_valid = 0

for i in range(len(hist_files)):

    chunk_n = chunk_sizes[i]

    t = np.memmap(time_files[i], dtype=np.float16, mode='r',
                  shape=(chunk_n, max_photons, 2))

    valid_mask = np.any(t != 0, axis=(1,2))
    n_valid = np.sum(valid_mask)

    valid_counts.append(valid_mask)
    total_valid += n_valid

    print(f"[INFO] Chunk {i}: {n_valid}/{chunk_n} valid events")

print(f"[INFO] Total valid events: {total_valid}")

# ------------------------------------------------------------
# allocate final memmaps
# ------------------------------------------------------------

HISTS  = np.memmap("HISTS_full.dat",  dtype=np.int8,    mode='w+', shape=(total_valid, hist_size))
TIMES  = np.memmap("TIMES_full.dat",  dtype=np.float16, mode='w+', shape=(total_valid, max_photons, 2))
ANGLES = np.memmap("ANGLES_full.dat", dtype=np.float16, mode='w+', shape=(total_valid, angle_size))
LABELS = np.memmap("LABELS_full.dat", dtype=np.int8,    mode='w+', shape=(total_valid,))

offset = 0

# ------------------------------------------------------------
# PASS 2: copy only valid events
# ------------------------------------------------------------

for i in range(len(hist_files)):

    print(f"[INFO] Merging chunk {i+1}/{len(hist_files)}")

    chunk_n = chunk_sizes[i]
    valid_mask = valid_counts[i]

    h = np.memmap(hist_files[i],  dtype=np.int8,    mode='r', shape=(chunk_n, hist_size))
    t = np.memmap(time_files[i],  dtype=np.float16, mode='r', shape=(chunk_n, max_photons, 2))
    a = np.memmap(angle_files[i], dtype=np.float16, mode='r', shape=(chunk_n, angle_size))
    l = np.memmap(label_files[i], dtype=np.int8,    mode='r', shape=(chunk_n,))

    n_valid = np.sum(valid_mask)

    HISTS[offset:offset+n_valid]  = h[valid_mask]
    TIMES[offset:offset+n_valid]  = t[valid_mask]
    ANGLES[offset:offset+n_valid] = a[valid_mask]
    LABELS[offset:offset+n_valid] = l[valid_mask]

    offset += n_valid

# ------------------------------------------------------------
# flush
# ------------------------------------------------------------

HISTS.flush()
TIMES.flush()
ANGLES.flush()
LABELS.flush()

print("[INFO] Merge complete")
print(f"[INFO] Final dataset size: {total_valid}")