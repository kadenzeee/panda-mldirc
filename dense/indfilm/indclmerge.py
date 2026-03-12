#!/usr/bin/env python3

import numpy as np
import os
import glob
import argparse

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="Merge chunk files into final memmaps")
parser.add_argument('-i', '--input', required=True, help="Input directory containing chunk files")
args = parser.parse_args()

# ------------------------------------------------------------
# Configure shapes
# ------------------------------------------------------------

nbins = 10        # must match training config
npmt  = 8         # adjust if needed
npix  = 64        # adjust if needed

max_photons = 128

hist_size = nbins * npmt
angle_size = 7

# ------------------------------------------------------------
# Find chunk files
# ------------------------------------------------------------

hist_files  = sorted(glob.glob(f"{args.input}/*_HISTS.dat"))
time_files  = sorted(glob.glob(f"{args.input}/*_TIMES.dat"))
angle_files = sorted(glob.glob(f"{args.input}/*_ANGLES.dat"))
label_files = sorted(glob.glob(f"{args.input}/*_LABELS.dat"))

assert len(hist_files) > 0, "No chunk files found"
assert len(hist_files) == len(time_files) == len(angle_files) == len(label_files)

print(f"[INFO] Found {len(hist_files)} chunks")

chunk_sizes = []
for f in hist_files:
    size = os.path.getsize(f)
    entries = size // hist_size   # int8 = 1 byte
    chunk_sizes.append(entries)

total_entries = sum(chunk_sizes)

print(f"[INFO] Total entries: {total_entries}")

# ------------------------------------------------------------
# final memmaps
# ------------------------------------------------------------

HISTS  = np.memmap("HISTS_full.dat",  dtype=np.int8,    mode='w+', shape=(total_entries, hist_size))
TIMES  = np.memmap("TIMES_full.dat",  dtype=np.float16, mode='w+', shape=(total_entries, max_photons, 2))
ANGLES = np.memmap("ANGLES_full.dat", dtype=np.float16, mode='w+', shape=(total_entries, angle_size))
LABELS = np.memmap("LABELS_full.dat", dtype=np.int8,    mode='w+', shape=(total_entries,))


offset = 0

for i in range(len(hist_files)):

    print(f"[INFO] Merging chunk {i+1}/{len(hist_files)}")

    chunk_n = chunk_sizes[i]

    h = np.memmap(hist_files[i],  dtype=np.int8,    mode='r', shape=(chunk_n, hist_size))
    t = np.memmap(time_files[i],  dtype=np.float16, mode='r', shape=(chunk_n, max_photons, 2))
    a = np.memmap(angle_files[i], dtype=np.float16, mode='r', shape=(chunk_n, angle_size))
    l = np.memmap(label_files[i], dtype=np.int8,    mode='r', shape=(chunk_n,))

    HISTS[offset:offset+chunk_n]  = h
    TIMES[offset:offset+chunk_n]  = t
    ANGLES[offset:offset+chunk_n] = a
    LABELS[offset:offset+chunk_n] = l

    offset += chunk_n

# ------------------------------------------------------------
# Flush
# ------------------------------------------------------------

HISTS.flush()
TIMES.flush()
ANGLES.flush()
LABELS.flush()

print("[INFO] Merge complete")
