#!/usr/bin/env python3

'''
Converts prtdirc simulation data into a format suitable for training a the dualfilm NN.
Designed to be run in parallel on a cluster, with each job processing a chunk of the data and writing to memmaps.
'''

import os
import subprocess
import ROOT             #   type:ignore
import numpy as np
import argparse
import time
import platform

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="Parallel ROOT → dualfilm converter")

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=False, help="Output file prefix (default: same as input)", default=None)
parser.add_argument('-n', '--nbins', type=int, default=10)
parser.add_argument('-tsmear', '--tsmear', type=float, default=0.1)
parser.add_argument('-asmear', '--asmear', type=float, default=3E-03)

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)

parser.add_argument('--shuffle', action='store_true')

args = parser.parse_args()

# ------------------------------------------------------------
# ROOT setup
# ------------------------------------------------------------

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')

libbase = "../../../prtdirc/build/libPrt"
libpath = libbase + (".dylib" if platform.system()=="Darwin" else ".so")
ROOT.gSystem.Load(libpath)

t = ROOT.PrtTools(args.input)

entries_total = t.entries()
start_entry = args.start
end_entry = args.end if args.end is not None else entries_total

entries = end_entry - start_entry

print(f"[INFO] Processing entries {start_entry} → {end_entry}")
print(f"[INFO] Total in this job: {entries}")

# ------------------------------------------------------------
# Detector constants
# ------------------------------------------------------------

npmt  = t.npmt()
npix  = t.npix()
nchan = npmt * npix
nbins = args.nbins

max_photons = 128

tedges = np.linspace(-50, 50, nbins+1)
mcpedges = np.arange(npmt+1) * npix

# ------------------------------------------------------------
# Allocate memmaps
# ------------------------------------------------------------

if args.output is not None:
    subprocess.run(["mkdir", "-p", args.output], check=True)

base = os.path.basename(args.input).replace(".root","")
tag  = f"{args.output}/{base}_dualfilm_{args.tsmear}ns_{start_entry}_{end_entry}"

HISTS  = np.memmap(f"{tag}_HISTS.dat",  dtype=np.int8,    mode='w+', shape=(entries, nbins*npmt))
TIMES  = np.memmap(f"{tag}_TIMES.dat",  dtype=np.float16, mode='w+', shape=(entries, max_photons, 2))
ANGLES = np.memmap(f"{tag}_ANGLES.dat", dtype=np.float16, mode='w+', shape=(entries, 7))
LABELS = np.memmap(f"{tag}_LABELS.dat", dtype=np.int8,    mode='w+', shape=(entries,))

# ------------------------------------------------------------
# Event Loop
# ------------------------------------------------------------

write_index = 0
t0_global = time.time()

while t.next():

    i = t.i()

    if i < start_entry:
        continue
    if i >= end_entry:
        break

    event = t.event()
    hits  = event.getHits()


    if not hits:
        continue

    nh = len(hits)


    times = np.empty(nh, dtype=np.float32)
    chs   = np.empty(nh, dtype=np.int32)

    for k, photon in enumerate(hits):
        times[k] = photon.getLeadTime()
        chs[k]   = photon.getChannel()

    # Smearing (vectorised)
    if args.tsmear > 0:
        times += np.random.normal(0, args.tsmear, nh)

    # Basic time stats
    mu  = times.mean()
    std = times.std()
    tmin = times.min()
    tmax = times.max()

    times -= mu
    
    phits = np.vstack((times, chs)).T
    
    for k, hit in enumerate(phits):
        if k < max_photons:
            TIMES[write_index][k]  = phits[k]
        else:
            break
    
    # 2D histogram
    hist2d = np.histogram2d(times, chs, bins=[tedges, mcpedges])[0]
    
    # Momentum
    momentum = np.sqrt(event.getMomentum()[0]**2 + event.getMomentum()[1]**2 + event.getMomentum()[2]**2)
    theta = np.arccos(event.getMomentum()[2]/momentum) + 22*np.pi/180 # add polar angle of bar normal to get absolute polar angle of track entry into radiator
    if event.getMomentum()[0] != 0:
        phi   = np.arctan(event.getMomentum()[1]/event.getMomentum()[0])
    else:
        phi = 0.0
    
    theta += np.random.normal(0, args.asmear)
    phi   += np.random.normal(0, args.asmear)
    
    vecmomentum = [momentum*np.sin(theta)*np.cos(phi), momentum*np.sin(theta)*np.sin(phi), momentum*np.cos(theta)] 

    # Write directly to memmaps
    HISTS[write_index]  = hist2d.flatten().astype(np.int8)
    ANGLES[write_index] = [mu, std, tmin, tmax] + vecmomentum
    LABELS[write_index] = t.pid() - 2 # PID 2 = pion, 3 = kaon -> map to 0,1 for binary classification
    
    write_index += 1

valid_events = write_index

# ------------------------------------------------------------
# Shrink memmaps to valid events only
# ------------------------------------------------------------

if valid_events < entries:
    print(f"[INFO] Shrinking memmaps from {entries} → {valid_events} valid events")

    # create new memmaps for cleaned dataset
    HISTS_clean  = np.memmap(f"{tag}_HISTS.dat",  dtype=np.int8,    mode='w+', shape=(valid_events, nbins*npmt))
    TIMES_clean  = np.memmap(f"{tag}_TIMES.dat",  dtype=np.float16, mode='w+', shape=(valid_events, max_photons, 2))
    ANGLES_clean = np.memmap(f"{tag}_ANGLES.dat", dtype=np.float16, mode='w+', shape=(valid_events, 7))
    LABELS_clean = np.memmap(f"{tag}_LABELS.dat", dtype=np.int8,    mode='w+', shape=(valid_events,))

    # copy only the valid portion
    HISTS_clean[:]  = HISTS[:valid_events]
    TIMES_clean[:]  = TIMES[:valid_events]
    ANGLES_clean[:] = ANGLES[:valid_events]
    LABELS_clean[:] = LABELS[:valid_events]

    # flush changes to disk
    HISTS_clean.flush()
    TIMES_clean.flush()
    ANGLES_clean.flush()
    LABELS_clean.flush()

    # optional: delete the old oversized memmaps from memory
    del HISTS, TIMES, ANGLES, LABELS

    # rename the cleaned memmaps to original variables if needed
    HISTS, TIMES, ANGLES, LABELS = HISTS_clean, TIMES_clean, ANGLES_clean, LABELS_clean

# ------------------------------------------------------------
# Optional Shuffle
# ------------------------------------------------------------

if args.shuffle:
    print("[INFO] Shuffling...")
    perm = np.random.permutation(LABELS.shape[0])
    HISTS[:]  = HISTS[perm]
    TIMES[:]  = TIMES[perm]
    ANGLES[:] = ANGLES[perm]
    LABELS[:] = LABELS[perm]

HISTS.flush()
TIMES.flush()
ANGLES.flush()
LABELS.flush()

print(f"[INFO] Done in {time.time() - t0_global:.2f} seconds")
print(f"[INFO] Written: {LABELS.shape[0]} events")
