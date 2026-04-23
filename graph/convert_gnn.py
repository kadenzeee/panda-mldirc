#!/usr/bin/env python3
#filepath: mldirc/graph/convert_gnn.py

'''
Converts prtdirc simulation data into a format suitable for training a GNN.
'''

import argparse 
import platform
import time
import ROOT             # type: ignore
import numpy as np

program_start = time.time()

# ----- CLI ----- #

parser = argparse.ArgumentParser(prog='convert_gnn',
                                 description='Converts prtdirc simulation data into a format suitable for training a GNN.')

parser.add_argument('-i', '--input', type=str, required=True, help='Path to input ROOT file.')
parser.add_argument('-o', '--output', type=str, required=False, help='Path to output .npz file')
parser.add_argument('-tadj', '--threshold-time-adjacency', type = float, default=15.0, help='Time threshold for edge adjacency of photons (ns). Default: 50.0')
parser.add_argument('-radj', '--threshold-radial-adjacency', type=float, default=350.0, help='Radial distance threshold for edge adjacency of photons (mm). Default: 350.0')
parser.add_argument('--verbose', action='store_true', help='Print verbose output during edge construction.')
parser.add_argument('--save-sparsities', action='store_true', help='Save sparsity of each event to sparsities.csv')

args = parser.parse_args()

ROOT.gInterpreter.ProcessLine('#include "../../prttools/PrtTools.h"')

libbase = "../../prtdirc/build/libPrt"
libpath = libbase + (".dylib" if platform.system()=="Darwin" else ".so")
ROOT.gSystem.Load(libpath)

# ----- Load ----- #

f = ROOT.PrtTools(args.input)
entries = f.entries()

# ----- Run ----- #

all_hits    = []
all_globals = []
all_labels  = []

while f.next() and len(all_hits) < entries:
    
    if not f.event().getHits():
        continue
    
    if f.event().getMomentum()[0] == 0 or f.event().getMomentum()[1] == 0 or f.event().getMomentum()[2] == 0:
        continue
    
    # ------ PID ----- #
    
    pid = f.event().getPid() - 2 # minus 2 because prtdirc simulation labels Pi+ : 2 and Kaon+ : 3
    
    # ------ Track ----- #
    
    track_pos   = f.event().getPosition()
    t0          = 0              # impingement time (always zero in simulation)
    p           = f.event().getMomentum()
    mag_p       = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    
    track_features = np.array([
        track_pos[0],
        track_pos[1],
        track_pos[2],
        t0,
        mag_p,
        p[0]/mag_p,
        p[1]/mag_p,
        p[2]/mag_p
    ])
    
    # ----- Hits ----- # 
    
    hits = f.event().getHits()
    
    node_features = []
    
    for h in hits:
        x = h.getPosition()[0]
        y = h.getPosition()[1]
        t = h.getLeadTime()
        
        # Relative features to track impingement position
        node_features.append([
            x - track_pos[0],
            y - track_pos[1],
            t - t0
        ])
    
    if len(node_features) < 2:
        continue
    
    node_features = np.array(node_features)
    
    all_hits.append(node_features)
    all_globals.append(track_features)
    all_labels.append(pid)

# ----- Edges ----- #

def build_edges(nodes, radj=args.threshold_radial_adjacency,  tadj=args.threshold_time_adjacency):
    edges = []
    edge_features = []
    
    N = len(nodes)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            
            dx = nodes[j][0] - nodes[i][0]
            dy = nodes[j][1] - nodes[i][1]
            dt = nodes[j][2] - nodes[i][2]
            
            distsq = dx*dx + dy*dy

            if distsq < radj*radj and abs(dt) < tadj:
                edges.append([i, j])
                edge_features.append([dx, dy, dt])
                
    return np.array(edges), np.array(edge_features)

all_edges = []
all_edge_features = []

for nodes in all_hits:
    e, ef = build_edges(nodes)
    all_edges.append(e)
    all_edge_features.append(ef)

if args.verbose:
    for event in range(len(all_hits)):
        print(f'Event {event}: {len(all_hits[event])} nodes, {len(all_edges[event])} edges')
        print(f'{len(all_hits[event]) * (len(all_hits[event]) - 1) - len(all_edges[event])} edges deleted out of {len(all_hits[event]) * (len(all_hits[event]) - 1)} possible edges ({100.0 * (1 - len(all_edges[event]) / (len(all_hits[event]) * (len(all_hits[event]) - 1))):.2f}% sparsity)')
        print('---')

if args.save_sparsities:
    with open(f'sparsities-tadj{args.threshold_time_adjacency}-radj{args.threshold_radial_adjacency}.csv', 'w') as f:
        f.write('event,sparsity\n')
        for event in range(len(all_hits)):
            sparsity = 100.0 * (1 - len(all_edges[event]) / (len(all_hits[event]) * (len(all_hits[event]) - 1)))
            f.write(f'{event},{sparsity:.2f}\n')

# ----- Save to .pkl ----- #

data = {
    "nodes": all_hits,                  # list of (Ni, 3)
    "edges": all_edges,                 # list of (Ei, 2)
    "edge_features": all_edge_features, # list of (Ei, 3)
    "globals": all_globals,             # (N_events, Fg)
    "labels": all_labels                # (N_events,)
}

import pickle, json

outfilename = args.output if args.output is not None else "gnn_data.pkl"
with open(outfilename, "wb") as f:
    pickle.dump(data, f)

with open(outfilename.replace('.pkl', '_header.json'), 'w') as f:
    json.dump({
        "nnodes" : [len(hits) for hits in all_hits],
        "nedges" : [len(edges) for edges in all_edges],
        "nevents" : len(all_hits)
    }, f)