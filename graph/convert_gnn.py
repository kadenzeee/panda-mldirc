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
    
    # ------ PID ----- #
    
    pid = f.event().getPid() - 2 # minus 2 because prtdirc simulation labels Pi+ : 2 and Kaon+ : 3
    
    # ------ Track ----- #
    
    track_pos   = f.event().getPosition()
    t0          = 0              # impingement time (always zero in simulation)
    p           = f.event().getMomentum()
    
    track_features = np.array([
        track_pos[0],
        track_pos[1],
        track_pos[2],
        t0,
        p[0],
        p[1],
        p[2]
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

def build_edges(nodes, radius=50.0, alpha=1.0):
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
            
            distsq = dx*dx + dy*dy + alpha * dt*dt

            if distsq < radius*radius:
                edges.append([i, j])
                edge_features.append([dx, dy, dt])
                
    return np.array(edges), np.array(edge_features)

all_edges = []
all_edge_features = []

for nodes in all_hits:
    e, ef = build_edges(nodes)
    all_edges.append(e)
    all_edge_features.append(ef)

# ----- Save to .pkl ----- #

data = {
    "nodes": all_hits,                  # list of (Ni, 3)
    "edges": all_edges,                 # list of (Ei, 2)
    "edge_features": all_edge_features, # list of (Ei, 3)
    "globals": all_globals,             # (N_events, Fg)
    "labels": all_labels                # (N_events,)
}

import pickle

outfilename = args.output if args.output is not None else "gnn_data.pkl"
with open(outfilename, "wb") as f:
    pickle.dump(data, f)