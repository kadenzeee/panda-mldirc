#!/usr/bin/env python
'''
Defines and trains GNN model for PANDA Barrel DIRC.

Run from CLI:
python panda_gnn.py -i dataset.pkl -o models
'''

import torch

# ----- MLP Layer ----- #

import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PandaGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')    # Sum aggregation; other options include 'mean' and 'max'
    
        # MLP encoder for edges 
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MLP encoder for nodes
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i = reciever node
        # x_j = sender node
        
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(m)
    
    def update(self, aggr_out, x):
        # aggr_out = sum of incoming messages
        
        h = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(h)


# ----- Model Architecture Definition ----- #

from torch_geometric.nn import global_mean_pool

class PandaGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, n_classes):
        super().__init__()
        
        self.gnn1 = PandaGNNLayer(node_dim, edge_dim, hidden_dim)
        self.gnn2 = PandaGNNLayer(hidden_dim, edge_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        
        x = self.gnn1(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        x = self.gnn2(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        # Pool
        x = global_mean_pool(x, batch)
        g = data.global_features
        g = g.view(data.num_graphs, -1) # reshape because PyTorch batching concatenates global features lazily, not per graph
        
        x = torch.cat([x, g], dim=-1)
        
        return self.classifier(x)

if __name__ == "__main__":

    # ----- CLI ----- #
    
    import argparse
    
    parser = argparse.ArgumentParser(prog='panda_gnn', description='Defines and trains GNN model for PANDA Barrel DIRC.')
    
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input pickle file.')
    parser.add_argument('-o', '--output', type=str, required=False, default='tmp', help='Path to output model file')
    
    args = parser.parse_args()
    
    # ----- Load ----- #
    
    import pickle
    from torch_geometric.data import Data

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    nevents = len(data['labels'])

    dataset = []

    for i in range(nevents):

        x           = torch.tensor(data['nodes'][i], dtype=torch.float)
        edge_index  = torch.tensor(data['edges'][i].T, dtype=torch.long)
        edge_attr   = torch.tensor(data['edge_features'][i], dtype=torch.float)

        y   = torch.tensor([data['labels'][i]], dtype=torch.long)
        g   = torch.tensor(data['globals'][i], dtype=torch.float) 

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.global_features = g
        
        dataset.append(graph)
    
    
# ----- Training Loop ----- #
# -----      Save     ----- #

    from torch_geometric.loader import DataLoader
    import os,subprocess,datetime
    
    n = 1
    while True:
        outdir = f'models/{args.output}_{n}__{datetime.date.today().strftime('%Y-%m-%d')}'
        if not os.path.exists(outdir):
            break
        n += 1
    
    subprocess.run(f"mkdir -p {outdir}", shell=True)
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = PandaGNN(node_dim=3, edge_dim=3, global_dim=4, hidden_dim=64, n_classes=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        total_loss = 0
        
        for batch in loader:
            optimiser.zero_grad()
            
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            
            loss.backward()
            optimiser.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
        torch.save(model.state_dict(), f"{outdir}/weights__epoch-{epoch}__loss-{total_loss:.4f}.pt")
    
    with open(f"{outdir}/run_info.txt", "w") as f:
        f.write(str(model))
        f.write("\n\n")

        f.write(f"Batch size: {loader.batch_size}\n")
        f.write(f"Num events: {len(dataset)}\n")

