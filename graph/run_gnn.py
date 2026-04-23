#!/usr/bin/env python

'''
Imports a trained GNN model and evaluates it on a given test dataset, printing the accuracy.
'''


import torch
from torch_geometric.loader import DataLoader
from panda_gnn import PandaGNN

def evaluate(model, dataset, batch_size=1024):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            
            preds = out.argmax(dim=1)
            
            labels = batch.y.view(-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, all_preds, all_labels



if __name__ == "__main__":
    
    # ----- CLI ----- #
    
    import argparse
    
    parser = argparse.ArgumentParser(prog='run_gnn', description='Evaluates GNN on given test data.')
    
    parser.add_argument('-im', '--model_input', type=str, required=True, help='Path to input model weights.')
    parser.add_argument('-id', '--data_input', type=str, required=True, help='Path to input .pkl file to run tests on.')
    
    args = parser.parse_args()
    
    # ----- Load ----- #
    
    import pickle
    from torch_geometric.data import Data

    with open(args.data_input, "rb") as f:
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
    
    model = PandaGNN(node_dim=3, edge_dim=3, global_dim=8, hidden_dim=64, n_classes=2)
    model = torch.compile(model)
    model.load_state_dict(torch.load(args.model_input))
    
    accuracy, all_preds, all_labels = evaluate(model, dataset)