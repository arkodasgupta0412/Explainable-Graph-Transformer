import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, to_undirected

from utils.positional_encoding import laplacian_positional_encoding
from utils.graph_utils import make_full_graph


def load_dataset(dataset_name):

    """
    Loads datasets from PyG.
    Supports Planetoid, WikipediaNetwork, Actor, Amazon, Coauthor.
    """

    root = './data'
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=dataset_name, transform=NormalizeFeatures())
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=root, name=dataset_name, transform=NormalizeFeatures())
    elif dataset_name == 'Actor':
        dataset = Actor(root=root, transform=NormalizeFeatures())
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=root, name=dataset_name, transform=NormalizeFeatures())
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=root, name=dataset_name, transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset[0], dataset_name



def prepare_data(dataset_name, pos_enc_dim=8, make_full=False):
    """
    Loads dataset, creates splits, adds positional encodings, 
    and optionally makes the graph fully connected.
    """
    
    data, name = load_dataset(dataset_name)

    if dataset_name not in ["Cora", "Citeseer", "PubMed"]:
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            split1 = int(num_nodes * 0.6)
            split2 = int(num_nodes * 0.8)

            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            data.train_mask[indices[:split1]] = True
            data.val_mask[indices[split1:split2]] = True
            data.test_mask[indices[split2:]] = True

    data.edge_index = to_undirected(data.edge_index)
    data.edge_index = add_self_loops(data.edge_index)[0]

    if make_full:
        data = make_full_graph(data)

    data = laplacian_positional_encoding(data, pos_enc_dim)

    return data
