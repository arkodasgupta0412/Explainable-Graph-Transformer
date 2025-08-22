import torch
from torch_geometric.utils import to_undirected

def make_full_graph(g):

    """
    Converts input graph to a fully connected graph.
    Ensures undirectedness and removes duplicate edges.
    """

    num_nodes = g.num_nodes
    full_edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    full_edge_index = torch.cat([g.edge_index, full_edge_index], dim=1)
    full_edge_index = torch.unique(full_edge_index, dim=1)
    full_edge_index = to_undirected(full_edge_index)
    g.edge_index = full_edge_index
    
    return g
