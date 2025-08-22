import random
import torch
from collections import defaultdict
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig, ExplanationType
from models.wrapped_model import WrappedGTNet


def get_pruned_edges(model, data, num_epochs=100, max_explain_nodes=100, sampling='proportional', topk_percent=10):

    """
    Uses GNNExplainer to compute edge importance scores and prune the graph.
    Keeps only the top-k% most important edges.
    """

    model.eval()
    wrapped_gtnet = WrappedGTNet(model, data)

    explainer = Explainer(
        model=wrapped_gtnet,
        algorithm=GNNExplainer(epochs=num_epochs, lr=0.1),
        explanation_type=ExplanationType.model,
        edge_mask_type='object',
        node_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        )
    )

    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    train_labels = data.y[train_nodes]

    class_to_nodes = defaultdict(list)
    for idx, label in zip(train_nodes.tolist(), train_labels.tolist()):
        class_to_nodes[label].append(idx)

    total_train = len(train_nodes)
    classes = sorted(class_to_nodes.keys())
    explain_node_indices = []

    for cls in classes:
        candidates = class_to_nodes[cls]
        num_samples = (len(candidates) // total_train * max_explain_nodes) if sampling == 'proportional' else (max_explain_nodes // len(classes))
        num_samples = min(num_samples, len(candidates))
        explain_node_indices.extend(random.sample(candidates, num_samples))

    edge_mask_sum = torch.zeros(data.edge_index.size(1), device=data.edge_index.device)

    for node_idx in explain_node_indices:
        explanation = explainer(x=data.x, edge_index=data.edge_index, index=int(node_idx))
        edge_mask = explanation.edge_mask
        if edge_mask.max().item() > 0:
            edge_mask = edge_mask / edge_mask.max()
        edge_mask_sum += edge_mask

    edge_mask_avg = edge_mask_sum / len(explain_node_indices)
    num_edges_to_keep = int((topk_percent / 100.0) * edge_mask_avg.numel())
    topk_idxs = torch.topk(edge_mask_avg, k=num_edges_to_keep).indices
    important_edge_mask = torch.zeros_like(edge_mask_avg, dtype=torch.bool)
    important_edge_mask[topk_idxs] = True

    return data.edge_index[:, important_edge_mask]


def prune_graph(data, pruned_edge_index):

    """Replaces original graph edges with pruned edges."""

    pruned_data = data.clone()
    pruned_data.edge_index = pruned_edge_index

    return pruned_data
