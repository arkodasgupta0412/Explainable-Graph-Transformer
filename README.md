## Explainable Graph Transformer

The graph Transformer emerges as a new architecture and has shown superior performance on various graph mining tasks. However, the major bottleneck for them is the time taken for the self-attention scores computation. The quadratic time complexity on the number of nodes is a major computation overhead that limits usage of graph transformer to large and dense graphs.

Several innovative solutions have been proposed on this domain to reduce this bottleneck. Some of the seminal works include:

1. NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs
2. SGFormer: Simplified Graph Transformers
3. DualFormer: Dual Graph Transformer

In our approach, we try to reduce the attention time complexity with the help of GNN Explainability paradigm.
GNNExplainer is the first general, model-agnostic approach for providing interpretable explanations for predictions of any GNN-based model on any graph-based machine learning task. Given an instance, GNNExplainer identifies a compact subgraph structure and a small subset of node features that have a crucial role in GNN's prediction.

We tried to identify this compact subgraph which eliminated nodes that had significantly lower contribution to
attention scores. In this way, we do not have to deal with all the node-pair tokens of the given dataset.

The above method surpassed SoTA metrics in two of the most popular graph datasets. (Chameleon and CS)
