import torch.nn as nn

class WrappedGTNet(nn.Module):

    """
    Wrapper for GraphTransformerNet so it can be used with Explainers.
    Allows manipulation of x and edge_index independently.
    """
    
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data  

    def forward(self, x, edge_index):
        data = self.data.clone()
        data.x = x
        data.edge_index = edge_index
        return self.model(data)
