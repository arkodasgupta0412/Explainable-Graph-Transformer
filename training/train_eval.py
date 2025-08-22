import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error


def train(model, data, optimizer, device):
    """
    Trains the model for one step using cross-entropy loss.
    """
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, data, mask):
    """
    Evaluates accuracy and mean absolute error on a given mask (train/val/test).
    """

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).float().mean().item()
        mae = mean_absolute_error(data.y[mask].cpu(), pred[mask].cpu())
        
    return acc, mae
