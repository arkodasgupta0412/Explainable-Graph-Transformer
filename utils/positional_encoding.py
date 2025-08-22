import numpy as np
import scipy.sparse as sp
import torch

def laplacian_positional_encoding(data, pos_enc_dim):
    
    """
    Computes Laplacian positional encodings for nodes using eigenvectors
    of the normalized Laplacian.
    """

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    row, col = edge_index.cpu().numpy()
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes)).astype(float)
    
    degrees = np.array(adj.sum(1)).flatten()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(np.power(degrees, -0.5))
    L = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1].real).float()
    
    data.lap_pos_enc = lap_pos_enc.to(data.x.device)
    return data
