import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import numpy as np


# NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: [batch_size, projection_dim]
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate all embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2*batch_size, 2*batch_size]

        # Create mask to remove self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ], dim=0)  # [2*batch_size]

        # Compute log-sum-exp for all negatives
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        loss = loss.mean()

        return loss