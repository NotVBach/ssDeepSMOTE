import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf


# Random permutation mapping every index to a different index with the same label
def same_class_derangement(labels):
    perm = torch.arange(labels.numel(), device=labels.device)
    for value in torch.unique(labels):
        members = torch.nonzero(labels == value, as_tuple=True)[0]
        if members.numel() > 1:
            order = members[torch.randperm(members.numel(), device=labels.device)]
            perm[order] = order.roll(-1)
    return perm


# Mixing weights lambda ~ U(low, high), one per row, shaped (n, 1)
def sample_lambda(n, low, high, device):
    return torch.empty(n, 1, device=device).uniform_(float(low), float(high))


# SMOTE interpolation
def interpolate(a, b, lam):
    return a.float() + lam * (b.float() - a.float())


# Indices of the k nearest rows of z by cosine similarity, excluding each row itself
def cosine_knn(z, k):
    zn = F.normalize(z.float(), dim=1)
    sim = zn @ zn.T
    sim.fill_diagonal_(float("-inf"))
    return sim.topk(min(int(k), z.shape[0] - 1), dim=1).indices


# Highest cosine similarity between each query row and any reference row
def max_cosine(query, reference):
    q = F.normalize(query.float(), dim=1)
    r = F.normalize(reference.float(), dim=1)
    return (q @ r.T).max(dim=1).values


class ClassGaussian:
    # Fit a Gaussian to one class's latents with Ledoit-Wolf shrinkage toward a scaled identity
    def __init__(self, z):
        estimator = LedoitWolf().fit(np.asarray(z, dtype=np.float64))
        self.mean = estimator.location_
        self.precision = estimator.precision_

    # Mahalanobis distance of every row of z to the class mean
    def distance(self, z):
        diff = np.asarray(z, dtype=np.float64) - self.mean
        squared = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return np.sqrt(np.maximum(squared, 0.0))
