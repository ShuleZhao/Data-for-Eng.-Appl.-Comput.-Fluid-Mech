"""
Earth Mover's Distance (EMD) loss implementation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


def sinkhorn_iterations(cost_matrix: torch.Tensor, 
                       num_iterations: int = 100, 
                       regularization: float = 0.01) -> torch.Tensor:
    """
    Sinkhorn iterations for approximating optimal transport
    
    Args:
        cost_matrix: Cost matrix (B, N, M)
        num_iterations: Number of Sinkhorn iterations
        regularization: Regularization parameter
        
    Returns:
        transport_matrix: Optimal transport matrix
    """
    B, N, M = cost_matrix.shape
    
    # Initialize dual variables
    u = torch.zeros(B, N, device=cost_matrix.device)
    v = torch.zeros(B, M, device=cost_matrix.device)
    
    # Marginals (uniform distributions)
    mu = torch.ones(B, N, device=cost_matrix.device) / N
    nu = torch.ones(B, M, device=cost_matrix.device) / M
    
    for _ in range(num_iterations):
        # Update u
        u = regularization * (torch.log(mu + 1e-8) - torch.logsumexp(
            (-cost_matrix + v.unsqueeze(1)) / regularization, dim=2))
        
        # Update v
        v = regularization * (torch.log(nu + 1e-8) - torch.logsumexp(
            (-cost_matrix + u.unsqueeze(2)) / regularization, dim=1))
    
    # Compute transport matrix
    transport_matrix = torch.exp((-cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / regularization)
    
    return transport_matrix


def earth_movers_distance(pred: torch.Tensor, 
                         target: torch.Tensor,
                         p: int = 2,
                         num_iterations: int = 100,
                         regularization: float = 0.01) -> torch.Tensor:
    """
    Compute Earth Mover's Distance using Sinkhorn iterations
    
    Args:
        pred: Predicted point cloud (B, N, 3)
        target: Target point cloud (B, M, 3)
        p: p-norm for distance computation
        num_iterations: Number of Sinkhorn iterations
        regularization: Regularization parameter
        
    Returns:
        emd: Earth Mover's Distance
    """
    B, N, D = pred.shape
    _, M, _ = target.shape
    
    # Compute cost matrix (pairwise distances)
    pred_expanded = pred.unsqueeze(2).expand(B, N, M, D)
    target_expanded = target.unsqueeze(1).expand(B, N, M, D)
    
    if p == 2:
        cost_matrix = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)
    elif p == 1:
        cost_matrix = torch.sum(torch.abs(pred_expanded - target_expanded), dim=3)
    else:
        cost_matrix = torch.sum(torch.abs(pred_expanded - target_expanded) ** p, dim=3) ** (1/p)
    
    # Compute optimal transport matrix
    transport_matrix = sinkhorn_iterations(cost_matrix, num_iterations, regularization)
    
    # Compute EMD
    emd = torch.sum(cost_matrix * transport_matrix, dim=(1, 2))  # (B,)
    
    return emd


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance Loss
    
    Args:
        p: p-norm for distance computation (1 or 2)
        num_iterations: Number of Sinkhorn iterations
        regularization: Regularization parameter for Sinkhorn
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, 
                 p: int = 2,
                 num_iterations: int = 100,
                 regularization: float = 0.01,
                 reduction: str = 'mean'):
        super(EMDLoss, self).__init__()
        
        self.p = p
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: EMD loss
        """
        emd = earth_movers_distance(pred, target, self.p, self.num_iterations, self.regularization)
        
        if self.reduction == 'mean':
            return torch.mean(emd)
        elif self.reduction == 'sum':
            return torch.sum(emd)
        else:
            return emd


class ApproximateEMDLoss(nn.Module):
    """
    Approximate EMD Loss using Hungarian algorithm approximation
    
    Args:
        num_samples: Number of samples for approximation
        reduction: Reduction method
    """
    
    def __init__(self, num_samples: int = 1000, reduction: str = 'mean'):
        super(ApproximateEMDLoss, self).__init__()
        
        self.num_samples = num_samples
        self.reduction = reduction
        
    def hungarian_matching(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Approximate Hungarian algorithm for assignment
        
        Args:
            cost_matrix: Cost matrix (N, M)
            
        Returns:
            assignment: Assignment matrix (N, M)
        """
        # Greedy approximation of Hungarian algorithm
        N, M = cost_matrix.shape
        assignment = torch.zeros_like(cost_matrix)
        
        # For each row, find the minimum column
        min_indices = torch.argmin(cost_matrix, dim=1)
        for i, j in enumerate(min_indices):
            assignment[i, j] = 1.0
        
        return assignment
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Approximate EMD loss
        """
        B, N, D = pred.shape
        _, M, _ = target.shape
        
        # Sample points if too many
        if N > self.num_samples:
            pred_indices = torch.randperm(N, device=pred.device)[:self.num_samples]
            pred = pred[:, pred_indices, :]
            N = self.num_samples
            
        if M > self.num_samples:
            target_indices = torch.randperm(M, device=target.device)[:self.num_samples]
            target = target[:, target_indices, :]
            M = self.num_samples
        
        total_loss = 0.0
        
        for b in range(B):
            # Compute cost matrix for this batch
            pred_b = pred[b]  # (N, D)
            target_b = target[b]  # (M, D)
            
            cost_matrix = torch.cdist(pred_b, target_b, p=2) ** 2  # (N, M)
            
            # Greedy assignment (approximation of Hungarian)
            assignment = self.hungarian_matching(cost_matrix)
            
            # Compute loss
            loss_b = torch.sum(cost_matrix * assignment)
            total_loss += loss_b
        
        if self.reduction == 'mean':
            return total_loss / B
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / B


class MultiScaleEMDLoss(nn.Module):
    """
    Multi-scale EMD Loss
    
    Args:
        scales: List of scales for subsampling
        weights: Weights for each scale
        reduction: Reduction method
    """
    
    def __init__(self, 
                 scales: list = [1.0, 0.5, 0.25],
                 weights: Optional[list] = None,
                 reduction: str = 'mean'):
        super(MultiScaleEMDLoss, self).__init__()
        
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
        # Create EMD loss for each scale
        self.emd_losses = nn.ModuleList([
            ApproximateEMDLoss(num_samples=min(500, int(2048 * scale)), reduction=reduction)
            for scale in scales
        ])
        
    def subsample_points(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Subsample points according to scale
        
        Args:
            points: Input points (B, N, 3)
            scale: Subsampling scale
            
        Returns:
            subsampled_points: Subsampled points
        """
        if scale >= 1.0:
            return points
        
        B, N, D = points.shape
        num_points = int(N * scale)
        
        # Farthest point sampling for better coverage
        indices = self.farthest_point_sample(points, num_points)
        return torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, D))
    
    def farthest_point_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest point sampling
        
        Args:
            points: Input points (B, N, 3)
            num_samples: Number of samples
            
        Returns:
            indices: Sampled indices (B, num_samples)
        """
        B, N, D = points.shape
        device = points.device
        
        # Initialize
        centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(num_samples):
            centroids[:, i] = farthest
            centroid = points[torch.arange(B, device=device), farthest, :].view(B, 1, D)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, dim=1)
        
        return centroids
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Multi-scale EMD loss
        """
        total_loss = 0.0
        
        for scale, weight, emd_loss in zip(self.scales, self.weights, self.emd_losses):
            # Subsample both point clouds
            pred_sub = self.subsample_points(pred, scale)
            target_sub = self.subsample_points(target, scale)
            
            # Compute loss at this scale
            scale_loss = emd_loss(pred_sub, target_sub)
            total_loss += weight * scale_loss
        
        return total_loss


class CombinedEMDChamferLoss(nn.Module):
    """
    Combined EMD and Chamfer Distance Loss
    
    Args:
        emd_weight: Weight for EMD loss
        chamfer_weight: Weight for Chamfer loss
        reduction: Reduction method
    """
    
    def __init__(self, 
                 emd_weight: float = 1.0,
                 chamfer_weight: float = 1.0,
                 reduction: str = 'mean'):
        super(CombinedEMDChamferLoss, self).__init__()
        
        self.emd_weight = emd_weight
        self.chamfer_weight = chamfer_weight
        
        from .chamfer import ChamferLoss
        self.emd_loss = ApproximateEMDLoss(reduction=reduction)
        self.chamfer_loss = ChamferLoss(reduction=reduction)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Combined EMD and Chamfer loss
        """
        emd_loss = self.emd_loss(pred, target)
        chamfer_loss = self.chamfer_loss(pred, target)
        
        return self.emd_weight * emd_loss + self.chamfer_weight * chamfer_loss