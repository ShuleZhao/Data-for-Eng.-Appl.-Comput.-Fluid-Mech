"""
Chamfer Distance loss implementation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def chamfer_distance_squared(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute squared Chamfer Distance between two point clouds
    
    Args:
        pred: Predicted point cloud (B, N, 3)
        target: Target point cloud (B, M, 3)
        
    Returns:
        dist1: Distance from pred to target (B, N)
        dist2: Distance from target to pred (B, M)
    """
    B, N, D = pred.shape
    _, M, _ = target.shape
    
    # Compute pairwise distances
    # pred: (B, N, 1, D), target: (B, 1, M, D)
    pred_expanded = pred.unsqueeze(2).expand(B, N, M, D)
    target_expanded = target.unsqueeze(1).expand(B, N, M, D)
    
    # Squared L2 distance
    distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=3)  # (B, N, M)
    
    # Find minimum distances
    dist1, _ = torch.min(distances, dim=2)  # (B, N) - distance from pred to target
    dist2, _ = torch.min(distances, dim=1)  # (B, M) - distance from target to pred
    
    return dist1, dist2


class ChamferLoss(nn.Module):
    """
    Chamfer Distance Loss
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
        use_sqrt: Whether to use square root (L2) or squared distance
        forward_weight: Weight for forward direction (pred -> target)
        backward_weight: Weight for backward direction (target -> pred)
    """
    
    def __init__(self, 
                 reduction: str = 'mean',
                 use_sqrt: bool = False,
                 forward_weight: float = 1.0,
                 backward_weight: float = 1.0):
        super(ChamferLoss, self).__init__()
        
        self.reduction = reduction
        self.use_sqrt = use_sqrt
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Chamfer distance loss
        """
        dist1, dist2 = chamfer_distance_squared(pred, target)
        
        if self.use_sqrt:
            dist1 = torch.sqrt(dist1 + 1e-8)
            dist2 = torch.sqrt(dist2 + 1e-8)
        
        # Weighted sum of both directions
        loss = self.forward_weight * dist1 + self.backward_weight * dist2
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class MultiScaleChamferLoss(nn.Module):
    """
    Multi-scale Chamfer Distance Loss
    
    Args:
        scales: List of scales for subsampling
        weights: Weights for each scale
        reduction: Reduction method
        use_sqrt: Whether to use square root
    """
    
    def __init__(self, 
                 scales: list = [1.0, 0.5, 0.25],
                 weights: Optional[list] = None,
                 reduction: str = 'mean',
                 use_sqrt: bool = False):
        super(MultiScaleChamferLoss, self).__init__()
        
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
        # Create Chamfer loss for each scale
        self.chamfer_losses = nn.ModuleList([
            ChamferLoss(reduction=reduction, use_sqrt=use_sqrt)
            for _ in scales
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
        
        # Random subsampling
        indices = torch.randperm(N, device=points.device)[:num_points]
        return points[:, indices, :]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Multi-scale Chamfer distance loss
        """
        total_loss = 0.0
        
        for scale, weight, chamfer_loss in zip(self.scales, self.weights, self.chamfer_losses):
            # Subsample both point clouds
            pred_sub = self.subsample_points(pred, scale)
            target_sub = self.subsample_points(target, scale)
            
            # Compute loss at this scale
            scale_loss = chamfer_loss(pred_sub, target_sub)
            total_loss += weight * scale_loss
        
        return total_loss


class SymmetricChamferLoss(nn.Module):
    """
    Symmetric Chamfer Distance Loss with additional penalty for uneven distribution
    
    Args:
        alpha: Weight for standard Chamfer distance
        beta: Weight for symmetry penalty
        reduction: Reduction method
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, reduction: str = 'mean'):
        super(SymmetricChamferLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.chamfer_loss = ChamferLoss(reduction=reduction)
        
    def compute_symmetry_penalty(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for asymmetric point distribution
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            penalty: Symmetry penalty
        """
        dist1, dist2 = chamfer_distance_squared(pred, target)
        
        # Mean distances in both directions
        mean_dist1 = torch.mean(dist1, dim=1)  # (B,)
        mean_dist2 = torch.mean(dist2, dim=1)  # (B,)
        
        # Penalty for difference in mean distances
        penalty = torch.abs(mean_dist1 - mean_dist2)
        
        return torch.mean(penalty)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Symmetric Chamfer distance loss
        """
        chamfer_loss = self.chamfer_loss(pred, target)
        symmetry_penalty = self.compute_symmetry_penalty(pred, target)
        
        return self.alpha * chamfer_loss + self.beta * symmetry_penalty


class RobustChamferLoss(nn.Module):
    """
    Robust Chamfer Distance Loss with outlier handling
    
    Args:
        threshold: Distance threshold for outlier detection
        reduction: Reduction method
        use_sqrt: Whether to use square root
    """
    
    def __init__(self, threshold: float = 0.1, reduction: str = 'mean', use_sqrt: bool = False):
        super(RobustChamferLoss, self).__init__()
        
        self.threshold = threshold
        self.reduction = reduction
        self.use_sqrt = use_sqrt
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            loss: Robust Chamfer distance loss
        """
        dist1, dist2 = chamfer_distance_squared(pred, target)
        
        if self.use_sqrt:
            dist1 = torch.sqrt(dist1 + 1e-8)
            dist2 = torch.sqrt(dist2 + 1e-8)
        
        # Apply robust loss function (Huber-like)
        def robust_loss(distances, threshold):
            mask = distances <= threshold
            loss = torch.where(mask, 
                             distances, 
                             threshold + torch.log(distances / threshold + 1e-8))
            return loss
        
        robust_dist1 = robust_loss(dist1, self.threshold)
        robust_dist2 = robust_loss(dist2, self.threshold)
        
        loss = robust_dist1 + robust_dist2
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss