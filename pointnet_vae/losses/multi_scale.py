"""
Multi-scale loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Dict, Any
import math


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for supervision at different resolution levels
    
    Args:
        base_loss: Base loss function to apply at each scale
        scales: List of scales (subsample rates)
        weights: Weights for each scale
        sampling_method: Method for sampling ('random', 'fps', 'uniform')
    """
    
    def __init__(self, 
                 base_loss: nn.Module,
                 scales: List[float] = [1.0, 0.5, 0.25],
                 weights: Optional[List[float]] = None,
                 sampling_method: str = 'fps'):
        super(MultiScaleLoss, self).__init__()
        
        self.base_loss = base_loss
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        self.sampling_method = sampling_method
        
        assert len(self.scales) == len(self.weights), "Scales and weights must have same length"
    
    def farthest_point_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest point sampling for better coverage
        
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
    
    def uniform_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Uniform sampling
        
        Args:
            points: Input points (B, N, D)
            num_samples: Number of samples
            
        Returns:
            indices: Sampled indices (B, num_samples)
        """
        B, N, D = points.shape
        indices = torch.linspace(0, N-1, num_samples, dtype=torch.long, device=points.device)
        indices = indices.unsqueeze(0).expand(B, -1)
        return indices
    
    def random_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Random sampling
        
        Args:
            points: Input points (B, N, D)
            num_samples: Number of samples
            
        Returns:
            indices: Sampled indices (B, num_samples)
        """
        B, N, D = points.shape
        indices = torch.randperm(N, device=points.device)[:num_samples]
        indices = indices.unsqueeze(0).expand(B, -1)
        return indices
    
    def subsample_points(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Subsample points according to scale and method
        
        Args:
            points: Input points (B, N, D)
            scale: Subsampling scale
            
        Returns:
            subsampled_points: Subsampled points
        """
        if scale >= 1.0:
            return points
        
        B, N, D = points.shape
        num_samples = max(1, int(N * scale))
        
        if self.sampling_method == 'fps':
            indices = self.farthest_point_sample(points, num_samples)
        elif self.sampling_method == 'uniform':
            indices = self.uniform_sample(points, num_samples)
        elif self.sampling_method == 'random':
            indices = self.random_sample(points, num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # Gather points using indices
        indices_expanded = indices.unsqueeze(-1).expand(B, num_samples, D)
        subsampled_points = torch.gather(points, 1, indices_expanded)
        
        return subsampled_points
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, D)
            target: Target point cloud (B, M, D)
            
        Returns:
            loss: Multi-scale loss
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            # Subsample both point clouds
            pred_sub = self.subsample_points(pred, scale)
            target_sub = self.subsample_points(target, scale)
            
            # Compute loss at this scale
            scale_loss = self.base_loss(pred_sub, target_sub)
            total_loss += weight * scale_loss
        
        return total_loss


class PyramidLoss(nn.Module):
    """
    Pyramid loss with hierarchical supervision
    
    Args:
        loss_functions: List of loss functions for each level
        level_weights: Weights for each pyramid level
        downsample_factors: Downsampling factors for each level
    """
    
    def __init__(self, 
                 loss_functions: List[nn.Module],
                 level_weights: Optional[List[float]] = None,
                 downsample_factors: List[int] = [1, 2, 4, 8]):
        super(PyramidLoss, self).__init__()
        
        self.loss_functions = nn.ModuleList(loss_functions)
        self.level_weights = level_weights if level_weights is not None else [1.0] * len(loss_functions)
        self.downsample_factors = downsample_factors
        
        assert len(self.loss_functions) == len(self.level_weights) == len(self.downsample_factors), \
            "All lists must have the same length"
    
    def create_pyramid(self, points: torch.Tensor) -> List[torch.Tensor]:
        """
        Create pyramid of point clouds at different resolutions
        
        Args:
            points: Input point cloud (B, N, D)
            
        Returns:
            pyramid: List of point clouds at different resolutions
        """
        pyramid = []
        
        for factor in self.downsample_factors:
            if factor == 1:
                pyramid.append(points)
            else:
                # Downsample by factor
                B, N, D = points.shape
                num_points = max(1, N // factor)
                
                # Use farthest point sampling for downsampling
                indices = self.farthest_point_sample(points, num_points)
                indices_expanded = indices.unsqueeze(-1).expand(B, num_points, D)
                downsampled = torch.gather(points, 1, indices_expanded)
                pyramid.append(downsampled)
        
        return pyramid
    
    def farthest_point_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Farthest point sampling (same as in MultiScaleLoss)"""
        B, N, D = points.shape
        device = points.device
        
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
            pred: Predicted point cloud (B, N, D)
            target: Target point cloud (B, M, D)
            
        Returns:
            loss: Pyramid loss
        """
        # Create pyramids
        pred_pyramid = self.create_pyramid(pred)
        target_pyramid = self.create_pyramid(target)
        
        total_loss = 0.0
        
        for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.level_weights)):
            level_loss = loss_fn(pred_pyramid[i], target_pyramid[i])
            total_loss += weight * level_loss
        
        return total_loss


class AdaptiveMultiScaleLoss(nn.Module):
    """
    Adaptive multi-scale loss that adjusts weights based on loss values
    
    Args:
        base_loss: Base loss function
        scales: List of scales
        adaptation_rate: Rate of weight adaptation
        min_weight: Minimum weight value
    """
    
    def __init__(self, 
                 base_loss: nn.Module,
                 scales: List[float] = [1.0, 0.5, 0.25],
                 adaptation_rate: float = 0.1,
                 min_weight: float = 0.1):
        super(AdaptiveMultiScaleLoss, self).__init__()
        
        self.base_loss = base_loss
        self.scales = scales
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        
        # Initialize weights
        self.register_buffer('weights', torch.ones(len(scales)))
        self.register_buffer('loss_history', torch.zeros(len(scales)))
        self.register_buffer('step_count', torch.tensor(0))
    
    def update_weights(self, scale_losses: torch.Tensor):
        """
        Update weights based on loss values
        
        Args:
            scale_losses: Loss values at different scales
        """
        # Update loss history
        self.loss_history = (1 - self.adaptation_rate) * self.loss_history + \
                           self.adaptation_rate * scale_losses.detach()
        
        # Adaptive weights (inverse of relative loss magnitude)
        relative_losses = self.loss_history / (torch.sum(self.loss_history) + 1e-8)
        adaptive_weights = 1.0 / (relative_losses + 1e-8)
        adaptive_weights = adaptive_weights / torch.sum(adaptive_weights)
        
        # Apply minimum weight constraint
        adaptive_weights = torch.clamp(adaptive_weights, min=self.min_weight)
        adaptive_weights = adaptive_weights / torch.sum(adaptive_weights)
        
        self.weights = adaptive_weights
        self.step_count += 1
    
    def subsample_points(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """Subsample points (same as MultiScaleLoss)"""
        if scale >= 1.0:
            return points
        
        B, N, D = points.shape
        num_samples = max(1, int(N * scale))
        
        # Random sampling for simplicity
        indices = torch.randperm(N, device=points.device)[:num_samples]
        return points[:, indices, :]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, D)
            target: Target point cloud (B, M, D)
            
        Returns:
            loss: Adaptive multi-scale loss
        """
        scale_losses = []
        
        # Compute loss at each scale
        for scale in self.scales:
            pred_sub = self.subsample_points(pred, scale)
            target_sub = self.subsample_points(target, scale)
            scale_loss = self.base_loss(pred_sub, target_sub)
            scale_losses.append(scale_loss)
        
        scale_losses = torch.stack(scale_losses)
        
        # Update weights if in training mode
        if self.training:
            self.update_weights(scale_losses)
        
        # Weighted sum
        total_loss = torch.sum(self.weights * scale_losses)
        
        return total_loss


class ProgressiveMultiScaleLoss(nn.Module):
    """
    Progressive multi-scale loss that starts with coarse scales and gradually adds finer scales
    
    Args:
        base_loss: Base loss function
        scales: List of scales (from coarse to fine)
        progression_epochs: Epochs at which to add each scale
        final_weights: Final weights for each scale
    """
    
    def __init__(self, 
                 base_loss: nn.Module,
                 scales: List[float] = [0.25, 0.5, 1.0],
                 progression_epochs: List[int] = [50, 100, 150],
                 final_weights: Optional[List[float]] = None):
        super(ProgressiveMultiScaleLoss, self).__init__()
        
        self.base_loss = base_loss
        self.scales = scales
        self.progression_epochs = progression_epochs
        self.final_weights = final_weights if final_weights is not None else [1.0] * len(scales)
        
        self.register_buffer('current_epoch', torch.tensor(0))
        
        assert len(scales) == len(progression_epochs) == len(self.final_weights), \
            "All lists must have the same length"
    
    def get_active_scales(self) -> List[int]:
        """
        Get indices of currently active scales
        
        Returns:
            active_indices: List of active scale indices
        """
        active_indices = []
        for i, epoch_threshold in enumerate(self.progression_epochs):
            if self.current_epoch >= epoch_threshold:
                active_indices.append(i)
        
        # Always include at least the coarsest scale
        if not active_indices:
            active_indices = [0]
        
        return active_indices
    
    def subsample_points(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """Subsample points (same as MultiScaleLoss)"""
        if scale >= 1.0:
            return points
        
        B, N, D = points.shape
        num_samples = max(1, int(N * scale))
        
        indices = torch.randperm(N, device=points.device)[:num_samples]
        return points[:, indices, :]
    
    def set_epoch(self, epoch: int):
        """
        Set current epoch for progressive training
        
        Args:
            epoch: Current epoch
        """
        self.current_epoch = torch.tensor(epoch)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted point cloud (B, N, D)
            target: Target point cloud (B, M, D)
            
        Returns:
            loss: Progressive multi-scale loss
        """
        active_indices = self.get_active_scales()
        total_loss = 0.0
        total_weight = 0.0
        
        for i in active_indices:
            scale = self.scales[i]
            weight = self.final_weights[i]
            
            pred_sub = self.subsample_points(pred, scale)
            target_sub = self.subsample_points(target, scale)
            
            scale_loss = self.base_loss(pred_sub, target_sub)
            total_loss += weight * scale_loss
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss