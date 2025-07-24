"""
Feature matching loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss in high-dimensional feature space
    
    Args:
        feature_layers: List of layer names to extract features from
        layer_weights: Weights for each layer
        distance_type: Type of distance ('l2', 'l1', 'cosine')
        reduction: Reduction method
    """
    
    def __init__(self, 
                 feature_layers: List[str] = ['layer1', 'layer2', 'layer3'],
                 layer_weights: Optional[List[float]] = None,
                 distance_type: str = 'l2',
                 reduction: str = 'mean'):
        super(FeatureMatchingLoss, self).__init__()
        
        self.feature_layers = feature_layers
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * len(feature_layers)
        self.distance_type = distance_type
        self.reduction = reduction
        
        assert len(self.feature_layers) == len(self.layer_weights), \
            "Number of layers and weights must match"
    
    def extract_features(self, model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from intermediate layers
        
        Args:
            model: Neural network model
            x: Input tensor
            
        Returns:
            features: Dictionary of features from different layers
        """
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in self.feature_layers:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to extract features
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features
    
    def compute_feature_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between features
        
        Args:
            feat1: Features from first input
            feat2: Features from second input
            
        Returns:
            distance: Feature distance
        """
        if self.distance_type == 'l2':
            distance = F.mse_loss(feat1, feat2, reduction='none')
        elif self.distance_type == 'l1':
            distance = F.l1_loss(feat1, feat2, reduction='none')
        elif self.distance_type == 'cosine':
            # Cosine similarity distance
            feat1_norm = F.normalize(feat1, p=2, dim=-1)
            feat2_norm = F.normalize(feat2, p=2, dim=-1)
            cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=-1)
            distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return distance
    
    def forward(self, 
                model: nn.Module,
                pred_input: torch.Tensor, 
                target_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            model: Model to extract features from
            pred_input: Input that generates predicted output
            target_input: Input that generates target output
            
        Returns:
            loss: Feature matching loss
        """
        # Extract features for both inputs
        pred_features = self.extract_features(model, pred_input)
        target_features = self.extract_features(model, target_input)
        
        total_loss = 0.0
        
        for layer_name, weight in zip(self.feature_layers, self.layer_weights):
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                
                # Compute distance
                distance = self.compute_feature_distance(pred_feat, target_feat)
                
                # Apply reduction
                if self.reduction == 'mean':
                    layer_loss = torch.mean(distance)
                elif self.reduction == 'sum':
                    layer_loss = torch.sum(distance)
                else:
                    layer_loss = distance
                
                total_loss += weight * layer_loss
        
        return total_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features
    
    Args:
        feature_extractor: Pre-trained feature extractor
        layer_weights: Weights for different layers
        normalize_features: Whether to normalize features
    """
    
    def __init__(self, 
                 feature_extractor: nn.Module,
                 layer_weights: Dict[str, float],
                 normalize_features: bool = True):
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights
        self.normalize_features = normalize_features
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted output
            target: Target output
            
        Returns:
            loss: Perceptual loss
        """
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        total_loss = 0.0
        
        for layer_name, weight in self.layer_weights.items():
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                
                if self.normalize_features:
                    pred_feat = F.normalize(pred_feat, p=2, dim=1)
                    target_feat = F.normalize(target_feat, p=2, dim=1)
                
                loss = F.mse_loss(pred_feat, target_feat)
                total_loss += weight * loss
        
        return total_loss


class LocalFeatureMatchingLoss(nn.Module):
    """
    Local feature matching loss for point clouds
    
    Args:
        k_neighbors: Number of neighbors for local feature computation
        feature_dim: Feature dimension
        distance_type: Distance type for matching
    """
    
    def __init__(self, 
                 k_neighbors: int = 16,
                 feature_dim: int = 256,
                 distance_type: str = 'l2'):
        super(LocalFeatureMatchingLoss, self).__init__()
        
        self.k_neighbors = k_neighbors
        self.feature_dim = feature_dim
        self.distance_type = distance_type
        
        # Local feature extractor
        self.local_feature_net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1)
        )
    
    def get_knn_features(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Get k-nearest neighbor features
        
        Args:
            points: Point coordinates (B, N, 3)
            features: Point features (B, N, C)
            
        Returns:
            knn_features: KNN aggregated features (B, N, C)
        """
        B, N, C = features.shape
        
        # Compute pairwise distances
        points_T = points.transpose(1, 2)  # (B, 3, N)
        dist = torch.cdist(points, points)  # (B, N, N)
        
        # Get k nearest neighbors
        _, knn_idx = torch.topk(dist, self.k_neighbors, dim=2, largest=False)  # (B, N, k)
        
        # Gather neighbor features
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(B, N, self.k_neighbors, C)
        neighbor_features = torch.gather(features.unsqueeze(2).expand(B, N, N, C), 
                                       2, knn_idx_expanded)  # (B, N, k, C)
        
        # Aggregate neighbor features (mean pooling)
        knn_features = torch.mean(neighbor_features, dim=2)  # (B, N, C)
        
        return knn_features
    
    def forward(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred_points: Predicted point cloud (B, N, 3)
            target_points: Target point cloud (B, N, 3)
            
        Returns:
            loss: Local feature matching loss
        """
        # Extract local features
        pred_features = self.local_feature_net(pred_points.transpose(1, 2)).transpose(1, 2)
        target_features = self.local_feature_net(target_points.transpose(1, 2)).transpose(1, 2)
        
        # Get k-nearest neighbor features
        pred_knn_features = self.get_knn_features(pred_points, pred_features)
        target_knn_features = self.get_knn_features(target_points, target_features)
        
        # Compute feature matching loss
        if self.distance_type == 'l2':
            loss = F.mse_loss(pred_knn_features, target_knn_features)
        elif self.distance_type == 'l1':
            loss = F.l1_loss(pred_knn_features, target_knn_features)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return loss


class HierarchicalFeatureMatchingLoss(nn.Module):
    """
    Hierarchical feature matching loss at multiple resolutions
    
    Args:
        scales: List of scales for hierarchical matching
        feature_dims: Feature dimensions at each scale
        scale_weights: Weights for each scale
    """
    
    def __init__(self, 
                 scales: List[float] = [1.0, 0.5, 0.25],
                 feature_dims: List[int] = [256, 128, 64],
                 scale_weights: Optional[List[float]] = None):
        super(HierarchicalFeatureMatchingLoss, self).__init__()
        
        self.scales = scales
        self.feature_dims = feature_dims
        self.scale_weights = scale_weights if scale_weights is not None else [1.0] * len(scales)
        
        # Feature extractors for each scale
        self.feature_extractors = nn.ModuleList([
            LocalFeatureMatchingLoss(feature_dim=dim) 
            for dim in feature_dims
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
    
    def forward(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred_points: Predicted point cloud (B, N, 3)
            target_points: Target point cloud (B, N, 3)
            
        Returns:
            loss: Hierarchical feature matching loss
        """
        total_loss = 0.0
        
        for scale, weight, feature_extractor in zip(self.scales, self.scale_weights, self.feature_extractors):
            # Subsample points
            pred_sub = self.subsample_points(pred_points, scale)
            target_sub = self.subsample_points(target_points, scale)
            
            # Compute feature matching loss at this scale
            scale_loss = feature_extractor(pred_sub, target_sub)
            total_loss += weight * scale_loss
        
        return total_loss


class AdversarialFeatureLoss(nn.Module):
    """
    Adversarial feature loss using discriminator features
    
    Args:
        discriminator: Discriminator network
        layer_names: Names of layers to extract features from
        layer_weights: Weights for each layer
    """
    
    def __init__(self, 
                 discriminator: nn.Module,
                 layer_names: List[str],
                 layer_weights: Optional[List[float]] = None):
        super(AdversarialFeatureLoss, self).__init__()
        
        self.discriminator = discriminator
        self.layer_names = layer_names
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * len(layer_names)
        
    def extract_discriminator_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from discriminator
        
        Args:
            x: Input tensor
            
        Returns:
            features: Dictionary of features
        """
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.discriminator.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.discriminator(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted output
            target: Target output
            
        Returns:
            loss: Adversarial feature loss
        """
        # Extract features from discriminator
        pred_features = self.extract_discriminator_features(pred)
        target_features = self.extract_discriminator_features(target)
        
        total_loss = 0.0
        
        for layer_name, weight in zip(self.layer_names, self.layer_weights):
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                
                loss = F.mse_loss(pred_feat, target_feat)
                total_loss += weight * loss
        
        return total_loss