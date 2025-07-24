"""
Core PointNet architecture implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for point cloud feature extraction
    
    Args:
        input_dim: Input point dimension (default: 3 for xyz)
        hidden_dims: List of hidden layer dimensions
        global_feature_dim: Dimension of global feature vector
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: List[int] = [64, 128, 256, 512],
                 global_feature_dim: int = 1024,
                 use_batch_norm: bool = True):
        super(PointNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.global_feature_dim = global_feature_dim
        self.use_batch_norm = use_batch_norm
        
        # Point-wise feature extraction layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.conv_layers.append(nn.Conv1d(in_dim, hidden_dim, 1))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        
        # Global feature extraction
        self.global_conv = nn.Conv1d(hidden_dims[-1], global_feature_dim, 1)
        if use_batch_norm:
            self.global_bn = nn.BatchNorm1d(global_feature_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PointNet encoder
        
        Args:
            x: Input point cloud (B, N, input_dim)
            
        Returns:
            local_features: Point-wise features (B, N, hidden_dims[-1])
            global_feature: Global feature vector (B, global_feature_dim)
        """
        batch_size, num_points, _ = x.shape
        
        # Transpose to (B, input_dim, N) for conv1d
        x = x.transpose(2, 1)
        
        # Point-wise feature extraction
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
        
        # Store local features
        local_features = x.transpose(2, 1)  # (B, N, hidden_dims[-1])
        
        # Global feature extraction
        x = self.global_conv(x)
        if self.use_batch_norm:
            x = self.global_bn(x)
        x = F.relu(x)
        
        # Max pooling for global feature
        global_feature, _ = torch.max(x, dim=2)  # (B, global_feature_dim)
        
        return local_features, global_feature


class PointNetDecoder(nn.Module):
    """
    PointNet decoder for point cloud reconstruction
    
    Args:
        latent_dim: Dimension of latent vector
        hidden_dims: List of hidden layer dimensions (in reverse order)
        output_dim: Output point dimension (default: 3 for xyz)
        num_points: Number of output points
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(self,
                 latent_dim: int = 512,
                 hidden_dims: List[int] = [512, 256, 128, 64],
                 output_dim: int = 3,
                 num_points: int = 2048,
                 use_batch_norm: bool = True):
        super(PointNetDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_points = num_points
        self.use_batch_norm = use_batch_norm
        
        # Fully connected layers for decoding
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_points * output_dim)
        
        # Coordinate refinement layers
        self.refine_conv = nn.Conv1d(output_dim, output_dim, 1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PointNet decoder
        
        Args:
            z: Latent vector (B, latent_dim)
            
        Returns:
            reconstructed_points: Reconstructed point cloud (B, num_points, output_dim)
        """
        batch_size = z.shape[0]
        
        # Decode through fully connected layers
        x = z
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
        
        # Generate raw coordinates
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_points, self.output_dim)
        
        # Coordinate refinement
        x_refined = x.transpose(2, 1)  # (B, output_dim, num_points)
        x_refined = self.refine_conv(x_refined)
        x_refined = x_refined.transpose(2, 1)  # (B, num_points, output_dim)
        
        # Residual connection
        reconstructed_points = x + x_refined
        
        return reconstructed_points


class TNet(nn.Module):
    """
    Transformation network for spatial transformer
    
    Args:
        input_dim: Input dimension
        output_dim: Output transformation matrix dimension
    """
    
    def __init__(self, input_dim: int = 3, output_dim: int = 3):
        super(TNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim * output_dim)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of T-Net
        
        Args:
            x: Input tensor (B, N, input_dim)
            
        Returns:
            transformation_matrix: Transformation matrix (B, output_dim, output_dim)
        """
        batch_size = x.shape[0]
        
        x = x.transpose(2, 1)  # (B, input_dim, N)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(-1, 1024)  # (B, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, output_dim^2)
        
        # Initialize as identity matrix
        identity = torch.eye(self.output_dim, device=x.device, dtype=x.dtype)
        identity = identity.view(1, self.output_dim * self.output_dim).repeat(batch_size, 1)
        
        x = x + identity
        x = x.view(-1, self.output_dim, self.output_dim)
        
        return x


class PointNetWithTransform(nn.Module):
    """
    PointNet with spatial transformer networks
    
    Args:
        input_dim: Input point dimension
        hidden_dims: Hidden layer dimensions
        global_feature_dim: Global feature dimension
        use_input_transform: Whether to use input transformation
        use_feature_transform: Whether to use feature transformation
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dims: List[int] = [64, 128, 256, 512],
                 global_feature_dim: int = 1024,
                 use_input_transform: bool = True,
                 use_feature_transform: bool = True):
        super(PointNetWithTransform, self).__init__()
        
        self.input_dim = input_dim
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform
        
        # Transformation networks
        if use_input_transform:
            self.input_transform = TNet(input_dim, input_dim)
        
        if use_feature_transform:
            self.feature_transform = TNet(hidden_dims[0], hidden_dims[0])
        
        # PointNet encoder
        self.encoder = PointNetEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            global_feature_dim=global_feature_dim
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with transformations
        
        Args:
            x: Input point cloud (B, N, input_dim)
            
        Returns:
            local_features: Local features
            global_feature: Global feature
            transforms: Dictionary of transformation matrices
        """
        transforms = {}
        
        # Input transformation
        if self.use_input_transform:
            input_trans = self.input_transform(x)
            x = torch.bmm(x, input_trans)
            transforms['input'] = input_trans
        
        # Extract features
        local_features, global_feature = self.encoder(x)
        
        # Feature transformation
        if self.use_feature_transform:
            feature_trans = self.feature_transform(local_features)
            # Apply transformation to local features
            local_features_trans = torch.bmm(local_features, feature_trans)
            transforms['feature'] = feature_trans
            local_features = local_features_trans
        
        return local_features, global_feature, transforms