"""
PointNet++ architecture for hierarchical feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between points
    
    Args:
        src: Source points (B, N, C)
        dst: Destination points (B, M, C)
        
    Returns:
        dist: Squared distances (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points based on indices
    
    Args:
        points: Input points (B, N, C)
        idx: Indices (B, S) or (B, S, K)
        
    Returns:
        indexed_points: Selected points
    """
    device = points.device
    B = points.shape[0]
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling
    
    Args:
        xyz: Point coordinates (B, N, 3)
        npoint: Number of points to sample
        
    Returns:
        centroids: Sampled point indices (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Query ball point grouping
    
    Args:
        radius: Radius of the ball
        nsample: Maximum number of points in each ball
        xyz: All points (B, N, 3)
        new_xyz: Centroids (B, S, 3)
        
    Returns:
        group_idx: Grouped point indices (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer
    
    Args:
        npoint: Number of points after sampling
        radius: Ball query radius
        nsample: Number of samples in each ball
        in_channel: Input channel dimension
        mlp: MLP layer dimensions
        group_all: Whether to group all points
    """
    
    def __init__(self, 
                 npoint: int, 
                 radius: float, 
                 nsample: int, 
                 in_channel: int, 
                 mlp: List[int], 
                 group_all: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            xyz: Point coordinates (B, N, 3)
            points: Point features (B, N, C)
            
        Returns:
            new_xyz: Sampled point coordinates (B, npoint, 3)
            new_points: Aggregated features (B, npoint, mlp[-1])
        """
        B, N, C = xyz.shape
        
        if points is not None:
            points = points.permute(0, 2, 1)
        
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.permute(0, 2, 1).unsqueeze(-1)
        else:
            # Farthest point sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            # Ball query grouping
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, 3)
            
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), idx)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm
            
            grouped_points = grouped_points.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
            new_points = grouped_points
        
        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        new_points = torch.max(new_points, -1)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation Layer
    
    Args:
        in_channel: Input channel dimension
        mlp: MLP layer dimensions
    """
    
    def __init__(self, in_channel: int, mlp: List[int]):
        super(PointNetFeaturePropagation, self).__init__()
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, 
                xyz1: torch.Tensor, 
                xyz2: torch.Tensor, 
                points1: torch.Tensor, 
                points2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            xyz1: Sparse point coordinates (B, N1, 3)
            xyz2: Dense point coordinates (B, N2, 3)
            points1: Sparse point features (B, N1, C1)
            points2: Dense point features (B, N2, C2)
            
        Returns:
            new_points: Interpolated features (B, N2, mlp[-1])
        """
        B, N1, C1 = points1.shape
        _, N2, C2 = points2.shape
        
        if N1 == 1:
            interpolated_points = points1.repeat(1, N2, 1)
        else:
            dists = square_distance(xyz2, xyz1)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # (B, N2, 3)
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(index_points(points1, idx) * weight.view(B, N2, 3, 1), dim=2)
        
        if points2 is not None:
            new_points = torch.cat([points2, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.permute(0, 2, 1)


class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ Encoder with hierarchical feature learning
    
    Args:
        input_dim: Input point dimension
        num_points: Number of input points
        use_normals: Whether input includes normals
    """
    
    def __init__(self, 
                 input_dim: int = 3, 
                 num_points: int = 2048, 
                 use_normals: bool = False):
        super(PointNetPlusPlusEncoder, self).__init__()
        
        self.num_points = num_points
        in_channel = input_dim - 3 if use_normals else 0
        
        # Set abstraction layers (hierarchical sampling and grouping)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=128, 
            in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True
        )
    
    def forward(self, xyz: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            xyz: Input point cloud (B, N, 3) or (B, N, 6) with normals
            
        Returns:
            xyz_list: List of coordinates at each level
            points_list: List of features at each level
            global_feature: Global feature vector
        """
        B, N, C = xyz.shape
        
        if C > 3:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            norm = None
        
        xyz_list = [xyz]
        points_list = [norm]
        
        # Hierarchical feature extraction
        l1_xyz, l1_points = self.sa1(xyz, norm)
        xyz_list.append(l1_xyz)
        points_list.append(l1_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        xyz_list.append(l2_xyz)
        points_list.append(l2_points)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        xyz_list.append(l3_xyz)
        points_list.append(l3_points)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        xyz_list.append(l4_xyz)
        points_list.append(l4_points)
        
        # Global feature is the output of the last layer
        global_feature = l4_points.squeeze(1)  # (B, 1024)
        
        return xyz_list, points_list, global_feature


class PointNetPlusPlusDecoder(nn.Module):
    """
    PointNet++ Decoder with feature propagation
    
    Args:
        latent_dim: Latent vector dimension
        num_points: Number of output points
        output_dim: Output point dimension
    """
    
    def __init__(self, 
                 latent_dim: int = 1024,
                 num_points: int = 2048, 
                 output_dim: int = 3):
        super(PointNetPlusPlusDecoder, self).__init__()
        
        self.num_points = num_points
        self.output_dim = output_dim
        
        # Feature propagation layers
        self.fp4 = PointNetFeaturePropagation(1024 + 512, [512, 512])
        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        # Output layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_dim, 1)
        
        # Latent to coordinate transformation
        self.latent_fc = nn.Linear(latent_dim, num_points * 3)
        
    def forward(self, 
                latent: torch.Tensor,
                xyz_list: List[torch.Tensor], 
                points_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            latent: Latent vector (B, latent_dim)
            xyz_list: List of coordinates from encoder
            points_list: List of features from encoder
            
        Returns:
            output_points: Reconstructed point cloud (B, num_points, output_dim)
        """
        # Generate base coordinates from latent vector
        base_coords = self.latent_fc(latent)
        base_coords = base_coords.view(-1, self.num_points, 3)
        
        # Feature propagation (upsampling)
        l3_points = self.fp4(xyz_list[3], xyz_list[2], points_list[4], points_list[3])
        l2_points = self.fp3(xyz_list[2], xyz_list[1], l3_points, points_list[2])
        l1_points = self.fp2(xyz_list[1], xyz_list[0], l2_points, points_list[1])
        l0_points = self.fp1(xyz_list[0], base_coords, l1_points, points_list[0])
        
        # Output generation
        feat = self.drop1(F.relu(self.bn1(self.conv1(l0_points.permute(0, 2, 1)))))
        output = self.conv2(feat)  # (B, output_dim, num_points)
        output = output.permute(0, 2, 1)  # (B, num_points, output_dim)
        
        # Add residual connection with base coordinates
        if self.output_dim == 3:
            output = output + base_coords
        
        return output


class PointNetPlusPlus(nn.Module):
    """
    Complete PointNet++ architecture
    
    Args:
        input_dim: Input point dimension
        num_points: Number of points
        latent_dim: Latent dimension for VAE
        use_normals: Whether to use normal vectors
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 num_points: int = 2048,
                 latent_dim: int = 1024,
                 use_normals: bool = False):
        super(PointNetPlusPlus, self).__init__()
        
        self.encoder = PointNetPlusPlusEncoder(input_dim, num_points, use_normals)
        self.decoder = PointNetPlusPlusDecoder(latent_dim, num_points, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input point cloud (B, N, C)
            
        Returns:
            global_feature: Global feature vector
            xyz_list: List of coordinates at each level
            points_list: List of features at each level
        """
        xyz_list, points_list, global_feature = self.encoder(x)
        return global_feature, xyz_list, points_list
    
    def decode(self, 
               latent: torch.Tensor,
               xyz_list: List[torch.Tensor], 
               points_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode latent vector to point cloud
        
        Args:
            latent: Latent vector
            xyz_list: Coordinate lists from encoder
            points_list: Feature lists from encoder
            
        Returns:
            reconstructed: Reconstructed point cloud
        """
        return self.decoder(latent, xyz_list, points_list)