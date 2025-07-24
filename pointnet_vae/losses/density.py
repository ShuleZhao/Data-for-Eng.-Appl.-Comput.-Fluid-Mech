"""
Density regularization loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DensityRegularizationLoss(nn.Module):
    """
    Density regularization loss to ensure reasonable point cloud density distribution
    
    Args:
        kernel_size: Size of the density estimation kernel
        sigma: Standard deviation for Gaussian kernel
        lambda_uniform: Weight for uniform density penalty
        lambda_sparse: Weight for sparsity penalty
        reduction: Reduction method
    """
    
    def __init__(self, 
                 kernel_size: float = 0.1,
                 sigma: float = 0.05,
                 lambda_uniform: float = 1.0,
                 lambda_sparse: float = 0.1,
                 reduction: str = 'mean'):
        super(DensityRegularizationLoss, self).__init__()
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambda_uniform = lambda_uniform
        self.lambda_sparse = lambda_sparse
        self.reduction = reduction
    
    def compute_density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute local density for each point using Gaussian kernel
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            densities: Local densities (B, N)
        """
        B, N, D = points.shape
        
        # Compute pairwise distances
        points_expanded_1 = points.unsqueeze(2).expand(B, N, N, D)
        points_expanded_2 = points.unsqueeze(1).expand(B, N, N, D)
        distances = torch.sum((points_expanded_1 - points_expanded_2) ** 2, dim=3)
        
        # Gaussian kernel
        kernel_weights = torch.exp(-distances / (2 * self.sigma ** 2))
        
        # Sum weights to get density (exclude self-weight)
        mask = torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1)
        kernel_weights = kernel_weights * (1 - mask)
        densities = torch.sum(kernel_weights, dim=2)
        
        return densities
    
    def compute_uniformity_loss(self, densities: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that encourages uniform density distribution
        
        Args:
            densities: Local densities (B, N)
            
        Returns:
            uniformity_loss: Uniformity loss
        """
        # Variance of densities (higher variance = less uniform)
        mean_density = torch.mean(densities, dim=1, keepdim=True)
        variance = torch.mean((densities - mean_density) ** 2, dim=1)
        
        return torch.mean(variance)
    
    def compute_sparsity_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that penalizes overly sparse regions
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            sparsity_loss: Sparsity loss
        """
        B, N, D = points.shape
        
        # Compute minimum distances to other points
        points_expanded_1 = points.unsqueeze(2).expand(B, N, N, D)
        points_expanded_2 = points.unsqueeze(1).expand(B, N, N, D)
        distances = torch.sum((points_expanded_1 - points_expanded_2) ** 2, dim=3)
        
        # Set diagonal to infinity to exclude self-distances
        mask = torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1)
        distances = distances + mask * 1e10
        
        # Find minimum distances
        min_distances, _ = torch.min(distances, dim=2)
        
        # Penalize points that are too far from others
        sparsity_penalty = F.relu(min_distances - self.kernel_size)
        
        return torch.mean(sparsity_penalty)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            loss: Density regularization loss
        """
        # Compute local densities
        densities = self.compute_density(points)
        
        # Uniformity loss
        uniformity_loss = self.compute_uniformity_loss(densities)
        
        # Sparsity loss
        sparsity_loss = self.compute_sparsity_loss(points)
        
        # Combined loss
        total_loss = self.lambda_uniform * uniformity_loss + self.lambda_sparse * sparsity_loss
        
        return total_loss


class VoronoiDensityLoss(nn.Module):
    """
    Density loss based on Voronoi diagram analysis
    
    Args:
        num_neighbors: Number of neighbors to consider
        target_volume: Target volume for each Voronoi cell
        reduction: Reduction method
    """
    
    def __init__(self, 
                 num_neighbors: int = 8,
                 target_volume: Optional[float] = None,
                 reduction: str = 'mean'):
        super(VoronoiDensityLoss, self).__init__()
        
        self.num_neighbors = num_neighbors
        self.target_volume = target_volume
        self.reduction = reduction
    
    def compute_voronoi_volumes(self, points: torch.Tensor) -> torch.Tensor:
        """
        Approximate Voronoi cell volumes using k-nearest neighbors
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            volumes: Approximate Voronoi cell volumes (B, N)
        """
        B, N, D = points.shape
        
        # Compute pairwise distances
        points_expanded_1 = points.unsqueeze(2).expand(B, N, N, D)
        points_expanded_2 = points.unsqueeze(1).expand(B, N, N, D)
        distances = torch.sum((points_expanded_1 - points_expanded_2) ** 2, dim=3)
        
        # Find k nearest neighbors (excluding self)
        distances = distances + torch.eye(N, device=points.device).unsqueeze(0) * 1e10
        k_distances, _ = torch.topk(distances, self.num_neighbors, dim=2, largest=False)
        
        # Approximate volume as sphere with radius = mean distance to k neighbors
        mean_distances = torch.mean(k_distances, dim=2)
        volumes = (4/3) * math.pi * mean_distances ** 3
        
        return volumes
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            loss: Voronoi density loss
        """
        volumes = self.compute_voronoi_volumes(points)
        
        if self.target_volume is None:
            # Encourage uniform volumes
            mean_volume = torch.mean(volumes, dim=1, keepdim=True)
            loss = torch.mean((volumes - mean_volume) ** 2)
        else:
            # Encourage specific target volume
            loss = torch.mean((volumes - self.target_volume) ** 2)
        
        return loss


class RepulsionLoss(nn.Module):
    """
    Repulsion loss to prevent points from clustering too closely
    
    Args:
        min_distance: Minimum allowed distance between points
        repulsion_strength: Strength of repulsion force
        reduction: Reduction method
    """
    
    def __init__(self, 
                 min_distance: float = 0.01,
                 repulsion_strength: float = 1.0,
                 reduction: str = 'mean'):
        super(RepulsionLoss, self).__init__()
        
        self.min_distance = min_distance
        self.repulsion_strength = repulsion_strength
        self.reduction = reduction
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            loss: Repulsion loss
        """
        B, N, D = points.shape
        
        # Compute pairwise distances
        points_expanded_1 = points.unsqueeze(2).expand(B, N, N, D)
        points_expanded_2 = points.unsqueeze(1).expand(B, N, N, D)
        distances = torch.sqrt(torch.sum((points_expanded_1 - points_expanded_2) ** 2, dim=3) + 1e-8)
        
        # Exclude self-distances
        mask = torch.eye(N, device=points.device).unsqueeze(0).expand(B, -1, -1)
        distances = distances + mask * 1e10
        
        # Repulsion force (inverse square law with cutoff)
        repulsion = torch.clamp(self.min_distance - distances, min=0)
        repulsion_force = self.repulsion_strength * repulsion ** 2
        
        # Sum over all pairs, divide by 2 to avoid double counting
        total_repulsion = torch.sum(repulsion_force, dim=(1, 2)) / 2
        
        if self.reduction == 'mean':
            return torch.mean(total_repulsion)
        elif self.reduction == 'sum':
            return torch.sum(total_repulsion)
        else:
            return total_repulsion


class CoverageAndDensityLoss(nn.Module):
    """
    Combined loss for point cloud coverage and density
    
    Args:
        coverage_weight: Weight for coverage loss
        density_weight: Weight for density loss
        grid_resolution: Resolution of coverage grid
        density_kernel_size: Kernel size for density estimation
    """
    
    def __init__(self, 
                 coverage_weight: float = 1.0,
                 density_weight: float = 1.0,
                 grid_resolution: int = 32,
                 density_kernel_size: float = 0.1):
        super(CoverageAndDensityLoss, self).__init__()
        
        self.coverage_weight = coverage_weight
        self.density_weight = density_weight
        self.grid_resolution = grid_resolution
        self.density_loss = DensityRegularizationLoss(kernel_size=density_kernel_size)
    
    def compute_coverage_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute coverage loss using voxel grid
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            coverage_loss: Coverage loss
        """
        B, N, D = points.shape
        
        # Normalize points to [0, 1]
        points_min = torch.min(points, dim=1, keepdim=True)[0]
        points_max = torch.max(points, dim=1, keepdim=True)[0]
        points_norm = (points - points_min) / (points_max - points_min + 1e-8)
        
        # Discretize to grid
        grid_coords = (points_norm * (self.grid_resolution - 1)).long()
        grid_coords = torch.clamp(grid_coords, 0, self.grid_resolution - 1)
        
        # Create occupancy grid
        total_coverage = 0.0
        
        for b in range(B):
            grid = torch.zeros(self.grid_resolution, self.grid_resolution, self.grid_resolution, 
                             device=points.device)
            
            coords = grid_coords[b]  # (N, 3)
            grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            
            # Coverage is the fraction of occupied voxels
            coverage = torch.sum(grid) / (self.grid_resolution ** 3)
            total_coverage += coverage
        
        # Encourage high coverage
        coverage_loss = 1.0 - (total_coverage / B)
        
        return coverage_loss
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            loss: Combined coverage and density loss
        """
        coverage_loss = self.compute_coverage_loss(points)
        density_loss = self.density_loss(points)
        
        return self.coverage_weight * coverage_loss + self.density_weight * density_loss


class AdaptiveDensityLoss(nn.Module):
    """
    Adaptive density loss that adjusts based on local geometry
    
    Args:
        base_kernel_size: Base kernel size for density estimation
        adaptive_factor: Factor for adaptive adjustment
        curvature_weight: Weight for curvature-based adaptation
    """
    
    def __init__(self, 
                 base_kernel_size: float = 0.1,
                 adaptive_factor: float = 0.5,
                 curvature_weight: float = 1.0):
        super(AdaptiveDensityLoss, self).__init__()
        
        self.base_kernel_size = base_kernel_size
        self.adaptive_factor = adaptive_factor
        self.curvature_weight = curvature_weight
    
    def estimate_curvature(self, points: torch.Tensor, k_neighbors: int = 16) -> torch.Tensor:
        """
        Estimate local curvature using k-nearest neighbors
        
        Args:
            points: Point cloud (B, N, 3)
            k_neighbors: Number of neighbors for curvature estimation
            
        Returns:
            curvatures: Local curvature estimates (B, N)
        """
        B, N, D = points.shape
        
        # Compute pairwise distances
        distances = torch.cdist(points, points)
        distances = distances + torch.eye(N, device=points.device).unsqueeze(0) * 1e10
        
        # Find k nearest neighbors
        _, knn_indices = torch.topk(distances, k_neighbors, dim=2, largest=False)
        
        curvatures = []
        
        for b in range(B):
            batch_curvatures = []
            
            for i in range(N):
                # Get neighbor points
                neighbor_indices = knn_indices[b, i]
                neighbors = points[b, neighbor_indices]  # (k, 3)
                
                # Compute covariance matrix
                centered = neighbors - torch.mean(neighbors, dim=0, keepdim=True)
                cov_matrix = torch.mm(centered.T, centered) / (k_neighbors - 1)
                
                # Eigenvalues indicate local geometry
                eigenvals, _ = torch.linalg.eigh(cov_matrix)
                
                # Curvature approximation using eigenvalue ratios
                curvature = eigenvals[0] / (eigenvals[2] + 1e-8)
                batch_curvatures.append(curvature)
            
            curvatures.append(torch.stack(batch_curvatures))
        
        return torch.stack(curvatures)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: Point cloud (B, N, 3)
            
        Returns:
            loss: Adaptive density loss
        """
        # Estimate local curvature
        curvatures = self.estimate_curvature(points)
        
        # Adaptive kernel sizes based on curvature
        adaptive_kernels = self.base_kernel_size * (1 + self.adaptive_factor * curvatures)
        
        # Compute density with adaptive kernels
        B, N, D = points.shape
        total_loss = 0.0
        
        for b in range(B):
            points_b = points[b]
            kernels_b = adaptive_kernels[b]
            
            # Pairwise distances
            distances = torch.cdist(points_b.unsqueeze(0), points_b.unsqueeze(0)).squeeze(0)
            
            # Adaptive Gaussian kernels
            kernel_matrix = torch.exp(-distances.unsqueeze(0) / (2 * kernels_b.unsqueeze(1) ** 2))
            
            # Remove self-weights
            mask = torch.eye(N, device=points.device)
            kernel_matrix = kernel_matrix * (1 - mask)
            
            # Local densities
            densities = torch.sum(kernel_matrix, dim=1)
            
            # Uniformity loss
            mean_density = torch.mean(densities)
            variance = torch.mean((densities - mean_density) ** 2)
            total_loss += variance
        
        return total_loss / B