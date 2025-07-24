"""
Advanced sampling methods for point clouds
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class FarthestPointSampling:
    """
    Farthest Point Sampling (FPS) for better point distribution
    
    Args:
        num_samples: Number of points to sample
    """
    
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
    
    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FPS sampling
        
        Args:
            points: Input points (B, N, 3) or (N, 3)
            
        Returns:
            sampled_points: Sampled points
            indices: Indices of sampled points
        """
        if len(points.shape) == 2:
            return self._fps_single(points)
        else:
            return self._fps_batch(points)
    
    def _fps_single(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FPS for single point cloud"""
        N, D = points.shape
        device = points.device
        
        if N <= self.num_samples:
            indices = torch.arange(N, device=device)
            return points, indices
        
        # Initialize
        centroids = torch.zeros(self.num_samples, dtype=torch.long, device=device)
        distance = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
        
        for i in range(self.num_samples):
            centroids[i] = farthest
            centroid = points[farthest].view(1, D)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance)
        
        sampled_points = points[centroids]
        return sampled_points, centroids
    
    def _fps_batch(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FPS for batch of point clouds"""
        B, N, D = points.shape
        device = points.device
        
        if N <= self.num_samples:
            indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return points, indices
        
        # Initialize
        centroids = torch.zeros(B, self.num_samples, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(self.num_samples):
            centroids[:, i] = farthest
            centroid = points[torch.arange(B, device=device), farthest, :].view(B, 1, D)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, dim=1)
        
        # Gather sampled points
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.num_samples)
        sampled_points = points[batch_indices, centroids]
        
        return sampled_points, centroids


class DensityBasedSampling:
    """
    Density-based sampling for adaptive point distribution
    
    Args:
        num_samples: Number of points to sample
        density_estimation_k: Number of neighbors for density estimation
        adaptive_factor: Factor for adaptive sampling based on density
    """
    
    def __init__(self, 
                 num_samples: int,
                 density_estimation_k: int = 16,
                 adaptive_factor: float = 0.5):
        self.num_samples = num_samples
        self.density_estimation_k = density_estimation_k
        self.adaptive_factor = adaptive_factor
    
    def estimate_density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Estimate local density for each point
        
        Args:
            points: Input points (N, 3)
            
        Returns:
            densities: Local density estimates (N,)
        """
        N, D = points.shape
        device = points.device
        
        # Compute pairwise distances
        distances = torch.cdist(points, points)
        
        # Find k nearest neighbors (excluding self)
        distances = distances + torch.eye(N, device=device) * 1e10
        k_distances, _ = torch.topk(distances, self.density_estimation_k, dim=1, largest=False)
        
        # Density approximation (inverse of mean distance to k neighbors)
        mean_distances = torch.mean(k_distances, dim=1)
        densities = 1.0 / (mean_distances + 1e-8)
        
        return densities
    
    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply density-based sampling
        
        Args:
            points: Input points (N, 3) or (B, N, 3)
            
        Returns:
            sampled_points: Sampled points
            indices: Indices of sampled points
        """
        if len(points.shape) == 3:
            return self._sample_batch(points)
        else:
            return self._sample_single(points)
    
    def _sample_single(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Density-based sampling for single point cloud"""
        N, D = points.shape
        device = points.device
        
        if N <= self.num_samples:
            indices = torch.arange(N, device=device)
            return points, indices
        
        # Estimate densities
        densities = self.estimate_density(points)
        
        # Create sampling probabilities (balance between uniform and density-based)
        uniform_prob = torch.ones(N, device=device) / N
        density_prob = densities / torch.sum(densities)
        
        sampling_prob = (1 - self.adaptive_factor) * uniform_prob + self.adaptive_factor * density_prob
        
        # Sample points based on probabilities
        indices = torch.multinomial(sampling_prob, self.num_samples, replacement=False)
        sampled_points = points[indices]
        
        return sampled_points, indices
    
    def _sample_batch(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Density-based sampling for batch of point clouds"""
        B, N, D = points.shape
        
        sampled_points_list = []
        indices_list = []
        
        for b in range(B):
            sampled_points_b, indices_b = self._sample_single(points[b])
            sampled_points_list.append(sampled_points_b)
            indices_list.append(indices_b)
        
        sampled_points = torch.stack(sampled_points_list)
        indices = torch.stack(indices_list)
        
        return sampled_points, indices


class VoxelGridSampling:
    """
    Voxel grid sampling for uniform spatial distribution
    
    Args:
        voxel_size: Size of each voxel
        max_points_per_voxel: Maximum points to keep per voxel
    """
    
    def __init__(self, voxel_size: float = 0.01, max_points_per_voxel: int = 1):
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
    
    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply voxel grid sampling
        
        Args:
            points: Input points (N, 3) or (B, N, 3)
            
        Returns:
            sampled_points: Sampled points
            indices: Indices of sampled points
        """
        if len(points.shape) == 3:
            return self._sample_batch(points)
        else:
            return self._sample_single(points)
    
    def _sample_single(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxel grid sampling for single point cloud"""
        if not OPEN3D_AVAILABLE:
            print("Open3D not available, falling back to random sampling")
            N, D = points.shape
            target_size = int(N * 0.8)  # Approximate downsampling
            indices = torch.randperm(N, device=points.device)[:target_size]
            return points[indices], indices
            
        N, D = points.shape
        device = points.device
        
        # Convert to numpy for easier processing
        points_np = points.cpu().numpy()
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        
        # Apply voxel grid downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Convert back to tensor
        sampled_points_np = np.asarray(downsampled_pcd.points)
        sampled_points = torch.from_numpy(sampled_points_np).float().to(device)
        
        # Find indices (approximate)
        indices = self._find_closest_indices(points, sampled_points)
        
        return sampled_points, indices
    
    def _sample_batch(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxel grid sampling for batch of point clouds"""
        B, N, D = points.shape
        
        sampled_points_list = []
        indices_list = []
        
        for b in range(B):
            sampled_points_b, indices_b = self._sample_single(points[b])
            sampled_points_list.append(sampled_points_b)
            indices_list.append(indices_b)
        
        # Pad to same length
        max_len = max(sp.shape[0] for sp in sampled_points_list)
        
        padded_points = []
        padded_indices = []
        
        for sp, idx in zip(sampled_points_list, indices_list):
            if sp.shape[0] < max_len:
                # Pad with last point
                padding = max_len - sp.shape[0]
                last_point = sp[-1:].expand(padding, -1)
                last_idx = idx[-1:].expand(padding)
                
                sp = torch.cat([sp, last_point], dim=0)
                idx = torch.cat([idx, last_idx], dim=0)
            
            padded_points.append(sp)
            padded_indices.append(idx)
        
        sampled_points = torch.stack(padded_points)
        indices = torch.stack(padded_indices)
        
        return sampled_points, indices
    
    def _find_closest_indices(self, original: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        """Find indices of closest points in original cloud"""
        distances = torch.cdist(sampled, original)
        indices = torch.argmin(distances, dim=1)
        return indices


class PoissonDiskSampling:
    """
    Poisson disk sampling for uniform distribution with minimum distance constraint
    
    Args:
        num_samples: Number of points to sample
        min_distance: Minimum distance between samples
        max_attempts: Maximum attempts per sample
    """
    
    def __init__(self, 
                 num_samples: int,
                 min_distance: float = 0.05,
                 max_attempts: int = 30):
        self.num_samples = num_samples
        self.min_distance = min_distance
        self.max_attempts = max_attempts
    
    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Poisson disk sampling
        
        Args:
            points: Input points (N, 3) or (B, N, 3)
            
        Returns:
            sampled_points: Sampled points
            indices: Indices of sampled points
        """
        if len(points.shape) == 3:
            return self._sample_batch(points)
        else:
            return self._sample_single(points)
    
    def _sample_single(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Poisson disk sampling for single point cloud"""
        N, D = points.shape
        device = points.device
        
        if N <= self.num_samples:
            indices = torch.arange(N, device=device)
            return points, indices
        
        # Initialize with random point
        sampled_indices = [torch.randint(0, N, (1,), device=device).item()]
        sampled_points = [points[sampled_indices[0]]]
        
        candidates = list(range(N))
        candidates.remove(sampled_indices[0])
        
        while len(sampled_indices) < self.num_samples and candidates:
            added = False
            attempts = 0
            
            while attempts < self.max_attempts and candidates and not added:
                # Random candidate
                candidate_idx = np.random.choice(len(candidates))
                candidate_global_idx = candidates[candidate_idx]
                candidate_point = points[candidate_global_idx]
                
                # Check minimum distance constraint
                valid = True
                for sp in sampled_points:
                    if torch.norm(candidate_point - sp) < self.min_distance:
                        valid = False
                        break
                
                if valid:
                    sampled_indices.append(candidate_global_idx)
                    sampled_points.append(candidate_point)
                    candidates.remove(candidate_global_idx)
                    added = True
                
                attempts += 1
            
            if not added:
                # Remove candidates that are too close to existing samples
                remaining_candidates = []
                for c_idx in candidates:
                    c_point = points[c_idx]
                    valid = True
                    for sp in sampled_points:
                        if torch.norm(c_point - sp) < self.min_distance:
                            valid = False
                            break
                    if valid:
                        remaining_candidates.append(c_idx)
                candidates = remaining_candidates
        
        # If we don't have enough samples, fill with FPS
        if len(sampled_indices) < self.num_samples:
            remaining_indices = [i for i in range(N) if i not in sampled_indices]
            if remaining_indices:
                fps = FarthestPointSampling(self.num_samples - len(sampled_indices))
                remaining_points = points[remaining_indices]
                additional_points, additional_local_indices = fps(remaining_points)
                additional_global_indices = [remaining_indices[i.item()] for i in additional_local_indices]
                sampled_indices.extend(additional_global_indices)
        
        indices = torch.tensor(sampled_indices[:self.num_samples], device=device)
        final_sampled_points = points[indices]
        
        return final_sampled_points, indices
    
    def _sample_batch(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Poisson disk sampling for batch of point clouds"""
        B, N, D = points.shape
        
        sampled_points_list = []
        indices_list = []
        
        for b in range(B):
            sampled_points_b, indices_b = self._sample_single(points[b])
            sampled_points_list.append(sampled_points_b)
            indices_list.append(indices_b)
        
        sampled_points = torch.stack(sampled_points_list)
        indices = torch.stack(indices_list)
        
        return sampled_points, indices


class AdaptiveSampling:
    """
    Adaptive sampling that combines multiple strategies based on point cloud characteristics
    
    Args:
        num_samples: Number of points to sample
        strategies: List of sampling strategies with weights
    """
    
    def __init__(self, 
                 num_samples: int,
                 strategies: List[Tuple[str, float]] = [('fps', 0.4), ('density', 0.4), ('voxel', 0.2)]):
        self.num_samples = num_samples
        self.strategies = strategies
        
        # Initialize samplers
        self.samplers = {}
        for strategy, _ in strategies:
            if strategy == 'fps':
                self.samplers[strategy] = FarthestPointSampling(num_samples)
            elif strategy == 'density':
                self.samplers[strategy] = DensityBasedSampling(num_samples)
            elif strategy == 'voxel':
                self.samplers[strategy] = VoxelGridSampling()
            elif strategy == 'poisson':
                self.samplers[strategy] = PoissonDiskSampling(num_samples)
    
    def analyze_point_cloud(self, points: torch.Tensor) -> Dict[str, float]:
        """
        Analyze point cloud characteristics to determine best sampling strategy
        
        Args:
            points: Input points (N, 3)
            
        Returns:
            characteristics: Dictionary of point cloud characteristics
        """
        N, D = points.shape
        
        # Compute basic statistics
        centroid = torch.mean(points, dim=0)
        distances_to_centroid = torch.norm(points - centroid, dim=1)
        
        # Density variation
        density_variation = torch.std(distances_to_centroid) / torch.mean(distances_to_centroid)
        
        # Local density variation
        k = min(16, N - 1)
        if k > 0:
            pairwise_distances = torch.cdist(points, points)
            pairwise_distances = pairwise_distances + torch.eye(N, device=points.device) * 1e10
            k_distances, _ = torch.topk(pairwise_distances, k, dim=1, largest=False)
            local_densities = 1.0 / (torch.mean(k_distances, dim=1) + 1e-8)
            local_density_variation = torch.std(local_densities) / torch.mean(local_densities)
        else:
            local_density_variation = torch.tensor(0.0)
        
        # Spatial distribution
        bbox_size = torch.max(points, dim=0)[0] - torch.min(points, dim=0)[0]
        volume = torch.prod(bbox_size)
        point_density = N / volume if volume > 0 else 1.0
        
        characteristics = {
            'density_variation': density_variation.item(),
            'local_density_variation': local_density_variation.item(),
            'point_density': point_density.item(),
            'num_points': N
        }
        
        return characteristics
    
    def select_strategy(self, characteristics: Dict[str, float]) -> str:
        """
        Select best sampling strategy based on characteristics
        
        Args:
            characteristics: Point cloud characteristics
            
        Returns:
            strategy: Selected strategy name
        """
        # High density variation -> density-based sampling
        if characteristics['local_density_variation'] > 0.5:
            return 'density'
        
        # Very uniform distribution -> voxel grid sampling
        elif characteristics['density_variation'] < 0.2:
            return 'voxel'
        
        # High point density -> Poisson disk sampling
        elif characteristics['point_density'] > 1000:
            return 'poisson'
        
        # Default to FPS
        else:
            return 'fps'
    
    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive sampling
        
        Args:
            points: Input points (N, 3) or (B, N, 3)
            
        Returns:
            sampled_points: Sampled points
            indices: Indices of sampled points
        """
        if len(points.shape) == 3:
            return self._sample_batch(points)
        else:
            return self._sample_single(points)
    
    def _sample_single(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive sampling for single point cloud"""
        # Analyze point cloud
        characteristics = self.analyze_point_cloud(points)
        
        # Select strategy
        strategy = self.select_strategy(characteristics)
        
        # Apply selected strategy
        if strategy in self.samplers:
            return self.samplers[strategy](points)
        else:
            # Fallback to FPS
            return self.samplers['fps'](points)
    
    def _sample_batch(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive sampling for batch of point clouds"""
        B, N, D = points.shape
        
        sampled_points_list = []
        indices_list = []
        
        for b in range(B):
            sampled_points_b, indices_b = self._sample_single(points[b])
            sampled_points_list.append(sampled_points_b)
            indices_list.append(indices_b)
        
        # Handle variable lengths
        max_len = max(sp.shape[0] for sp in sampled_points_list)
        
        padded_points = []
        padded_indices = []
        
        for sp, idx in zip(sampled_points_list, indices_list):
            if sp.shape[0] < max_len:
                # Pad with duplicated points
                padding = max_len - sp.shape[0]
                if sp.shape[0] > 0:
                    pad_points = sp[-1:].expand(padding, -1)
                    pad_indices = idx[-1:].expand(padding)
                else:
                    pad_points = torch.zeros(padding, D, device=points.device)
                    pad_indices = torch.zeros(padding, device=points.device, dtype=torch.long)
                
                sp = torch.cat([sp, pad_points], dim=0)
                idx = torch.cat([idx, pad_indices], dim=0)
            elif sp.shape[0] > max_len:
                sp = sp[:max_len]
                idx = idx[:max_len]
            
            padded_points.append(sp)
            padded_indices.append(idx)
        
        sampled_points = torch.stack(padded_points)
        indices = torch.stack(padded_indices)
        
        return sampled_points, indices