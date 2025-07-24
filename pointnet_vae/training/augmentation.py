"""
Data augmentation techniques for point clouds
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import random


class PointCloudAugmentation:
    """
    Base class for point cloud augmentation techniques
    """
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            return self.apply(points)
        return points
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomRotation(PointCloudAugmentation):
    """
    Random rotation augmentation
    
    Args:
        max_angle: Maximum rotation angle in degrees
        axes: Axes to rotate around ('x', 'y', 'z', 'xy', 'xyz')
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 max_angle: float = 15.0,
                 axes: str = 'xyz',
                 probability: float = 0.8):
        super().__init__(probability)
        self.max_angle = max_angle
        self.axes = axes
    
    def get_rotation_matrix(self, angles: Tuple[float, float, float]) -> torch.Tensor:
        """
        Get 3D rotation matrix from Euler angles
        
        Args:
            angles: Rotation angles (rx, ry, rz) in radians
            
        Returns:
            rotation_matrix: 3x3 rotation matrix
        """
        rx, ry, rz = angles
        
        # Rotation matrices for each axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rx), -torch.sin(rx)],
            [0, torch.sin(rx), torch.cos(rx)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [torch.cos(ry), 0, torch.sin(ry)],
            [0, 1, 0],
            [-torch.sin(ry), 0, torch.cos(ry)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [torch.cos(rz), -torch.sin(rz), 0],
            [torch.sin(rz), torch.cos(rz), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Combined rotation
        R = torch.mm(torch.mm(Rz, Ry), Rx)
        return R
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random rotation"""
        device = points.device
        max_rad = self.max_angle * np.pi / 180
        
        # Generate random angles
        angles = [0.0, 0.0, 0.0]
        if 'x' in self.axes:
            angles[0] = random.uniform(-max_rad, max_rad)
        if 'y' in self.axes:
            angles[1] = random.uniform(-max_rad, max_rad)
        if 'z' in self.axes:
            angles[2] = random.uniform(-max_rad, max_rad)
        
        # Get rotation matrix
        R = self.get_rotation_matrix(angles).to(device)
        
        # Apply rotation
        if len(points.shape) == 3:  # Batch dimension
            return torch.matmul(points, R.T)
        else:  # Single point cloud
            return torch.matmul(points, R.T)


class RandomScaling(PointCloudAugmentation):
    """
    Random scaling augmentation
    
    Args:
        scale_range: Range of scaling factors (min, max)
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 probability: float = 0.8):
        super().__init__(probability)
        self.scale_range = scale_range
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random scaling"""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        return points * scale


class RandomTranslation(PointCloudAugmentation):
    """
    Random translation augmentation
    
    Args:
        translation_range: Maximum translation in each direction
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 translation_range: float = 0.1,
                 probability: float = 0.5):
        super().__init__(probability)
        self.translation_range = translation_range
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random translation"""
        if len(points.shape) == 3:  # Batch dimension
            B, N, D = points.shape
            translation = torch.FloatTensor(B, 1, D).uniform_(-self.translation_range, self.translation_range)
            translation = translation.to(points.device)
        else:  # Single point cloud
            N, D = points.shape
            translation = torch.FloatTensor(1, D).uniform_(-self.translation_range, self.translation_range)
            translation = translation.to(points.device)
        
        return points + translation


class RandomNoise(PointCloudAugmentation):
    """
    Random Gaussian noise augmentation
    
    Args:
        noise_std: Standard deviation of Gaussian noise
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 probability: float = 0.8):
        super().__init__(probability)
        self.noise_std = noise_std
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random noise"""
        noise = torch.randn_like(points) * self.noise_std
        return points + noise


class RandomJitter(PointCloudAugmentation):
    """
    Random jittering (small random displacements)
    
    Args:
        jitter_std: Standard deviation of jittering
        jitter_clip: Clipping range for jittering
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 jitter_std: float = 0.01,
                 jitter_clip: float = 0.05,
                 probability: float = 0.8):
        super().__init__(probability)
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random jittering"""
        jitter = torch.randn_like(points) * self.jitter_std
        jitter = torch.clamp(jitter, -self.jitter_clip, self.jitter_clip)
        return points + jitter


class RandomDropout(PointCloudAugmentation):
    """
    Random point dropout augmentation
    
    Args:
        dropout_ratio: Ratio of points to drop
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 dropout_ratio: float = 0.1,
                 probability: float = 0.5):
        super().__init__(probability)
        self.dropout_ratio = dropout_ratio
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random dropout"""
        if len(points.shape) == 3:  # Batch dimension
            B, N, D = points.shape
            num_keep = int(N * (1 - self.dropout_ratio))
            
            augmented_points = []
            for b in range(B):
                indices = torch.randperm(N, device=points.device)[:num_keep]
                augmented_points.append(points[b, indices])
            
            # Pad to original size with duplicated points
            for b in range(B):
                while augmented_points[b].shape[0] < N:
                    num_duplicate = min(N - augmented_points[b].shape[0], augmented_points[b].shape[0])
                    indices = torch.randperm(augmented_points[b].shape[0], device=points.device)[:num_duplicate]
                    augmented_points[b] = torch.cat([augmented_points[b], augmented_points[b][indices]], dim=0)
            
            return torch.stack(augmented_points)
        else:  # Single point cloud
            N, D = points.shape
            num_keep = int(N * (1 - self.dropout_ratio))
            indices = torch.randperm(N, device=points.device)[:num_keep]
            
            # Duplicate points to maintain size
            kept_points = points[indices]
            while kept_points.shape[0] < N:
                num_duplicate = min(N - kept_points.shape[0], kept_points.shape[0])
                dup_indices = torch.randperm(kept_points.shape[0], device=points.device)[:num_duplicate]
                kept_points = torch.cat([kept_points, kept_points[dup_indices]], dim=0)
            
            return kept_points[:N]


class RandomCrop(PointCloudAugmentation):
    """
    Random cropping augmentation
    
    Args:
        crop_ratio: Ratio of the bounding box to keep
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 crop_ratio: float = 0.8,
                 probability: float = 0.3):
        super().__init__(probability)
        self.crop_ratio = crop_ratio
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random cropping"""
        if len(points.shape) == 3:  # Batch dimension
            B, N, D = points.shape
            augmented_points = []
            
            for b in range(B):
                points_b = points[b]
                
                # Get bounding box
                min_vals = torch.min(points_b, dim=0)[0]
                max_vals = torch.max(points_b, dim=0)[0]
                
                # Random crop box
                center = (min_vals + max_vals) / 2
                box_size = (max_vals - min_vals) * self.crop_ratio
                
                crop_min = center - box_size / 2
                crop_max = center + box_size / 2
                
                # Find points within crop box
                mask = torch.all((points_b >= crop_min) & (points_b <= crop_max), dim=1)
                cropped_points = points_b[mask]
                
                # If too few points, use original
                if cropped_points.shape[0] < N // 2:
                    augmented_points.append(points_b)
                else:
                    # Pad or sample to maintain size
                    if cropped_points.shape[0] >= N:
                        indices = torch.randperm(cropped_points.shape[0], device=points.device)[:N]
                        augmented_points.append(cropped_points[indices])
                    else:
                        # Duplicate points to maintain size
                        while cropped_points.shape[0] < N:
                            num_duplicate = min(N - cropped_points.shape[0], cropped_points.shape[0])
                            indices = torch.randperm(cropped_points.shape[0], device=points.device)[:num_duplicate]
                            cropped_points = torch.cat([cropped_points, cropped_points[indices]], dim=0)
                        augmented_points.append(cropped_points[:N])
            
            return torch.stack(augmented_points)
        else:
            # Single point cloud version
            N, D = points.shape
            
            min_vals = torch.min(points, dim=0)[0]
            max_vals = torch.max(points, dim=0)[0]
            
            center = (min_vals + max_vals) / 2
            box_size = (max_vals - min_vals) * self.crop_ratio
            
            crop_min = center - box_size / 2
            crop_max = center + box_size / 2
            
            mask = torch.all((points >= crop_min) & (points <= crop_max), dim=1)
            cropped_points = points[mask]
            
            if cropped_points.shape[0] < N // 2:
                return points
            
            if cropped_points.shape[0] >= N:
                indices = torch.randperm(cropped_points.shape[0], device=points.device)[:N]
                return cropped_points[indices]
            else:
                while cropped_points.shape[0] < N:
                    num_duplicate = min(N - cropped_points.shape[0], cropped_points.shape[0])
                    indices = torch.randperm(cropped_points.shape[0], device=points.device)[:num_duplicate]
                    cropped_points = torch.cat([cropped_points, cropped_points[indices]], dim=0)
                return cropped_points[:N]


class RandomFlip(PointCloudAugmentation):
    """
    Random flipping augmentation
    
    Args:
        axes: Axes to flip ('x', 'y', 'z', 'xy', 'xyz')
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 axes: str = 'xyz',
                 probability: float = 0.5):
        super().__init__(probability)
        self.axes = axes
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random flipping"""
        flipped_points = points.clone()
        
        if 'x' in self.axes and random.random() < 0.5:
            flipped_points[..., 0] = -flipped_points[..., 0]
        if 'y' in self.axes and random.random() < 0.5:
            flipped_points[..., 1] = -flipped_points[..., 1]
        if 'z' in self.axes and random.random() < 0.5:
            flipped_points[..., 2] = -flipped_points[..., 2]
        
        return flipped_points


class ElasticDeformation(PointCloudAugmentation):
    """
    Elastic deformation augmentation
    
    Args:
        deformation_strength: Strength of deformation
        num_control_points: Number of control points for deformation
        probability: Probability of applying augmentation
    """
    
    def __init__(self, 
                 deformation_strength: float = 0.1,
                 num_control_points: int = 4,
                 probability: float = 0.3):
        super().__init__(probability)
        self.deformation_strength = deformation_strength
        self.num_control_points = num_control_points
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation"""
        if len(points.shape) == 3:  # Batch dimension
            B, N, D = points.shape
            deformed_points = []
            
            for b in range(B):
                deformed = self._deform_single(points[b])
                deformed_points.append(deformed)
            
            return torch.stack(deformed_points)
        else:
            return self._deform_single(points)
    
    def _deform_single(self, points: torch.Tensor) -> torch.Tensor:
        """Apply deformation to single point cloud"""
        N, D = points.shape
        device = points.device
        
        # Create control points
        min_vals = torch.min(points, dim=0)[0]
        max_vals = torch.max(points, dim=0)[0]
        
        control_points = []
        for _ in range(self.num_control_points):
            control_point = torch.rand(D, device=device) * (max_vals - min_vals) + min_vals
            control_points.append(control_point)
        control_points = torch.stack(control_points)
        
        # Generate random deformation vectors
        deformation_vectors = torch.randn(self.num_control_points, D, device=device) * self.deformation_strength
        
        # Apply deformation using inverse distance weighting
        deformed_points = points.clone()
        
        for i in range(N):
            point = points[i]
            total_weight = 0.0
            deformation = torch.zeros(D, device=device)
            
            for j in range(self.num_control_points):
                distance = torch.norm(point - control_points[j]) + 1e-8
                weight = 1.0 / (distance ** 2)
                deformation += weight * deformation_vectors[j]
                total_weight += weight
            
            deformation = deformation / total_weight
            deformed_points[i] = point + deformation
        
        return deformed_points


class ComposedAugmentation:
    """
    Composition of multiple augmentation techniques
    
    Args:
        augmentations: List of augmentation techniques
    """
    
    def __init__(self, augmentations: List[PointCloudAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations in sequence"""
        augmented_points = points
        for aug in self.augmentations:
            augmented_points = aug(augmented_points)
        return augmented_points


def create_default_augmentation_pipeline(config: Dict[str, Any]) -> ComposedAugmentation:
    """
    Create default augmentation pipeline
    
    Args:
        config: Augmentation configuration
        
    Returns:
        augmentation_pipeline: Composed augmentation pipeline
    """
    augmentations = []
    
    # Rotation
    if config.get('rotation', {}).get('enabled', True):
        rotation_config = config.get('rotation', {})
        augmentations.append(RandomRotation(
            max_angle=rotation_config.get('max_angle', 15.0),
            axes=rotation_config.get('axes', 'xyz'),
            probability=rotation_config.get('probability', 0.8)
        ))
    
    # Scaling
    if config.get('scaling', {}).get('enabled', True):
        scaling_config = config.get('scaling', {})
        augmentations.append(RandomScaling(
            scale_range=scaling_config.get('scale_range', (0.8, 1.2)),
            probability=scaling_config.get('probability', 0.8)
        ))
    
    # Translation
    if config.get('translation', {}).get('enabled', False):
        translation_config = config.get('translation', {})
        augmentations.append(RandomTranslation(
            translation_range=translation_config.get('translation_range', 0.1),
            probability=translation_config.get('probability', 0.5)
        ))
    
    # Noise
    if config.get('noise', {}).get('enabled', True):
        noise_config = config.get('noise', {})
        augmentations.append(RandomNoise(
            noise_std=noise_config.get('noise_std', 0.01),
            probability=noise_config.get('probability', 0.8)
        ))
    
    # Jittering
    if config.get('jitter', {}).get('enabled', True):
        jitter_config = config.get('jitter', {})
        augmentations.append(RandomJitter(
            jitter_std=jitter_config.get('jitter_std', 0.01),
            jitter_clip=jitter_config.get('jitter_clip', 0.05),
            probability=jitter_config.get('probability', 0.8)
        ))
    
    # Dropout
    if config.get('dropout', {}).get('enabled', False):
        dropout_config = config.get('dropout', {})
        augmentations.append(RandomDropout(
            dropout_ratio=dropout_config.get('dropout_ratio', 0.1),
            probability=dropout_config.get('probability', 0.5)
        ))
    
    # Flipping
    if config.get('flip', {}).get('enabled', False):
        flip_config = config.get('flip', {})
        augmentations.append(RandomFlip(
            axes=flip_config.get('axes', 'xyz'),
            probability=flip_config.get('probability', 0.5)
        ))
    
    return ComposedAugmentation(augmentations)