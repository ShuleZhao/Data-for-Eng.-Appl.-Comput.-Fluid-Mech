"""
Curriculum learning implementation for progressive training
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable
import numpy as np


class CurriculumScheduler:
    """
    Base class for curriculum learning schedulers
    
    Args:
        total_epochs: Total number of training epochs
        stages: List of curriculum stages
    """
    
    def __init__(self, total_epochs: int, stages: List[Dict[str, Any]]):
        self.total_epochs = total_epochs
        self.stages = stages
        self.current_epoch = 0
        self.current_stage = 0
    
    def step(self, epoch: int):
        """Update curriculum based on current epoch"""
        self.current_epoch = epoch
        
        # Find current stage
        for i, stage in enumerate(self.stages):
            if epoch >= stage.get('start_epoch', 0) and epoch < stage.get('end_epoch', float('inf')):
                self.current_stage = i
                break
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level [0, 1]"""
        if self.current_stage >= len(self.stages):
            return 1.0
        
        stage = self.stages[self.current_stage]
        return stage.get('difficulty', 1.0)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current curriculum configuration"""
        if self.current_stage >= len(self.stages):
            return self.stages[-1]
        
        return self.stages[self.current_stage]


class PointCloudCurriculumScheduler(CurriculumScheduler):
    """
    Curriculum scheduler specifically for point cloud tasks
    
    Args:
        total_epochs: Total training epochs
        initial_num_points: Initial number of points
        final_num_points: Final number of points
        noise_schedule: Schedule for noise injection
        complexity_schedule: Schedule for shape complexity
    """
    
    def __init__(self, 
                 total_epochs: int,
                 initial_num_points: int = 512,
                 final_num_points: int = 2048,
                 noise_schedule: Optional[List[float]] = None,
                 complexity_schedule: Optional[List[str]] = None):
        
        # Create curriculum stages
        stages = []
        
        # Stage 1: Simple shapes, fewer points, more noise
        stages.append({
            'start_epoch': 0,
            'end_epoch': total_epochs // 3,
            'num_points': initial_num_points,
            'noise_level': 0.02,
            'complexity': 'simple',
            'difficulty': 0.3
        })
        
        # Stage 2: Medium complexity, medium points, medium noise  
        stages.append({
            'start_epoch': total_epochs // 3,
            'end_epoch': 2 * total_epochs // 3,
            'num_points': (initial_num_points + final_num_points) // 2,
            'noise_level': 0.01,
            'complexity': 'medium',
            'difficulty': 0.6
        })
        
        # Stage 3: Full complexity, all points, minimal noise
        stages.append({
            'start_epoch': 2 * total_epochs // 3,
            'end_epoch': total_epochs,
            'num_points': final_num_points,
            'noise_level': 0.005,
            'complexity': 'complex',
            'difficulty': 1.0
        })
        
        super().__init__(total_epochs, stages)
    
    def get_num_points(self) -> int:
        """Get current number of points for training"""
        config = self.get_current_config()
        return config.get('num_points', 2048)
    
    def get_noise_level(self) -> float:
        """Get current noise level"""
        config = self.get_current_config()
        return config.get('noise_level', 0.01)
    
    def get_complexity_level(self) -> str:
        """Get current complexity level"""
        config = self.get_current_config()
        return config.get('complexity', 'complex')


class LossCurriculumScheduler:
    """
    Curriculum scheduler for loss function weights
    
    Args:
        loss_weights: Dictionary of loss weights with schedules
        total_epochs: Total training epochs
    """
    
    def __init__(self, 
                 loss_weights: Dict[str, Dict[str, Any]],
                 total_epochs: int):
        self.loss_weights = loss_weights
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def step(self, epoch: int):
        """Update loss weights based on current epoch"""
        self.current_epoch = epoch
    
    def get_loss_weight(self, loss_name: str) -> float:
        """Get current weight for a specific loss"""
        if loss_name not in self.loss_weights:
            return 1.0
        
        weight_config = self.loss_weights[loss_name]
        schedule_type = weight_config.get('schedule', 'constant')
        
        if schedule_type == 'constant':
            return weight_config.get('weight', 1.0)
        
        elif schedule_type == 'linear':
            start_weight = weight_config.get('start_weight', 0.0)
            end_weight = weight_config.get('end_weight', 1.0)
            start_epoch = weight_config.get('start_epoch', 0)
            end_epoch = weight_config.get('end_epoch', self.total_epochs)
            
            if self.current_epoch <= start_epoch:
                return start_weight
            elif self.current_epoch >= end_epoch:
                return end_weight
            else:
                progress = (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
                return start_weight + progress * (end_weight - start_weight)
        
        elif schedule_type == 'exponential':
            base_weight = weight_config.get('base_weight', 1.0)
            decay_rate = weight_config.get('decay_rate', 0.95)
            return base_weight * (decay_rate ** self.current_epoch)
        
        elif schedule_type == 'step':
            base_weight = weight_config.get('base_weight', 1.0)
            step_epochs = weight_config.get('step_epochs', [])
            step_factors = weight_config.get('step_factors', [])
            
            current_weight = base_weight
            for step_epoch, step_factor in zip(step_epochs, step_factors):
                if self.current_epoch >= step_epoch:
                    current_weight *= step_factor
            
            return current_weight
        
        return 1.0
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current loss weights"""
        return {name: self.get_loss_weight(name) for name in self.loss_weights.keys()}


class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum scheduler that adjusts based on training performance
    
    Args:
        difficulty_levels: List of difficulty configurations
        performance_threshold: Threshold for advancing to next level
        patience: Number of epochs to wait before advancing
        metric_name: Name of metric to monitor
    """
    
    def __init__(self, 
                 difficulty_levels: List[Dict[str, Any]],
                 performance_threshold: float = 0.9,
                 patience: int = 10,
                 metric_name: str = 'accuracy'):
        self.difficulty_levels = difficulty_levels
        self.performance_threshold = performance_threshold
        self.patience = patience
        self.metric_name = metric_name
        
        self.current_level = 0
        self.performance_history = []
        self.epochs_at_level = 0
        self.best_performance = 0.0
        self.epochs_since_improvement = 0
    
    def step(self, epoch: int, metrics: Dict[str, float]):
        """Update curriculum based on performance metrics"""
        current_performance = metrics.get(self.metric_name, 0.0)
        self.performance_history.append(current_performance)
        self.epochs_at_level += 1
        
        # Check if performance improved
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Check if ready to advance to next level
        if (current_performance >= self.performance_threshold and 
            self.epochs_since_improvement <= self.patience and
            self.current_level < len(self.difficulty_levels) - 1):
            
            self.current_level += 1
            self.epochs_at_level = 0
            self.epochs_since_improvement = 0
            print(f"Advanced to curriculum level {self.current_level}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current difficulty configuration"""
        return self.difficulty_levels[self.current_level]
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level"""
        config = self.get_current_config()
        return config.get('difficulty', 1.0)


class CurriculumDataSampler:
    """
    Data sampler that implements curriculum learning for point clouds
    
    Args:
        dataset: Dataset to sample from
        curriculum_scheduler: Curriculum scheduler
        sample_strategy: Strategy for sampling ('random', 'difficulty_based')
    """
    
    def __init__(self, 
                 dataset,
                 curriculum_scheduler: CurriculumScheduler,
                 sample_strategy: str = 'difficulty_based'):
        self.dataset = dataset
        self.curriculum_scheduler = curriculum_scheduler
        self.sample_strategy = sample_strategy
        
        # Assign difficulty scores to dataset items if not present
        if not hasattr(self.dataset, 'difficulty_scores'):
            self._assign_difficulty_scores()
    
    def _assign_difficulty_scores(self):
        """Assign difficulty scores to dataset items"""
        # Simple heuristic: more points = higher difficulty
        difficulty_scores = []
        
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if isinstance(item, dict) and 'points' in item:
                points = item['points']
            else:
                points = item
            
            # Difficulty based on number of points and geometric complexity
            num_points = len(points) if hasattr(points, '__len__') else 1000
            
            # Add geometric complexity (variance in distances)
            if hasattr(points, 'numpy'):
                points_np = points.numpy()
            else:
                points_np = np.array(points)
            
            if len(points_np.shape) == 2 and points_np.shape[1] >= 3:
                centroid = np.mean(points_np, axis=0)
                distances = np.linalg.norm(points_np - centroid, axis=1)
                complexity = np.std(distances)
            else:
                complexity = 1.0
            
            # Combined difficulty score
            difficulty = (num_points / 2048.0 + complexity) / 2.0
            difficulty_scores.append(min(1.0, max(0.0, difficulty)))
        
        self.dataset.difficulty_scores = difficulty_scores
    
    def get_curriculum_indices(self, batch_size: int) -> List[int]:
        """Get indices for current curriculum level"""
        current_difficulty = self.curriculum_scheduler.get_current_difficulty()
        
        if self.sample_strategy == 'random':
            return np.random.choice(len(self.dataset), size=batch_size, replace=True).tolist()
        
        elif self.sample_strategy == 'difficulty_based':
            # Sample items with difficulty <= current_difficulty
            valid_indices = []
            for i, difficulty in enumerate(self.dataset.difficulty_scores):
                if difficulty <= current_difficulty:
                    valid_indices.append(i)
            
            if not valid_indices:
                valid_indices = list(range(len(self.dataset)))
            
            # Sample from valid indices
            sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
            return sampled_indices.tolist()
        
        return list(range(batch_size))


class CurriculumAugmentation:
    """
    Curriculum-based data augmentation
    
    Args:
        curriculum_scheduler: Curriculum scheduler
        augmentation_configs: Dictionary of augmentation configurations
    """
    
    def __init__(self, 
                 curriculum_scheduler: CurriculumScheduler,
                 augmentation_configs: Dict[str, Dict[str, Any]]):
        self.curriculum_scheduler = curriculum_scheduler
        self.augmentation_configs = augmentation_configs
    
    def apply_augmentation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply curriculum-appropriate augmentation"""
        current_config = self.curriculum_scheduler.get_current_config()
        difficulty = current_config.get('difficulty', 1.0)
        
        augmented_points = points.clone()
        
        # Rotation augmentation
        if 'rotation' in self.augmentation_configs:
            rotation_config = self.augmentation_configs['rotation']
            max_angle = rotation_config.get('max_angle', 15.0)
            current_angle = max_angle * difficulty
            
            if current_angle > 0:
                angle = np.random.uniform(-current_angle, current_angle) * np.pi / 180
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                rotation_matrix = torch.tensor([
                    [cos_angle, -sin_angle, 0],
                    [sin_angle, cos_angle, 0],
                    [0, 0, 1]
                ], dtype=points.dtype, device=points.device)
                
                augmented_points = torch.matmul(augmented_points, rotation_matrix.T)
        
        # Noise augmentation (inverse curriculum - less noise as difficulty increases)
        if 'noise' in self.augmentation_configs:
            noise_config = self.augmentation_configs['noise']
            max_noise = noise_config.get('max_std', 0.01)
            current_noise = max_noise * (1.0 - difficulty * 0.8)  # Reduce noise as difficulty increases
            
            if current_noise > 0:
                noise = torch.randn_like(augmented_points) * current_noise
                augmented_points = augmented_points + noise
        
        # Scaling augmentation
        if 'scaling' in self.augmentation_configs:
            scaling_config = self.augmentation_configs['scaling']
            scale_range = scaling_config.get('scale_range', (0.9, 1.1))
            current_range = (1.0 - (1.0 - scale_range[0]) * difficulty,
                           1.0 + (scale_range[1] - 1.0) * difficulty)
            
            scale = np.random.uniform(current_range[0], current_range[1])
            augmented_points = augmented_points * scale
        
        return augmented_points


def create_default_curriculum(total_epochs: int) -> Dict[str, Any]:
    """
    Create default curriculum configuration
    
    Args:
        total_epochs: Total training epochs
        
    Returns:
        curriculum_config: Default curriculum configuration
    """
    return {
        'point_curriculum': PointCloudCurriculumScheduler(
            total_epochs=total_epochs,
            initial_num_points=512,
            final_num_points=2048
        ),
        'loss_curriculum': LossCurriculumScheduler(
            loss_weights={
                'chamfer': {
                    'schedule': 'constant',
                    'weight': 1.0
                },
                'emd': {
                    'schedule': 'linear',
                    'start_weight': 0.0,
                    'end_weight': 0.5,
                    'start_epoch': total_epochs // 4,
                    'end_epoch': total_epochs // 2
                },
                'feature_matching': {
                    'schedule': 'linear',
                    'start_weight': 0.0,
                    'end_weight': 0.1,
                    'start_epoch': total_epochs // 2,
                    'end_epoch': 3 * total_epochs // 4
                },
                'density': {
                    'schedule': 'linear',
                    'start_weight': 0.0,
                    'end_weight': 0.01,
                    'start_epoch': 3 * total_epochs // 4,
                    'end_epoch': total_epochs
                }
            },
            total_epochs=total_epochs
        ),
        'augmentation': {
            'rotation': {'max_angle': 15.0},
            'noise': {'max_std': 0.01},
            'scaling': {'scale_range': (0.9, 1.1)}
        }
    }