"""
Evaluation metrics for point cloud reconstruction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
import time


def chamfer_distance_numpy(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Chamfer Distance using numpy
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        
    Returns:
        cd_forward: Forward direction distance
        cd_backward: Backward direction distance
        cd_total: Total Chamfer distance
    """
    # Use sklearn for efficient nearest neighbor search
    nn_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pred)
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    
    # Forward direction: pred -> target
    distances_forward, _ = nn_target.kneighbors(pred)
    cd_forward = np.mean(distances_forward)
    
    # Backward direction: target -> pred
    distances_backward, _ = nn_pred.kneighbors(target)
    cd_backward = np.mean(distances_backward)
    
    # Total Chamfer distance
    cd_total = cd_forward + cd_backward
    
    return cd_forward, cd_backward, cd_total


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two point clouds
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        
    Returns:
        hausdorff_dist: Hausdorff distance
    """
    dist_forward = directed_hausdorff(pred, target)[0]
    dist_backward = directed_hausdorff(target, pred)[0]
    return max(dist_forward, dist_backward)


def earth_movers_distance_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Approximate Earth Mover's Distance using optimal transport
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        
    Returns:
        emd: Earth Mover's Distance
    """
    try:
        import ot
        
        # Compute cost matrix (squared distances)
        cost_matrix = np.sum((pred[:, None, :] - target[None, :, :]) ** 2, axis=2)
        
        # Uniform distributions
        a = np.ones(pred.shape[0]) / pred.shape[0]
        b = np.ones(target.shape[0]) / target.shape[0]
        
        # Solve optimal transport
        emd = ot.emd2(a, b, cost_matrix)
        
        return emd
    except ImportError:
        print("Python Optimal Transport (POT) not available, using approximation")
        # Fallback to Hungarian algorithm approximation
        from scipy.optimize import linear_sum_assignment
        
        cost_matrix = np.sum((pred[:, None, :] - target[None, :, :]) ** 2, axis=2)
        
        # Pad to square matrix
        n, m = cost_matrix.shape
        if n != m:
            if n < m:
                cost_matrix = np.vstack([cost_matrix, np.full((m - n, m), cost_matrix.max())])
            else:
                cost_matrix = np.hstack([cost_matrix, np.full((n, n - m), cost_matrix.max())])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        emd = cost_matrix[row_ind, col_ind].sum() / max(n, m)
        
        return emd


def coverage_metric(pred: np.ndarray, target: np.ndarray, threshold: float = 0.01) -> float:
    """
    Compute coverage metric (percentage of target points covered by prediction)
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        threshold: Distance threshold for coverage
        
    Returns:
        coverage: Coverage percentage [0, 1]
    """
    nn_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pred)
    distances, _ = nn_pred.kneighbors(target)
    covered = np.sum(distances.flatten() < threshold)
    return covered / len(target)


def completeness_metric(pred: np.ndarray, target: np.ndarray, threshold: float = 0.01) -> float:
    """
    Compute completeness metric (percentage of predicted points that are valid)
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        threshold: Distance threshold for validity
        
    Returns:
        completeness: Completeness percentage [0, 1]
    """
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    distances, _ = nn_target.kneighbors(pred)
    valid = np.sum(distances.flatten() < threshold)
    return valid / len(pred)


def f_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.01) -> Tuple[float, float, float]:
    """
    Compute F-score based on coverage and completeness
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        threshold: Distance threshold
        
    Returns:
        precision: Precision (completeness)
        recall: Recall (coverage)
        f_score: F-score
    """
    precision = completeness_metric(pred, target, threshold)
    recall = coverage_metric(pred, target, threshold)
    
    if precision + recall == 0:
        f_score_val = 0.0
    else:
        f_score_val = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f_score_val


def density_consistency_metric(pred: np.ndarray, target: np.ndarray, k: int = 16) -> float:
    """
    Compute density consistency metric
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        k: Number of neighbors for density estimation
        
    Returns:
        density_consistency: Density consistency score [0, 1]
    """
    def estimate_density(points, k):
        if len(points) <= k:
            return np.ones(len(points))
        
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
        distances, _ = nn.kneighbors(points)
        # Exclude self (first neighbor)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        densities = 1.0 / (mean_distances + 1e-8)
        return densities
    
    # Estimate densities
    pred_densities = estimate_density(pred, k)
    target_densities = estimate_density(target, k)
    
    # Normalize densities
    pred_densities = pred_densities / np.max(pred_densities)
    target_densities = target_densities / np.max(target_densities)
    
    # Find correspondences and compare densities
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    _, indices = nn_target.kneighbors(pred)
    
    corresponding_target_densities = target_densities[indices.flatten()]
    density_diff = np.abs(pred_densities - corresponding_target_densities)
    
    # Convert to similarity score
    consistency = 1.0 - np.mean(density_diff)
    return max(0.0, consistency)


def structural_similarity_metric(pred: np.ndarray, target: np.ndarray, radius: float = 0.05) -> float:
    """
    Compute structural similarity based on local neighborhoods
    
    Args:
        pred: Predicted points (N, 3)
        target: Target points (M, 3)
        radius: Radius for neighborhood analysis
        
    Returns:
        structural_similarity: Structural similarity score [0, 1]
    """
    def get_local_structure(points, radius):
        nn = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
        structures = []
        
        for point in points:
            neighbors = nn.radius_neighbors([point], return_distance=True)
            distances = neighbors[0][0]
            if len(distances) > 1:  # Exclude self
                structures.append(np.sort(distances[1:]))  # Exclude self
            else:
                structures.append(np.array([]))
        
        return structures
    
    pred_structures = get_local_structure(pred, radius)
    target_structures = get_local_structure(target, radius)
    
    # Find correspondences
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    _, indices = nn_target.kneighbors(pred)
    
    similarities = []
    for i, target_idx in enumerate(indices.flatten()):
        pred_struct = pred_structures[i]
        target_struct = target_structures[target_idx]
        
        if len(pred_struct) == 0 and len(target_struct) == 0:
            similarities.append(1.0)
        elif len(pred_struct) == 0 or len(target_struct) == 0:
            similarities.append(0.0)
        else:
            # Compare sorted distance arrays
            min_len = min(len(pred_struct), len(target_struct))
            if min_len > 0:
                pred_trunc = pred_struct[:min_len]
                target_trunc = target_struct[:min_len]
                similarity = 1.0 - np.mean(np.abs(pred_trunc - target_trunc) / (pred_trunc + target_trunc + 1e-8))
                similarities.append(max(0.0, similarity))
            else:
                similarities.append(0.0)
    
    return np.mean(similarities)


class PointCloudMetrics:
    """
    Comprehensive point cloud evaluation metrics
    
    Args:
        device: Computing device
        thresholds: List of thresholds for F-score computation
    """
    
    def __init__(self, device: torch.device, thresholds: List[float] = [0.001, 0.002, 0.005, 0.01, 0.02]):
        self.device = device
        self.thresholds = thresholds
        
    def compute_all_metrics(self, 
                           pred_batch: torch.Tensor, 
                           target_batch: torch.Tensor,
                           detailed: bool = True) -> Dict[str, float]:
        """
        Compute all metrics for a batch of point clouds
        
        Args:
            pred_batch: Predicted point clouds (B, N, 3)
            target_batch: Target point clouds (B, M, 3)
            detailed: Whether to compute detailed metrics
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        # Convert to numpy
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
        if isinstance(target_batch, torch.Tensor):
            target_batch = target_batch.detach().cpu().numpy()
        
        batch_size = pred_batch.shape[0]
        
        # Initialize metric accumulators
        metrics = {
            'chamfer_distance': 0.0,
            'chamfer_forward': 0.0,
            'chamfer_backward': 0.0,
        }
        
        if detailed:
            metrics.update({
                'hausdorff_distance': 0.0,
                'earth_movers_distance': 0.0,
                'density_consistency': 0.0,
                'structural_similarity': 0.0,
            })
            
            # F-score metrics for different thresholds
            for threshold in self.thresholds:
                metrics[f'precision_{threshold}'] = 0.0
                metrics[f'recall_{threshold}'] = 0.0
                metrics[f'f_score_{threshold}'] = 0.0
        
        # Compute metrics for each sample in batch
        for i in range(batch_size):
            pred = pred_batch[i]
            target = target_batch[i]
            
            # Chamfer distance
            cd_forward, cd_backward, cd_total = chamfer_distance_numpy(pred, target)
            metrics['chamfer_distance'] += cd_total
            metrics['chamfer_forward'] += cd_forward
            metrics['chamfer_backward'] += cd_backward
            
            if detailed:
                # Hausdorff distance
                hd = hausdorff_distance(pred, target)
                metrics['hausdorff_distance'] += hd
                
                # Earth Mover's Distance
                emd = earth_movers_distance_numpy(pred, target)
                metrics['earth_movers_distance'] += emd
                
                # Density consistency
                dc = density_consistency_metric(pred, target)
                metrics['density_consistency'] += dc
                
                # Structural similarity
                ss = structural_similarity_metric(pred, target)
                metrics['structural_similarity'] += ss
                
                # F-score for different thresholds
                for threshold in self.thresholds:
                    precision, recall, f_score_val = f_score(pred, target, threshold)
                    metrics[f'precision_{threshold}'] += precision
                    metrics[f'recall_{threshold}'] += recall
                    metrics[f'f_score_{threshold}'] += f_score_val
        
        # Average over batch
        for key in metrics:
            metrics[key] /= batch_size
        
        return metrics
    
    def compute_fast_metrics(self, 
                            pred_batch: torch.Tensor, 
                            target_batch: torch.Tensor) -> Dict[str, float]:
        """
        Compute only fast metrics (Chamfer distance)
        
        Args:
            pred_batch: Predicted point clouds (B, N, 3)
            target_batch: Target point clouds (B, M, 3)
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        return self.compute_all_metrics(pred_batch, target_batch, detailed=False)
    
    def evaluate_model(self, 
                      model: nn.Module, 
                      dataloader,
                      num_batches: Optional[int] = None,
                      detailed: bool = True) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            num_batches: Number of batches to evaluate (None for all)
            detailed: Whether to compute detailed metrics
            
        Returns:
            avg_metrics: Average metrics over dataset
        """
        model.eval()
        
        all_metrics = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward'):
                    reconstructed, _, _, _ = model(batch)
                else:
                    reconstructed = model(batch)
                
                # Compute metrics
                batch_metrics = self.compute_all_metrics(reconstructed, batch, detailed)
                all_metrics.append(batch_metrics)
                batch_count += 1
        
        # Average metrics
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def benchmark_inference_speed(self, 
                                 model: nn.Module, 
                                 sample_input: torch.Tensor,
                                 num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference speed
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_runs: Number of runs for benchmarking
            
        Returns:
            speed_metrics: Speed metrics
        """
        model.eval()
        sample_input = sample_input.to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        return {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'fps': fps,
            'num_runs': num_runs
        }
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        # Core metrics
        print(f"Chamfer Distance:     {metrics.get('chamfer_distance', 0):.6f}")
        print(f"  - Forward:          {metrics.get('chamfer_forward', 0):.6f}")
        print(f"  - Backward:         {metrics.get('chamfer_backward', 0):.6f}")
        
        if 'hausdorff_distance' in metrics:
            print(f"Hausdorff Distance:   {metrics['hausdorff_distance']:.6f}")
        
        if 'earth_movers_distance' in metrics:
            print(f"Earth Mover's Dist:   {metrics['earth_movers_distance']:.6f}")
        
        if 'density_consistency' in metrics:
            print(f"Density Consistency:  {metrics['density_consistency']:.4f}")
        
        if 'structural_similarity' in metrics:
            print(f"Structural Similarity: {metrics['structural_similarity']:.4f}")
        
        # F-score metrics
        f_score_metrics = {k: v for k, v in metrics.items() if k.startswith('f_score_')}
        if f_score_metrics:
            print("\nF-Score Metrics:")
            for threshold in self.thresholds:
                if f'f_score_{threshold}' in metrics:
                    precision = metrics[f'precision_{threshold}']
                    recall = metrics[f'recall_{threshold}']
                    f_score_val = metrics[f'f_score_{threshold}']
                    print(f"  Threshold {threshold:0.3f}: P={precision:.4f}, R={recall:.4f}, F={f_score_val:.4f}")
        
        print("="*50)