#!/usr/bin/env python3
"""
Evaluation script for PointNet VAE
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pointnet_vae.config import Config
from pointnet_vae.models import PointNetVAE
from pointnet_vae.utils import PointCloudMetrics, PointCloudVisualizer


class DummyPointCloudDataset:
    """Dummy dataset for evaluation (same as in train.py)"""
    
    def __init__(self, num_samples: int = 100, num_points: int = 2048, shape_type: str = 'sphere'):
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_type = shape_type
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.shape_type == 'sphere':
            # Generate sphere
            phi = np.random.uniform(0, np.pi, self.num_points)
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = np.random.uniform(0.8, 1.0, self.num_points)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points = np.stack([x, y, z], axis=1)
            
        elif self.shape_type == 'cube':
            # Generate cube
            points = np.random.uniform(-1, 1, (self.num_points, 3))
            
        elif self.shape_type == 'mixed':
            # Mix of shapes
            if np.random.random() < 0.5:
                # Sphere
                phi = np.random.uniform(0, np.pi, self.num_points)
                theta = np.random.uniform(0, 2*np.pi, self.num_points)
                r = np.random.uniform(0.8, 1.0, self.num_points)
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                points = np.stack([x, y, z], axis=1)
            else:
                # Cube
                points = np.random.uniform(-1, 1, (self.num_points, 3))
        
        # Add some noise
        points += np.random.normal(0, 0.01, points.shape)
        
        return torch.from_numpy(points).float()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = PointNetVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, config


def evaluate_reconstruction(model: torch.nn.Module, 
                          dataloader: DataLoader, 
                          metrics: PointCloudMetrics,
                          device: torch.device,
                          num_batches: int = None) -> dict:
    """
    Evaluate reconstruction quality
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        metrics: Metrics calculator
        device: Device
        num_batches: Number of batches to evaluate
        
    Returns:
        results: Evaluation results
    """
    print("Evaluating reconstruction quality...")
    
    # Compute detailed metrics
    detailed_metrics = metrics.evaluate_model(
        model, dataloader, num_batches, detailed=True
    )
    
    # Benchmark inference speed
    sample_batch = next(iter(dataloader))
    speed_metrics = metrics.benchmark_inference_speed(
        model, sample_batch[:1].to(device), num_runs=100
    )
    
    results = {
        'reconstruction_metrics': detailed_metrics,
        'speed_metrics': speed_metrics
    }
    
    return results


def evaluate_latent_space(model: torch.nn.Module,
                         dataloader: DataLoader,
                         device: torch.device,
                         output_dir: str,
                         num_samples: int = 500) -> dict:
    """
    Evaluate latent space properties
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device
        output_dir: Output directory
        num_samples: Number of samples to analyze
        
    Returns:
        latent_metrics: Latent space metrics
    """
    print("Evaluating latent space...")
    
    model.eval()
    latent_codes = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            
            # Use mean for evaluation
            latent_codes.append(mu.cpu())
            sample_count += batch.shape[0]
    
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    
    # Compute latent space statistics
    latent_mean = torch.mean(latent_codes, dim=0)
    latent_std = torch.std(latent_codes, dim=0)
    
    # Compute effective dimensionality (number of dimensions with significant variance)
    variance_threshold = 0.01
    effective_dims = torch.sum(latent_std > variance_threshold).item()
    
    # Visualize latent space
    visualizer = PointCloudVisualizer()
    fig = visualizer.plot_latent_space(
        latent_codes.numpy(),
        method='pca',
        title='Latent Space (PCA)',
        save_path=os.path.join(output_dir, 'latent_space_pca.png')
    )
    plt.close(fig)
    
    fig = visualizer.plot_latent_space(
        latent_codes.numpy(),
        method='tsne',
        title='Latent Space (t-SNE)',
        save_path=os.path.join(output_dir, 'latent_space_tsne.png')
    )
    plt.close(fig)
    
    latent_metrics = {
        'latent_dim': latent_codes.shape[1],
        'effective_dims': effective_dims,
        'mean_magnitude': torch.norm(latent_mean).item(),
        'mean_std': torch.mean(latent_std).item(),
        'std_std': torch.std(latent_std).item()
    }
    
    return latent_metrics


def generate_samples(model: torch.nn.Module,
                    device: torch.device,
                    output_dir: str,
                    num_samples: int = 16) -> None:
    """
    Generate samples from the model
    
    Args:
        model: Model to use for generation
        device: Device
        output_dir: Output directory
        num_samples: Number of samples to generate
    """
    print("Generating samples...")
    
    model.eval()
    
    with torch.no_grad():
        # Generate random samples
        generated = model.sample(num_samples, device)
        
        # Visualize generated samples
        visualizer = PointCloudVisualizer()
        
        # Plot individual samples
        for i in range(min(4, num_samples)):
            fig = visualizer.plot_point_cloud(
                generated[i].cpu(),
                title=f'Generated Sample {i+1}',
                save_path=os.path.join(output_dir, f'generated_sample_{i+1}.png')
            )
            plt.close(fig)
        
        # Plot batch of samples
        if num_samples >= 4:
            fig = plt.figure(figsize=(16, 16))
            for i in range(min(16, num_samples)):
                ax = fig.add_subplot(4, 4, i + 1, projection='3d')
                points = generated[i].cpu().numpy()
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, alpha=0.8)
                ax.set_title(f'Sample {i+1}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'generated_samples_grid.png'), dpi=300)
            plt.close(fig)


def interpolate_latent_space(model: torch.nn.Module,
                           dataloader: DataLoader,
                           device: torch.device,
                           output_dir: str,
                           num_steps: int = 10) -> None:
    """
    Perform latent space interpolation
    
    Args:
        model: Model to use
        dataloader: Data loader
        device: Device
        output_dir: Output directory
        num_steps: Number of interpolation steps
    """
    print("Performing latent space interpolation...")
    
    model.eval()
    
    # Get two random samples
    sample_batch = next(iter(dataloader))
    sample1 = sample_batch[0:1].to(device)
    sample2 = sample_batch[1:2].to(device)
    
    with torch.no_grad():
        # Encode samples
        mu1, _ = model.encode(sample1)
        mu2, _ = model.encode(sample2)
        
        # Interpolate in latent space
        interpolated_points = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_latent = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode
            decoded = model.decode(interpolated_latent)
            interpolated_points.append(decoded[0])
        
        # Create animation
        visualizer = PointCloudVisualizer()
        points_list = [p.cpu() for p in interpolated_points]
        visualizer.create_animation(
            points_list,
            save_path=os.path.join(output_dir, 'latent_interpolation.gif'),
            fps=2
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluate PointNet VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Number of batches to evaluate (None for all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples for latent space analysis')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create test dataset
    test_dataset = DummyPointCloudDataset(
        num_samples=args.batch_size * 50,
        num_points=config.model.num_points,
        shape_type='mixed'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Initialize metrics
    metrics = PointCloudMetrics(device)
    
    # Evaluate reconstruction quality
    reconstruction_results = evaluate_reconstruction(
        model, test_loader, metrics, device, args.num_batches
    )
    
    # Print results
    print("\n" + "="*60)
    print("RECONSTRUCTION EVALUATION RESULTS")
    print("="*60)
    metrics.print_metrics(reconstruction_results['reconstruction_metrics'])
    
    print("\nSPEED METRICS:")
    speed_metrics = reconstruction_results['speed_metrics']
    print(f"Average inference time: {speed_metrics['avg_inference_time']:.4f}s")
    print(f"Throughput: {speed_metrics['fps']:.2f} FPS")
    
    # Evaluate latent space
    latent_results = evaluate_latent_space(
        model, test_loader, device, args.output_dir, args.num_samples
    )
    
    print("\nLATENT SPACE METRICS:")
    print(f"Latent dimension: {latent_results['latent_dim']}")
    print(f"Effective dimensions: {latent_results['effective_dims']}")
    print(f"Mean magnitude: {latent_results['mean_magnitude']:.4f}")
    print(f"Mean std: {latent_results['mean_std']:.4f}")
    print(f"Std of std: {latent_results['std_std']:.4f}")
    
    # Generate samples
    generate_samples(model, device, args.output_dir, num_samples=16)
    
    # Latent space interpolation
    interpolate_latent_space(model, test_loader, device, args.output_dir, num_steps=10)
    
    # Create reconstruction comparison plots
    visualizer = PointCloudVisualizer()
    sample_batch = next(iter(test_loader))
    sample_batch = sample_batch[:4].to(device)
    
    with torch.no_grad():
        reconstructed, _, _, _ = model(sample_batch)
        
        fig = visualizer.plot_batch_comparison(
            sample_batch.cpu(),
            reconstructed.cpu(),
            num_samples=4,
            title="Reconstruction Results",
            save_path=os.path.join(args.output_dir, 'reconstruction_comparison.png')
        )
        plt.close(fig)
    
    # Save results
    results_summary = {
        'reconstruction_metrics': reconstruction_results['reconstruction_metrics'],
        'speed_metrics': reconstruction_results['speed_metrics'],
        'latent_metrics': latent_results,
        'config': config.to_dict()
    }
    
    import json
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    main()