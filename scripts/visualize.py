#!/usr/bin/env python3
"""
Visualization script for PointNet VAE
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
from pointnet_vae.utils import PointCloudVisualizer


class DummyPointCloudDataset:
    """Dummy dataset for visualization (same as in other scripts)"""
    
    def __init__(self, num_samples: int = 50, num_points: int = 2048, shape_type: str = 'sphere'):
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_type = shape_type
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.shape_type == 'sphere':
            phi = np.random.uniform(0, np.pi, self.num_points)
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = np.random.uniform(0.8, 1.0, self.num_points)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points = np.stack([x, y, z], axis=1)
            
        elif self.shape_type == 'cube':
            points = np.random.uniform(-1, 1, (self.num_points, 3))
            
        elif self.shape_type == 'mixed':
            if np.random.random() < 0.5:
                phi = np.random.uniform(0, np.pi, self.num_points)
                theta = np.random.uniform(0, 2*np.pi, self.num_points)
                r = np.random.uniform(0.8, 1.0, self.num_points)
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                points = np.stack([x, y, z], axis=1)
            else:
                points = np.random.uniform(-1, 1, (self.num_points, 3))
        
        points += np.random.normal(0, 0.01, points.shape)
        return torch.from_numpy(points).float()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PointNetVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def create_reconstruction_gallery(model: torch.nn.Module,
                                dataloader: DataLoader,
                                device: torch.device,
                                output_dir: str,
                                num_samples: int = 16) -> None:
    """Create a gallery of reconstruction results"""
    print("Creating reconstruction gallery...")
    
    model.eval()
    visualizer = PointCloudVisualizer()
    
    # Get samples
    sample_batch = next(iter(dataloader))
    sample_batch = sample_batch[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed, mu, logvar, _ = model(sample_batch)
    
    # Create individual comparison plots
    for i in range(min(8, num_samples)):
        fig = visualizer.plot_comparison(
            sample_batch[i].cpu(),
            reconstructed[i].cpu(),
            title=f"Sample {i+1}: Original vs Reconstructed",
            save_path=os.path.join(output_dir, f'reconstruction_{i+1}.png')
        )
        plt.close(fig)
    
    # Create batch comparison
    fig = visualizer.plot_batch_comparison(
        sample_batch.cpu(),
        reconstructed.cpu(),
        num_samples=min(8, num_samples),
        title="Reconstruction Gallery",
        save_path=os.path.join(output_dir, 'reconstruction_gallery.png')
    )
    plt.close(fig)


def create_generation_gallery(model: torch.nn.Module,
                            device: torch.device,
                            output_dir: str,
                            num_samples: int = 16) -> None:
    """Create a gallery of generated samples"""
    print("Creating generation gallery...")
    
    model.eval()
    visualizer = PointCloudVisualizer()
    
    with torch.no_grad():
        generated = model.sample(num_samples, device)
    
    # Create individual plots
    for i in range(min(8, num_samples)):
        fig = visualizer.plot_point_cloud(
            generated[i].cpu(),
            title=f'Generated Sample {i+1}',
            save_path=os.path.join(output_dir, f'generated_{i+1}.png')
        )
        plt.close(fig)
    
    # Create grid of generated samples
    fig = plt.figure(figsize=(20, 20))
    for i in range(min(16, num_samples)):
        ax = fig.add_subplot(4, 4, i + 1, projection='3d')
        points = generated[i].cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, alpha=0.8, c='blue')
        ax.set_title(f'Generated {i+1}', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                            points[:, 1].max() - points[:, 1].min(),
                            points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle('Generated Samples Gallery', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_gallery.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_interpolation_sequences(model: torch.nn.Module,
                                 dataloader: DataLoader,
                                 device: torch.device,
                                 output_dir: str,
                                 num_sequences: int = 3,
                                 num_steps: int = 10) -> None:
    """Create interpolation sequences"""
    print("Creating interpolation sequences...")
    
    model.eval()
    visualizer = PointCloudVisualizer()
    
    sample_batch = next(iter(dataloader))
    
    for seq_idx in range(num_sequences):
        # Get two samples
        sample1 = sample_batch[seq_idx * 2].unsqueeze(0).to(device)
        sample2 = sample_batch[seq_idx * 2 + 1].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Encode
            mu1, _ = model.encode(sample1)
            mu2, _ = model.encode(sample2)
            
            # Interpolate
            interpolated_points = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interpolated_latent = (1 - alpha) * mu1 + alpha * mu2
                decoded = model.decode(interpolated_latent)
                interpolated_points.append(decoded[0])
        
        # Create static interpolation plot
        fig = plt.figure(figsize=(20, 4))
        for i, points in enumerate(interpolated_points):
            ax = fig.add_subplot(1, num_steps, i + 1, projection='3d')
            points_np = points.cpu().numpy()
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      s=1.0, alpha=0.8, c='red')
            ax.set_title(f'Step {i+1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # Set equal aspect ratio
            max_range = np.array([points_np[:, 0].max() - points_np[:, 0].min(),
                                points_np[:, 1].max() - points_np[:, 1].min(),
                                points_np[:, 2].max() - points_np[:, 2].min()]).max() / 2.0
            mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
            mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
            mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.suptitle(f'Latent Space Interpolation Sequence {seq_idx + 1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'interpolation_sequence_{seq_idx + 1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create animation
        points_list = [p.cpu() for p in interpolated_points]
        visualizer.create_animation(
            points_list,
            save_path=os.path.join(output_dir, f'interpolation_animation_{seq_idx + 1}.gif'),
            fps=2
        )


def analyze_latent_space(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        output_dir: str,
                        num_samples: int = 500) -> None:
    """Analyze and visualize latent space"""
    print("Analyzing latent space...")
    
    model.eval()
    visualizer = PointCloudVisualizer()
    
    # Collect latent codes
    latent_codes = []
    original_points = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            
            latent_codes.append(mu.cpu())
            original_points.append(batch.cpu())
            sample_count += batch.shape[0]
    
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    original_points = torch.cat(original_points, dim=0)[:num_samples]
    
    # Create latent space visualizations
    for method in ['pca', 'tsne']:
        fig = visualizer.plot_latent_space(
            latent_codes.numpy(),
            method=method,
            title=f'Latent Space Visualization ({method.upper()})',
            save_path=os.path.join(output_dir, f'latent_space_{method}.png')
        )
        plt.close(fig)
    
    # Analyze latent dimensions
    latent_std = torch.std(latent_codes, dim=0)
    latent_mean = torch.mean(latent_codes, dim=0)
    
    # Plot latent dimension statistics
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Standard deviations
    axes[0].bar(range(len(latent_std)), latent_std.numpy())
    axes[0].set_title('Standard Deviation of Latent Dimensions')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Standard Deviation')
    axes[0].grid(True, alpha=0.3)
    
    # Means
    axes[1].bar(range(len(latent_mean)), latent_mean.numpy())
    axes[1].set_title('Mean of Latent Dimensions')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Mean Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_dimension_analysis.png'), dpi=300)
    plt.close(fig)


def create_training_history_plots(checkpoint_path: str, output_dir: str) -> None:
    """Create training history plots if available"""
    print("Creating training history plots...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            
            visualizer = PointCloudVisualizer()
            fig = visualizer.plot_training_curves(
                history,
                save_path=os.path.join(output_dir, 'training_history.png')
            )
            plt.close(fig)
            print("Training history plots created successfully")
        else:
            print("No training history found in checkpoint")
    except Exception as e:
        print(f"Failed to create training history plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize PointNet VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples for analysis')
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
    
    # Create dataset
    dataset = DummyPointCloudDataset(
        num_samples=args.batch_size * 20,
        num_points=config.model.num_points,
        shape_type='mixed'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Reconstruction gallery
    create_reconstruction_gallery(model, dataloader, device, args.output_dir, num_samples=16)
    
    # 2. Generation gallery
    create_generation_gallery(model, device, args.output_dir, num_samples=16)
    
    # 3. Interpolation sequences
    create_interpolation_sequences(model, dataloader, device, args.output_dir, 
                                 num_sequences=3, num_steps=10)
    
    # 4. Latent space analysis
    analyze_latent_space(model, dataloader, device, args.output_dir, 
                        num_samples=args.num_samples)
    
    # 5. Training history (if available)
    create_training_history_plots(args.checkpoint, args.output_dir)
    
    print(f"\nVisualization completed! Results saved to {args.output_dir}")
    
    # Print summary
    print("\nGenerated visualizations:")
    print("- reconstruction_gallery.png: Comparison of original and reconstructed point clouds")
    print("- generation_gallery.png: Gallery of generated samples")
    print("- interpolation_sequence_*.png: Latent space interpolation sequences")
    print("- interpolation_animation_*.gif: Animated interpolation sequences")
    print("- latent_space_pca.png: PCA visualization of latent space")
    print("- latent_space_tsne.png: t-SNE visualization of latent space")
    print("- latent_dimension_analysis.png: Analysis of latent dimensions")
    print("- training_history.png: Training loss curves (if available)")


if __name__ == '__main__':
    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    main()