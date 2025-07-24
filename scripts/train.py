#!/usr/bin/env python3
"""
Main training script for PointNet VAE
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pointnet_vae.config import Config
from pointnet_vae.models import PointNetVAE
from pointnet_vae.training import PointNetVAETrainer
from pointnet_vae.utils import PointCloudVisualizer


class DummyPointCloudDataset:
    """
    Dummy point cloud dataset for demonstration
    In practice, replace this with your actual dataset
    """
    
    def __init__(self, num_samples: int = 1000, num_points: int = 2048, shape_type: str = 'sphere'):
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


def create_data_loaders(config: Config) -> tuple:
    """
    Create training and validation data loaders
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Create datasets
    train_dataset = DummyPointCloudDataset(
        num_samples=config.training.batch_size * 100,  # Dummy size
        num_points=config.model.num_points,
        shape_type='mixed'
    )
    
    val_dataset = DummyPointCloudDataset(
        num_samples=config.training.batch_size * 20,  # Dummy size
        num_points=config.model.num_points,
        shape_type='mixed'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train PointNet VAE')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    config.output_path = args.output_dir
    config.checkpoint_path = args.checkpoint_dir
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    config.device = str(device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config.to_dict()}")
    
    # Create output directories
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Save configuration
    config.save_yaml(os.path.join(config.output_path, 'config.yaml'))
    
    # Create model
    model = PointNetVAE(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    trainer = PointNetVAETrainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create visualizer for monitoring
    visualizer = PointCloudVisualizer()
    
    # Show a sample before training
    sample_batch = next(iter(val_loader))
    sample_batch = sample_batch[:4]  # First 4 samples
    
    with torch.no_grad():
        model.eval()
        reconstructed, _, _, _ = model(sample_batch.to(device))
        
        # Save comparison plot
        fig = visualizer.plot_batch_comparison(
            sample_batch.cpu(),
            reconstructed.cpu(),
            num_samples=4,
            title="Before Training"
        )
        fig.savefig(os.path.join(config.output_path, 'before_training.png'))
        plt.close(fig)
    
    # Start training
    print("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
        print("Training completed successfully!")
        
        # Show results after training
        with torch.no_grad():
            model.eval()
            reconstructed, _, _, _ = model(sample_batch.to(device))
            
            # Save comparison plot
            fig = visualizer.plot_batch_comparison(
                sample_batch.cpu(),
                reconstructed.cpu(),
                num_samples=4,
                title="After Training"
            )
            fig.savefig(os.path.join(config.output_path, 'after_training.png'))
            plt.close(fig)
            
            # Plot training curves
            fig = visualizer.plot_training_curves(
                trainer.training_history,
                save_path=os.path.join(config.output_path, 'training_curves.png')
            )
            plt.close(fig)
        
        print(f"Results saved to {config.output_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save checkpoint
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        print("Checkpoint saved")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    main()