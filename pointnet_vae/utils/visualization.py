"""
Visualization utilities for point clouds
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from typing import Optional, List, Tuple, Union
import os

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available. Some visualization features will be limited.")


class PointCloudVisualizer:
    """
    Comprehensive point cloud visualization utilities
    
    Args:
        figsize: Figure size for matplotlib plots
        style: Plotting style
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        self.figsize = figsize
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_style(style)
        
    def plot_point_cloud(self,
                        points: Union[torch.Tensor, np.ndarray],
                        colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
                        title: str = "Point Cloud",
                        save_path: Optional[str] = None,
                        show_axes: bool = True,
                        point_size: float = 1.0,
                        alpha: float = 0.8) -> plt.Figure:
        """
        Plot a single point cloud
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) or (N,) for scalar coloring
            title: Plot title
            save_path: Path to save figure
            show_axes: Whether to show axes
            point_size: Size of points
            alpha: Transparency
            
        Returns:
            fig: Matplotlib figure
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if colors is not None and isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            if len(colors.shape) == 1:
                # Scalar coloring
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                   c=colors, cmap='viridis', s=point_size, alpha=alpha)
                plt.colorbar(scatter)
            else:
                # RGB coloring
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, s=point_size, alpha=alpha)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=point_size, alpha=alpha)
        
        ax.set_title(title)
        if show_axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
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
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(self,
                       original: Union[torch.Tensor, np.ndarray],
                       reconstructed: Union[torch.Tensor, np.ndarray],
                       title: str = "Original vs Reconstructed",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot original and reconstructed point clouds side by side
        
        Args:
            original: Original point cloud (N, 3)
            reconstructed: Reconstructed point cloud (M, 3)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(16, 8))
        
        # Original point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], 
                   c='blue', s=1.0, alpha=0.8)
        ax1.set_title('Original')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Reconstructed point cloud
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                   c='red', s=1.0, alpha=0.8)
        ax2.set_title('Reconstructed')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set same scale for both plots
        all_points = np.vstack([original, reconstructed])
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                            all_points[:, 1].max() - all_points[:, 1].min(),
                            all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        for ax in [ax1, ax2]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_batch_comparison(self,
                            original_batch: Union[torch.Tensor, np.ndarray],
                            reconstructed_batch: Union[torch.Tensor, np.ndarray],
                            num_samples: int = 4,
                            title: str = "Batch Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple original and reconstructed point clouds
        
        Args:
            original_batch: Original point clouds (B, N, 3)
            reconstructed_batch: Reconstructed point clouds (B, M, 3)
            num_samples: Number of samples to plot
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        if isinstance(original_batch, torch.Tensor):
            original_batch = original_batch.detach().cpu().numpy()
        if isinstance(reconstructed_batch, torch.Tensor):
            reconstructed_batch = reconstructed_batch.detach().cpu().numpy()
        
        batch_size = min(original_batch.shape[0], num_samples)
        
        fig = plt.figure(figsize=(4 * batch_size, 8))
        
        for i in range(batch_size):
            # Original
            ax1 = fig.add_subplot(2, batch_size, i + 1, projection='3d')
            ax1.scatter(original_batch[i, :, 0], original_batch[i, :, 1], original_batch[i, :, 2],
                       c='blue', s=1.0, alpha=0.8)
            ax1.set_title(f'Original {i+1}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            
            # Reconstructed
            ax2 = fig.add_subplot(2, batch_size, batch_size + i + 1, projection='3d')
            ax2.scatter(reconstructed_batch[i, :, 0], reconstructed_batch[i, :, 1], reconstructed_batch[i, :, 2],
                       c='red', s=1.0, alpha=0.8)
            ax2.set_title(f'Reconstructed {i+1}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            
            # Set same scale
            all_points = np.vstack([original_batch[i], reconstructed_batch[i]])
            max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                                all_points[:, 1].max() - all_points[:, 1].min(),
                                all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            
            for ax in [ax1, ax2]:
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(self,
                           training_history: dict,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves
        
        Args:
            training_history: Dictionary with training metrics
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(training_history['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in training_history:
            axes[0, 0].plot(training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        if 'reconstruction_loss' in training_history:
            axes[0, 1].plot(training_history['reconstruction_loss'], label='Reconstruction Loss', color='green')
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # KL loss
        if 'kl_loss' in training_history:
            axes[1, 0].plot(training_history['kl_loss'], label='KL Loss', color='orange')
            axes[1, 0].set_title('KL Divergence Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in training_history:
            axes[1, 1].plot(training_history['lr'], label='Learning Rate', color='purple')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_latent_space(self,
                         latent_codes: Union[torch.Tensor, np.ndarray],
                         labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
                         method: str = 'pca',
                         title: str = "Latent Space Visualization",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            latent_codes: Latent codes (N, latent_dim)
            labels: Optional labels for coloring (N,)
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        if isinstance(latent_codes, torch.Tensor):
            latent_codes = latent_codes.detach().cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_codes = reducer.fit_transform(latent_codes)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            reduced_codes = reducer.fit_transform(latent_codes)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                reduced_codes = reducer.fit_transform(latent_codes)
            except ImportError:
                print("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                reduced_codes = reducer.fit_transform(latent_codes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            scatter = ax.scatter(reduced_codes[:, 0], reduced_codes[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(reduced_codes[:, 0], reduced_codes[:, 1], alpha=0.7)
        
        ax.set_title(f"{title} ({method.upper()})")
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_weights(self,
                              attention_weights: Union[torch.Tensor, np.ndarray],
                              points: Union[torch.Tensor, np.ndarray],
                              title: str = "Attention Weights",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention weights on point cloud
        
        Args:
            attention_weights: Attention weights (N,) or (N, num_heads)
            points: Point coordinates (N, 3)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        # If multi-head attention, average across heads
        if len(attention_weights.shape) > 1:
            attention_weights = np.mean(attention_weights, axis=1)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=attention_weights, cmap='hot', s=10, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax, label='Attention Weight')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_point_cloud_ply(self,
                            points: Union[torch.Tensor, np.ndarray],
                            colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
                            save_path: str = "point_cloud.ply"):
        """
        Save point cloud as PLY file
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3), values in [0, 1]
            save_path: Path to save PLY file
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available, cannot save PLY file")
            return
        
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if colors is not None and isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0  # Normalize to [0, 1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to {save_path}")
    
    def create_animation(self,
                        point_sequences: List[Union[torch.Tensor, np.ndarray]],
                        save_path: str = "animation.gif",
                        fps: int = 10,
                        duration: Optional[float] = None):
        """
        Create animation from sequence of point clouds
        
        Args:
            point_sequences: List of point clouds (each N, 3)
            save_path: Path to save animation
            fps: Frames per second
            duration: Duration in seconds (if None, uses fps)
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("Animation libraries not available")
            return
        
        # Convert to numpy
        sequences = []
        for points in point_sequences:
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            sequences.append(points)
        
        # Set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get global bounds
        all_points = np.vstack(sequences)
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                            all_points[:, 1].max() - all_points[:, 1].min(),
                            all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Animation function
        def animate(frame):
            ax.clear()
            points = sequences[frame]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, alpha=0.8)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_title(f'Frame {frame + 1}/{len(sequences)}')
            return ax,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(sequences), 
                           interval=1000//fps, blit=False, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
        
        plt.close(fig)