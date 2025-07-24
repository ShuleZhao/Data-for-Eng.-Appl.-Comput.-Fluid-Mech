"""
Main trainer class for PointNet VAE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import numpy as np

from ..models import PointNetVAE
from ..losses import *
from .curriculum import PointCloudCurriculumScheduler, LossCurriculumScheduler
from .augmentation import create_default_augmentation_pipeline


class PointNetVAETrainer:
    """
    Comprehensive trainer for PointNet VAE with advanced features
    
    Args:
        model: PointNet VAE model
        config: Training configuration
        device: Training device
    """
    
    def __init__(self, 
                 model: PointNetVAE,
                 config,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup logging
        self.setup_logging()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize loss functions
        self.setup_loss_functions()
        
        # Initialize curriculum learning
        self.setup_curriculum()
        
        # Initialize data augmentation
        self.setup_augmentation()
        
        # Initialize tensorboard
        if config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(config.output_path, 'tensorboard'))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'lr': []
        }
        
        # Progressive training state
        self.progressive_stage = 0
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.output_path, exist_ok=True)
        log_file = os.path.join(self.config.output_path, 'training.log')
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizers(self):
        """Setup optimizers and learning rate schedulers"""
        # Main optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        if self.config.training.lr_scheduler == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.lr_scheduler == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_factor
            )
        elif self.config.training.lr_scheduler == 'exponential':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_decay_factor
            )
        else:
            self.lr_scheduler = None
        
        # Discriminator optimizer (for adversarial training)
        if self.config.training.use_adversarial:
            self.discriminator = self.create_discriminator()
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.training.discriminator_lr,
                weight_decay=1e-4
            )
    
    def create_discriminator(self) -> nn.Module:
        """Create discriminator for adversarial training"""
        return nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def setup_loss_functions(self):
        """Setup loss functions with weights"""
        self.loss_functions = {}
        
        # Reconstruction losses
        self.loss_functions['chamfer'] = ChamferLoss(reduction='mean')
        
        if self.config.loss.emd_weight > 0:
            self.loss_functions['emd'] = ApproximateEMDLoss(reduction='mean')
        
        if self.config.loss.feature_matching_weight > 0:
            self.loss_functions['feature_matching'] = LocalFeatureMatchingLoss()
        
        if self.config.loss.use_multiscale_loss:
            base_chamfer = ChamferLoss(reduction='mean')
            self.loss_functions['multiscale'] = MultiScaleLoss(
                base_loss=base_chamfer,
                scales=[1.0, 0.5, 0.25],
                weights=self.config.loss.multiscale_weights
            )
        
        if self.config.loss.density_weight > 0:
            self.loss_functions['density'] = DensityRegularizationLoss()
        
        # Adversarial loss
        if self.config.training.use_adversarial:
            self.loss_functions['adversarial'] = nn.BCELoss()
    
    def setup_curriculum(self):
        """Setup curriculum learning"""
        if self.config.training.use_curriculum:
            self.point_curriculum = PointCloudCurriculumScheduler(
                total_epochs=self.config.training.num_epochs,
                initial_num_points=self.config.training.curriculum_epochs[0] if self.config.training.curriculum_epochs else 512,
                final_num_points=self.config.model.num_points
            )
            
            self.loss_curriculum = LossCurriculumScheduler(
                loss_weights={
                    'chamfer': {'schedule': 'constant', 'weight': self.config.loss.chamfer_weight},
                    'emd': {
                        'schedule': 'linear',
                        'start_weight': 0.0,
                        'end_weight': self.config.loss.emd_weight,
                        'start_epoch': self.config.training.num_epochs // 4,
                        'end_epoch': self.config.training.num_epochs // 2
                    },
                    'feature_matching': {
                        'schedule': 'linear',
                        'start_weight': 0.0,
                        'end_weight': self.config.loss.feature_matching_weight,
                        'start_epoch': self.config.training.num_epochs // 2,
                        'end_epoch': 3 * self.config.training.num_epochs // 4
                    },
                    'density': {
                        'schedule': 'linear',
                        'start_weight': 0.0,
                        'end_weight': self.config.loss.density_weight,
                        'start_epoch': 3 * self.config.training.num_epochs // 4,
                        'end_epoch': self.config.training.num_epochs
                    }
                },
                total_epochs=self.config.training.num_epochs
            )
        else:
            self.point_curriculum = None
            self.loss_curriculum = None
    
    def setup_augmentation(self):
        """Setup data augmentation"""
        if self.config.training.use_augmentation:
            aug_config = {
                'rotation': {
                    'enabled': True,
                    'max_angle': self.config.training.rotation_range,
                    'probability': 0.8
                },
                'scaling': {
                    'enabled': True,
                    'scale_range': self.config.training.scaling_range,
                    'probability': 0.8
                },
                'noise': {
                    'enabled': True,
                    'noise_std': self.config.training.noise_std,
                    'probability': 0.8
                },
                'jitter': {
                    'enabled': True,
                    'jitter_std': 0.01,
                    'probability': 0.8
                }
            }
            self.augmentation = create_default_augmentation_pipeline(aug_config)
        else:
            self.augmentation = None
    
    def compute_losses(self, 
                      pred_points: torch.Tensor, 
                      target_points: torch.Tensor,
                      mu: torch.Tensor,
                      logvar: torch.Tensor,
                      epoch: int) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        
        Args:
            pred_points: Predicted point cloud
            target_points: Target point cloud
            mu: Latent mean
            logvar: Latent log variance
            epoch: Current epoch
            
        Returns:
            losses: Dictionary of loss components
        """
        losses = {}
        
        # Get curriculum weights
        if self.loss_curriculum:
            self.loss_curriculum.step(epoch)
            loss_weights = self.loss_curriculum.get_all_weights()
        else:
            loss_weights = {
                'chamfer': self.config.loss.chamfer_weight,
                'emd': self.config.loss.emd_weight,
                'feature_matching': self.config.loss.feature_matching_weight,
                'density': self.config.loss.density_weight
            }
        
        # Reconstruction losses
        if 'chamfer' in self.loss_functions:
            losses['chamfer'] = self.loss_functions['chamfer'](pred_points, target_points)
        
        if 'emd' in self.loss_functions and loss_weights.get('emd', 0) > 0:
            losses['emd'] = self.loss_functions['emd'](pred_points, target_points)
        
        if 'feature_matching' in self.loss_functions and loss_weights.get('feature_matching', 0) > 0:
            losses['feature_matching'] = self.loss_functions['feature_matching'](pred_points, target_points)
        
        if 'multiscale' in self.loss_functions:
            losses['multiscale'] = self.loss_functions['multiscale'](pred_points, target_points)
        
        if 'density' in self.loss_functions and loss_weights.get('density', 0) > 0:
            losses['density'] = self.loss_functions['density'](pred_points)
        
        # KL divergence loss
        losses['kl'] = self.model.compute_kl_loss(mu, logvar)
        
        # Total reconstruction loss
        reconstruction_loss = 0.0
        for loss_name, loss_value in losses.items():
            if loss_name != 'kl':
                weight = loss_weights.get(loss_name, 1.0)
                reconstruction_loss += weight * loss_value
        
        losses['reconstruction'] = reconstruction_loss
        
        # Total loss
        losses['total'] = reconstruction_loss + self.config.loss.kl_weight * losses['kl']
        
        return losses
    
    def train_step(self, batch: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Input batch
            epoch: Current epoch
            
        Returns:
            step_losses: Dictionary of loss values
        """
        self.model.train()
        
        # Apply data augmentation
        if self.augmentation:
            batch = self.augmentation(batch)
        
        batch = batch.to(self.device)
        
        # Forward pass
        reconstructed, mu, logvar, aux_outputs = self.model(batch)
        
        # Compute losses
        losses = self.compute_losses(reconstructed, batch, mu, logvar, epoch)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Adversarial training
        if self.config.training.use_adversarial and hasattr(self, 'discriminator'):
            self.adversarial_step(batch, reconstructed)
        
        # Convert to float for logging
        step_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        
        return step_losses
    
    def adversarial_step(self, real_points: torch.Tensor, fake_points: torch.Tensor):
        """Adversarial training step"""
        batch_size = real_points.shape[0]
        
        # Train discriminator
        self.discriminator_optimizer.zero_grad()
        
        # Real points
        real_pred = self.discriminator(real_points.transpose(1, 2))
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_loss = self.loss_functions['adversarial'](real_pred, real_labels)
        
        # Fake points
        fake_pred = self.discriminator(fake_points.detach().transpose(1, 2))
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_loss = self.loss_functions['adversarial'](fake_pred, fake_labels)
        
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Update generator (model) with adversarial loss
        fake_pred = self.discriminator(fake_points.transpose(1, 2))
        generator_loss = self.loss_functions['adversarial'](fake_pred, real_labels)
        
        # Add adversarial loss to total loss (already computed in train_step)
        adversarial_weight = self.config.training.adversarial_weight
        adversarial_loss = adversarial_weight * generator_loss
        adversarial_loss.backward(retain_graph=True)
    
    def validate_step(self, batch: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        Single validation step
        
        Args:
            batch: Input batch
            epoch: Current epoch
            
        Returns:
            step_losses: Dictionary of loss values
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            
            # Forward pass
            reconstructed, mu, logvar, aux_outputs = self.model(batch)
            
            # Compute losses
            losses = self.compute_losses(reconstructed, batch, mu, logvar, epoch)
        
        # Convert to float for logging
        step_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        
        return step_losses
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            epoch_losses: Average losses for the epoch
        """
        # Update curriculum
        if self.point_curriculum:
            self.point_curriculum.step(epoch)
        
        epoch_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.training.num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            step_losses = self.train_step(batch, epoch)
            
            # Accumulate losses
            for key, value in step_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_losses['total']:.4f}",
                'recon': f"{step_losses['reconstruction']:.4f}",
                'kl': f"{step_losses['kl']:.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            epoch_losses: Average losses for the epoch
        """
        epoch_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                step_losses = self.validate_step(batch, epoch)
                
                # Accumulate losses
                for key, value in step_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        if hasattr(self, 'discriminator'):
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_path, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        self.current_epoch = checkpoint['epoch']
        
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint and hasattr(self, 'discriminator'):
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def log_metrics(self, train_losses: Dict[str, float], val_losses: Dict[str, float], epoch: int):
        """Log training metrics"""
        # Update history
        self.training_history['train_loss'].append(train_losses['total'])
        self.training_history['val_loss'].append(val_losses['total'])
        self.training_history['reconstruction_loss'].append(train_losses['reconstruction'])
        self.training_history['kl_loss'].append(train_losses['kl'])
        self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to console
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_losses['total']:.4f}, "
            f"Val Loss: {val_losses['total']:.4f}, "
            f"Recon: {train_losses['reconstruction']:.4f}, "
            f"KL: {train_losses['kl']:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Validation', val_losses['total'], epoch)
            self.writer.add_scalar('Loss/Reconstruction', train_losses['reconstruction'], epoch)
            self.writer.add_scalar('Loss/KL', train_losses['kl'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log individual loss components
            for loss_name, loss_value in train_losses.items():
                if loss_name not in ['total', 'reconstruction', 'kl']:
                    self.writer.add_scalar(f'Loss_Components/{loss_name}', loss_value, epoch)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader:
                val_losses = self.validate_epoch(val_loader, epoch)
            else:
                val_losses = train_losses.copy()
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Log metrics
            self.log_metrics(train_losses, val_losses, epoch)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save final checkpoint
        self.save_checkpoint(self.config.training.num_epochs - 1)
        
        if self.writer:
            self.writer.close()