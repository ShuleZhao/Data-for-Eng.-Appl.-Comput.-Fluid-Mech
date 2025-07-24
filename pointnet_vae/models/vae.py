"""
Variational Autoencoder framework for point clouds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

from .pointnet import PointNetEncoder, PointNetDecoder, PointNetWithTransform
from .pointnet_plus import PointNetPlusPlusEncoder, PointNetPlusPlusDecoder
from .attention import MultiHeadAttention, SelfAttentionBlock, PointTransformer, LocalAttention


class VAEEncoder(nn.Module):
    """
    VAE encoder that outputs mean and log variance for latent distribution
    
    Args:
        backbone: Backbone encoder ('pointnet', 'pointnet++', 'transformer')
        input_dim: Input point dimension
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        use_attention: Whether to use attention mechanism
        attention_heads: Number of attention heads
    """
    
    def __init__(self,
                 backbone: str = 'pointnet++',
                 input_dim: int = 3,
                 latent_dim: int = 512,
                 hidden_dims: List[int] = [64, 128, 256, 512],
                 use_attention: bool = True,
                 attention_heads: int = 8):
        super(VAEEncoder, self).__init__()
        
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Initialize backbone encoder
        if backbone == 'pointnet':
            self.encoder = PointNetWithTransform(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                global_feature_dim=hidden_dims[-1]
            )
            feature_dim = hidden_dims[-1]
        elif backbone == 'pointnet++':
            self.encoder = PointNetPlusPlusEncoder(
                input_dim=input_dim,
                use_normals=(input_dim > 3)
            )
            feature_dim = 1024  # PointNet++ outputs 1024-dim features
        elif backbone == 'transformer':
            self.encoder = PointTransformer(
                input_dim=input_dim,
                d_model=hidden_dims[-1],
                num_heads=attention_heads,
                num_layers=4
            )
            feature_dim = hidden_dims[-1]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Attention mechanism
        if use_attention and backbone != 'transformer':
            self.attention = LocalAttention(
                d_model=feature_dim,
                num_heads=attention_heads,
                k_neighbors=16
            )
        
        # VAE head
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.logvar_head = nn.Linear(feature_dim, latent_dim)
        
        # Feature aggregation for non-PointNet++ backbones
        if backbone != 'pointnet++':
            self.global_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass
        
        Args:
            x: Input point cloud (B, N, input_dim)
            
        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
            aux_outputs: Auxiliary outputs for decoder
        """
        aux_outputs = {}
        
        if self.backbone == 'pointnet':
            local_features, global_feature, transforms = self.encoder(x)
            aux_outputs['transforms'] = transforms
            aux_outputs['local_features'] = local_features
            
            if self.use_attention:
                local_features = self.attention(local_features, x[:, :, :3])
                # Re-compute global feature after attention
                global_feature = torch.max(local_features, dim=1)[0]
            
            feature = global_feature
            
        elif self.backbone == 'pointnet++':
            xyz_list, points_list, global_feature = self.encoder(x)
            aux_outputs['xyz_list'] = xyz_list
            aux_outputs['points_list'] = points_list
            feature = global_feature
            
        elif self.backbone == 'transformer':
            pos = x[:, :, :3]
            if x.shape[-1] > 3:
                features = x
            else:
                features = x
            
            transformed_features = self.encoder(features, pos)
            aux_outputs['local_features'] = transformed_features
            
            # Global pooling
            feature = torch.max(transformed_features, dim=1)[0]
        
        # Compute mu and log variance
        mu = self.mu_head(feature)
        logvar = self.logvar_head(feature)
        
        return mu, logvar, aux_outputs


class VAEDecoder(nn.Module):
    """
    VAE decoder for point cloud reconstruction
    
    Args:
        backbone: Backbone decoder ('pointnet', 'pointnet++', 'mlp')
        latent_dim: Latent space dimension
        output_dim: Output point dimension
        num_points: Number of output points
        hidden_dims: Hidden layer dimensions
        use_skip_connections: Whether to use skip connections
    """
    
    def __init__(self,
                 backbone: str = 'pointnet++',
                 latent_dim: int = 512,
                 output_dim: int = 3,
                 num_points: int = 2048,
                 hidden_dims: List[int] = [512, 256, 128, 64],
                 use_skip_connections: bool = True):
        super(VAEDecoder, self).__init__()
        
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_points = num_points
        self.use_skip_connections = use_skip_connections
        
        # Initialize backbone decoder
        if backbone == 'pointnet':
            self.decoder = PointNetDecoder(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_points=num_points
            )
        elif backbone == 'pointnet++':
            self.decoder = PointNetPlusPlusDecoder(
                latent_dim=latent_dim,
                num_points=num_points,
                output_dim=output_dim
            )
        elif backbone == 'mlp':
            # Simple MLP decoder
            layers = []
            in_dim = latent_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                ])
                in_dim = hidden_dim
            
            layers.append(nn.Linear(hidden_dims[-1], num_points * output_dim))
            self.decoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Skip connection layers
        if use_skip_connections and backbone != 'pointnet++':
            self.skip_connections = nn.ModuleList([
                nn.Linear(latent_dim, hidden_dim) for hidden_dim in hidden_dims
            ])
    
    def forward(self, z: torch.Tensor, aux_inputs: Dict[str, Any] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Latent vector (B, latent_dim)
            aux_inputs: Auxiliary inputs from encoder
            
        Returns:
            reconstructed: Reconstructed point cloud (B, num_points, output_dim)
        """
        if aux_inputs is None:
            aux_inputs = {}
        
        if self.backbone == 'pointnet' or self.backbone == 'mlp':
            reconstructed = self.decoder(z)
            if self.backbone == 'mlp':
                reconstructed = reconstructed.view(-1, self.num_points, self.output_dim)
                
        elif self.backbone == 'pointnet++':
            xyz_list = aux_inputs.get('xyz_list', None)
            points_list = aux_inputs.get('points_list', None)
            
            if xyz_list is None or points_list is None:
                # Fallback to simple generation
                base_coords = torch.randn(z.shape[0], self.num_points, 3, device=z.device)
                xyz_list = [base_coords]
                points_list = [None]
            
            reconstructed = self.decoder(z, xyz_list, points_list)
        
        return reconstructed


class PointNetVAE(nn.Module):
    """
    Complete PointNet VAE model with advanced features
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config):
        super(PointNetVAE, self).__init__()
        
        self.config = config
        self.latent_dim = config.model.latent_dim
        self.kl_weight = config.model.kl_weight
        
        # Encoder
        self.encoder = VAEEncoder(
            backbone='pointnet++',
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            use_attention=config.model.use_attention,
            attention_heads=config.model.attention_heads
        )
        
        # Decoder
        self.decoder = VAEDecoder(
            backbone='pointnet++',
            latent_dim=config.model.latent_dim,
            output_dim=config.model.input_dim,
            num_points=config.model.num_points,
            hidden_dims=list(reversed(config.model.hidden_dims)),
            use_skip_connections=config.model.use_skip_connections
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass
        
        Args:
            x: Input point cloud (B, N, input_dim)
            
        Returns:
            reconstructed: Reconstructed point cloud
            mu: Latent mean
            logvar: Latent log variance
            aux_outputs: Auxiliary outputs
        """
        # Encode
        mu, logvar, aux_outputs = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z, aux_outputs)
        
        return reconstructed, mu, logvar, aux_outputs
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        
        Args:
            x: Input point cloud
            
        Returns:
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar, _ = self.encoder(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor, aux_inputs: Dict[str, Any] = None) -> torch.Tensor:
        """
        Decode latent vector to point cloud
        
        Args:
            z: Latent vector
            aux_inputs: Auxiliary inputs
            
        Returns:
            reconstructed: Reconstructed point cloud
        """
        return self.decoder(z, aux_inputs)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the learned distribution
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated point clouds
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss
        
        Args:
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            kl_loss: KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()
    
    def get_latent_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent codes for input point clouds
        
        Args:
            x: Input point clouds
            
        Returns:
            z: Latent codes
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z