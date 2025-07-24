"""
Configuration system for PointNet VAE
"""

import yaml
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # PointNet++ parameters
    num_points: int = 2048
    input_dim: int = 3
    hidden_dims: list = None
    num_layers: int = 4
    use_attention: bool = True
    attention_heads: int = 8
    
    # VAE parameters
    latent_dim: int = 512
    kl_weight: float = 0.001
    
    # Skip connections
    use_skip_connections: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]


@dataclass
class LossConfig:
    """Loss function configuration"""
    # Loss weights
    chamfer_weight: float = 1.0
    emd_weight: float = 0.5
    feature_matching_weight: float = 0.1
    density_weight: float = 0.01
    kl_weight: float = 0.001
    
    # Multi-scale loss
    use_multiscale_loss: bool = True
    multiscale_weights: list = None
    
    def __post_init__(self):
        if self.multiscale_weights is None:
            self.multiscale_weights = [1.0, 0.5, 0.25]


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 200
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_epochs: list = None
    curriculum_difficulties: list = None
    
    # Progressive training
    use_progressive: bool = True
    progressive_epochs: list = None
    progressive_resolutions: list = None
    
    # Adversarial training
    use_adversarial: bool = False
    discriminator_lr: float = 0.0002
    adversarial_weight: float = 0.1
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    noise_std: float = 0.01
    scaling_range: tuple = (0.9, 1.1)
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential"
    lr_decay_steps: int = 50
    lr_decay_factor: float = 0.7
    
    # Checkpointing
    save_interval: int = 10
    eval_interval: int = 5
    
    def __post_init__(self):
        if self.curriculum_epochs is None:
            self.curriculum_epochs = [50, 100, 150]
        if self.curriculum_difficulties is None:
            self.curriculum_difficulties = [0.3, 0.6, 1.0]
        if self.progressive_epochs is None:
            self.progressive_epochs = [50, 100]
        if self.progressive_resolutions is None:
            self.progressive_resolutions = [512, 1024, 2048]


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    loss: LossConfig = None
    training: TrainingConfig = None
    
    # Device and paths
    device: str = "cuda"
    data_path: str = "./data"
    output_path: str = "./outputs"
    checkpoint_path: str = "./checkpoints"
    
    # Logging
    log_level: str = "INFO"
    use_tensorboard: bool = True
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.loss is None:
            self.loss = LossConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        other_params = {k: v for k, v in config_dict.items() 
                       if k not in ['model', 'loss', 'training']}
        
        return cls(
            model=model_config,
            loss=loss_config,
            training=training_config,
            **other_params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'loss': asdict(self.loss),
            'training': asdict(self.training),
            'device': self.device,
            'data_path': self.data_path,
            'output_path': self.output_path,
            'checkpoint_path': self.checkpoint_path,
            'log_level': self.log_level,
            'use_tensorboard': self.use_tensorboard
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
default_config = Config()