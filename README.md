# Advanced PointNet VAE for Point Cloud Reconstruction

A comprehensive implementation of PointNet VAE with advanced features for high-quality point cloud reconstruction, featuring curriculum learning, multi-scale loss functions, attention mechanisms, and progressive training strategies.

## 🌟 Key Features

### Core Architecture
- **PointNet & PointNet++**: Both classic PointNet and hierarchical PointNet++ architectures
- **Advanced VAE Framework**: Flexible encoder-decoder with multiple backbone options
- **Attention Mechanisms**: Multi-head attention and transformer architectures for better feature learning
- **Skip Connections**: Preserve fine details during reconstruction

### Advanced Loss Functions
- **Chamfer Distance**: Multiple variants (multi-scale, robust, symmetric)
- **Earth Mover's Distance (EMD)**: Optimal transport-based loss with Sinkhorn approximation
- **Feature Matching Loss**: High-dimensional feature space matching
- **Multi-scale Supervision**: Loss computation at different resolution levels
- **Density Regularization**: Ensure reasonable point cloud density distribution

### Training Enhancements
- **Curriculum Learning**: Progressive difficulty increase during training
- **Data Augmentation**: Comprehensive augmentation pipeline (rotation, scaling, noise, elastic deformation)
- **Progressive Training**: From coarse to fine resolution
- **Adversarial Training**: Optional discriminator for improved quality
- **Smart Learning Rate Scheduling**: Cosine annealing, step decay, exponential decay

### Advanced Sampling & Post-processing
- **Farthest Point Sampling (FPS)**: Better point distribution
- **Density-based Sampling**: Adaptive sampling based on local density
- **Voxel Grid Sampling**: Uniform spatial distribution
- **Poisson Disk Sampling**: Minimum distance constraints
- **Adaptive Sampling**: Automatically selects best strategy

### Utilities & Tools
- **Comprehensive Metrics**: Chamfer distance, Hausdorff distance, EMD, F-score, coverage
- **Rich Visualization**: Matplotlib and Open3D integration for 3D plotting
- **Training Monitoring**: TensorBoard integration with detailed logging
- **Flexible Configuration**: YAML-based configuration system

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ShuleZhao/Data-for-Eng.-Appl.-Comput.-Fluid-Mech.git
cd Data-for-Eng.-Appl.-Comput.-Fluid-Mech

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Training

```bash
# Train with default configuration
python scripts/train.py --output_dir ./outputs --checkpoint_dir ./checkpoints

# Train with custom configuration
python scripts/train.py --config config.yaml --output_dir ./outputs

# Resume training from checkpoint
python scripts/train.py --resume ./checkpoints/best_model.pth
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint ./checkpoints/best_model.pth --output_dir ./evaluation_results

# Quick evaluation (fewer metrics)
python scripts/evaluate.py --checkpoint ./checkpoints/best_model.pth --num_batches 10
```

### Visualization

```bash
# Create comprehensive visualizations
python scripts/visualize.py --checkpoint ./checkpoints/best_model.pth --output_dir ./visualizations

# Generate specific number of samples
python scripts/visualize.py --checkpoint ./checkpoints/best_model.pth --num_samples 1000
```

## 📊 Model Architecture

### PointNet VAE Overview

```python
from pointnet_vae import PointNetVAE, Config

# Create configuration
config = Config()
config.model.latent_dim = 512
config.model.num_points = 2048
config.model.use_attention = True

# Initialize model
model = PointNetVAE(config)

# Forward pass
reconstructed, mu, logvar, aux_outputs = model(input_points)
```

### Advanced Loss Configuration

```python
from pointnet_vae.losses import *

# Multi-scale Chamfer loss
chamfer_loss = MultiScaleChamferLoss(
    scales=[1.0, 0.5, 0.25],
    weights=[1.0, 0.5, 0.25]
)

# Combined EMD + Chamfer loss
combined_loss = CombinedEMDChamferLoss(
    emd_weight=0.5,
    chamfer_weight=1.0
)

# Feature matching loss
feature_loss = LocalFeatureMatchingLoss(
    k_neighbors=16,
    feature_dim=256
)
```

## 🎯 Training Strategies

### Curriculum Learning

The model supports curriculum learning with gradually increasing difficulty:

```python
from pointnet_vae.training import PointCloudCurriculumScheduler

scheduler = PointCloudCurriculumScheduler(
    total_epochs=200,
    initial_num_points=512,
    final_num_points=2048
)
```

### Data Augmentation

Comprehensive augmentation pipeline:

```python
from pointnet_vae.training import create_default_augmentation_pipeline

augmentation = create_default_augmentation_pipeline({
    'rotation': {'enabled': True, 'max_angle': 15.0},
    'scaling': {'enabled': True, 'scale_range': (0.8, 1.2)},
    'noise': {'enabled': True, 'noise_std': 0.01},
    'jitter': {'enabled': True, 'jitter_std': 0.01}
})
```

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Chamfer Distance**: Bidirectional point-to-point distance
- **Hausdorff Distance**: Maximum distance between point sets
- **Earth Mover's Distance**: Optimal transport distance
- **F-Score**: Precision and recall at multiple thresholds
- **Coverage & Completeness**: Geometric accuracy measures
- **Density Consistency**: Local density preservation
- **Structural Similarity**: Local neighborhood analysis

## 🛠️ Configuration

### YAML Configuration Example

```yaml
model:
  num_points: 2048
  input_dim: 3
  latent_dim: 512
  hidden_dims: [64, 128, 256, 512]
  use_attention: true
  attention_heads: 8

loss:
  chamfer_weight: 1.0
  emd_weight: 0.5
  feature_matching_weight: 0.1
  density_weight: 0.01
  use_multiscale_loss: true

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 200
  use_curriculum: true
  use_augmentation: true
  lr_scheduler: "cosine"
```

### Programmatic Configuration

```python
from pointnet_vae.config import Config, ModelConfig, LossConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        num_points=2048,
        latent_dim=512,
        use_attention=True
    ),
    loss=LossConfig(
        chamfer_weight=1.0,
        emd_weight=0.5,
        use_multiscale_loss=True
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        use_curriculum=True
    )
)
```

## 📊 Visualization Examples

The framework provides rich visualization capabilities:

### Point Cloud Visualization
- **3D Scatter Plots**: Interactive and static visualization
- **Comparison Plots**: Side-by-side original vs reconstructed
- **Batch Galleries**: Multiple samples in grid layout
- **Animation Support**: Interpolation sequences as GIFs

### Training Monitoring
- **Loss Curves**: Training and validation losses over time
- **Learning Rate Schedules**: LR changes during training
- **Latent Space Visualization**: PCA and t-SNE plots
- **Attention Maps**: Visualization of attention weights

## 🧪 Advanced Features

### Progressive Training
Start with low resolution and gradually increase:

```python
trainer.progressive_stage = 0  # Start with 512 points
# Automatically increases to 1024, then 2048 points
```

### Adversarial Training
Optional discriminator for improved quality:

```python
config.training.use_adversarial = True
config.training.adversarial_weight = 0.1
```

### Adaptive Sampling
Automatically selects best sampling strategy:

```python
from pointnet_vae.utils import AdaptiveSampling

sampler = AdaptiveSampling(
    num_samples=2048,
    strategies=[('fps', 0.4), ('density', 0.4), ('voxel', 0.2)]
)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- Open3D 0.13+
- Matplotlib 3.4+
- scikit-learn 1.0+
- TensorBoard 2.7+
- SciPy 1.7+
- PyYAML 5.4+

Optional dependencies:
- Python Optimal Transport (POT) for exact EMD computation
- UMAP for advanced dimensionality reduction

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{pointnet_vae_advanced,
  title={Advanced PointNet VAE for Point Cloud Reconstruction},
  author={ShuleZhao},
  year={2024},
  url={https://github.com/ShuleZhao/Data-for-Eng.-Appl.-Comput.-Fluid-Mech}
}
```

## 🙏 Acknowledgments

This implementation builds upon:
- PointNet and PointNet++ architectures
- Variational Autoencoder frameworks
- Advanced point cloud processing techniques
- Modern deep learning best practices

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example configurations

---

**Built with ❤️ for the point cloud processing community**