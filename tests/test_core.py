#!/usr/bin/env python3
"""
Core functionality test - minimal imports
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_core_functionality():
    """Test core functionality without optional dependencies"""
    try:
        import torch
        import numpy as np
        
        # Test Config directly
        from pointnet_vae.config import Config, ModelConfig, LossConfig, TrainingConfig
        
        config = Config()
        assert config.model.num_points == 2048
        assert config.model.latent_dim == 512
        print("✓ Configuration system working")
        
        # Test core model components directly
        from pointnet_vae.models.vae import PointNetVAE
        
        # Create a minimal config for testing
        config.model.num_points = 512  # Smaller for testing
        config.model.latent_dim = 128
        
        model = PointNetVAE(config)
        model.eval()
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, config.model.num_points, 3)
        
        with torch.no_grad():
            reconstructed, mu, logvar, aux_outputs = model(dummy_input)
        
        # Check shapes
        assert reconstructed.shape == dummy_input.shape
        assert mu.shape == (batch_size, config.model.latent_dim)
        assert logvar.shape == (batch_size, config.model.latent_dim)
        print("✓ Model forward pass working")
        
        # Test encode/decode separately
        with torch.no_grad():
            mu, logvar = model.encode(dummy_input)
            z = model.reparameterize(mu, logvar)
            reconstructed2 = model.decode(z, aux_outputs)
        
        assert reconstructed2.shape == dummy_input.shape
        print("✓ Encode/decode working")
        
        # Test sampling
        with torch.no_grad():
            samples = model.sample(3, torch.device('cpu'))
        
        assert samples.shape == (3, config.model.num_points, 3)
        print("✓ Sampling working")
        
        # Test loss functions
        from pointnet_vae.losses.chamfer import ChamferLoss
        
        chamfer_loss = ChamferLoss()
        pred = torch.randn(2, 256, 3)
        target = torch.randn(2, 256, 3)
        
        loss_value = chamfer_loss(pred, target)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value >= 0
        print("✓ Chamfer loss working")
        
        # Test KL loss
        kl_loss = model.compute_kl_loss(mu, logvar)
        assert isinstance(kl_loss, torch.Tensor)
        print("✓ KL loss working")
        
        # Test basic sampling methods
        from pointnet_vae.utils.sampling import FarthestPointSampling
        
        fps = FarthestPointSampling(num_samples=128)
        points = torch.randn(512, 3)
        sampled_points, indices = fps(points)
        
        assert sampled_points.shape == (128, 3)
        assert indices.shape == (128,)
        print("✓ FPS sampling working")
        
        print("\n🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Testing core functionality (minimal dependencies)...\n")
    success = test_core_functionality()
    sys.exit(0 if success else 1)