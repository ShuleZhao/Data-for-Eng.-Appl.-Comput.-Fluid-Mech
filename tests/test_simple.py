#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test basic imports"""
    try:
        import torch
        import numpy as np
        print("✓ Basic dependencies imported successfully")
        
        from pointnet_vae.config import Config
        print("✓ Config imported successfully")
        
        from pointnet_vae.models import PointNetVAE
        print("✓ PointNetVAE imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration creation"""
    try:
        from pointnet_vae.config import Config
        
        config = Config()
        assert config.model.num_points == 2048
        assert config.model.latent_dim == 512
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    try:
        import torch
        from pointnet_vae.config import Config
        from pointnet_vae.models import PointNetVAE
        
        config = Config()
        model = PointNetVAE(config)
        
        # Test model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        print(f"✓ Model created successfully with {total_params:,} parameters")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test model forward pass"""
    try:
        import torch
        from pointnet_vae.config import Config
        from pointnet_vae.models import PointNetVAE
        
        config = Config()
        config.model.num_points = 512  # Smaller for testing
        config.model.latent_dim = 128
        
        model = PointNetVAE(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, config.model.num_points, 3)
        
        with torch.no_grad():
            reconstructed, mu, logvar, aux_outputs = model(dummy_input)
        
        # Check output shapes
        assert reconstructed.shape == dummy_input.shape
        assert mu.shape == (batch_size, config.model.latent_dim)
        assert logvar.shape == (batch_size, config.model.latent_dim)
        
        print("✓ Forward pass test passed")
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss functions"""
    try:
        import torch
        from pointnet_vae.losses import ChamferLoss
        
        # Create dummy data
        pred = torch.randn(2, 256, 3)
        target = torch.randn(2, 256, 3)
        
        # Test Chamfer loss
        chamfer_loss = ChamferLoss()
        loss_value = chamfer_loss(pred, target)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar
        assert loss_value >= 0  # Non-negative
        
        print("✓ Loss functions test passed")
        return True
    except Exception as e:
        print(f"❌ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting basic functionality tests...\n")
    
    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_forward_pass,
        test_loss_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)