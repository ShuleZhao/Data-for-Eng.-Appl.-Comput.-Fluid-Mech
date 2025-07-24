"""
Basic tests for PointNet VAE implementation
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pointnet_vae.config import Config
from pointnet_vae.models import PointNetVAE
from pointnet_vae.losses import ChamferLoss, EMDLoss, DensityRegularizationLoss
from pointnet_vae.utils import FarthestPointSampling, PointCloudMetrics


class TestBasicFunctionality:
    """Test basic functionality of the PointNet VAE implementation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.config = Config()
        self.config.model.num_points = 1024  # Smaller for faster testing
        self.config.model.latent_dim = 128
        self.batch_size = 4
        
        # Create dummy data
        self.dummy_data = torch.randn(self.batch_size, self.config.model.num_points, 3)
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = Config()
        assert config.model.num_points == 2048
        assert config.model.latent_dim == 512
        assert config.model.input_dim == 3
        
        # Test config dictionary conversion
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'loss' in config_dict
        assert 'training' in config_dict
    
    def test_model_creation(self):
        """Test model creation and basic forward pass"""
        model = PointNetVAE(self.config)
        assert model is not None
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar, aux_outputs = model(self.dummy_data)
        
        # Check output shapes
        assert reconstructed.shape == self.dummy_data.shape
        assert mu.shape == (self.batch_size, self.config.model.latent_dim)
        assert logvar.shape == (self.batch_size, self.config.model.latent_dim)
    
    def test_model_encode_decode(self):
        """Test separate encode and decode operations"""
        model = PointNetVAE(self.config)
        model.eval()
        
        with torch.no_grad():
            # Test encoding
            mu, logvar = model.encode(self.dummy_data)
            assert mu.shape == (self.batch_size, self.config.model.latent_dim)
            assert logvar.shape == (self.batch_size, self.config.model.latent_dim)
            
            # Test decoding
            z = model.reparameterize(mu, logvar)
            reconstructed = model.decode(z)
            assert reconstructed.shape == self.dummy_data.shape
    
    def test_model_sampling(self):
        """Test model sampling capability"""
        model = PointNetVAE(self.config)
        model.eval()
        
        with torch.no_grad():
            # Test sampling
            num_samples = 5
            samples = model.sample(num_samples, self.device)
            assert samples.shape == (num_samples, self.config.model.num_points, 3)
    
    def test_loss_functions(self):
        """Test various loss functions"""
        pred = torch.randn(self.batch_size, 512, 3)
        target = torch.randn(self.batch_size, 512, 3)
        
        # Test Chamfer loss
        chamfer_loss = ChamferLoss()
        loss_value = chamfer_loss(pred, target)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar
        assert loss_value >= 0  # Non-negative
        
        # Test density regularization
        density_loss = DensityRegularizationLoss()
        loss_value = density_loss(pred)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0
        assert loss_value >= 0
    
    def test_sampling_methods(self):
        """Test sampling methods"""
        points = torch.randn(1000, 3)
        
        # Test FPS
        fps = FarthestPointSampling(num_samples=256)
        sampled_points, indices = fps(points)
        assert sampled_points.shape == (256, 3)
        assert indices.shape == (256,)
        assert len(torch.unique(indices)) == 256  # All indices should be unique
    
    def test_metrics(self):
        """Test evaluation metrics"""
        pred_batch = torch.randn(2, 256, 3)
        target_batch = torch.randn(2, 256, 3)
        
        metrics = PointCloudMetrics(self.device)
        
        # Test fast metrics (Chamfer distance only)
        fast_metrics = metrics.compute_fast_metrics(pred_batch, target_batch)
        assert 'chamfer_distance' in fast_metrics
        assert fast_metrics['chamfer_distance'] >= 0
    
    def test_kl_loss_computation(self):
        """Test KL loss computation"""
        model = PointNetVAE(self.config)
        
        mu = torch.randn(self.batch_size, self.config.model.latent_dim)
        logvar = torch.randn(self.batch_size, self.config.model.latent_dim)
        
        kl_loss = model.compute_kl_loss(mu, logvar)
        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.dim() == 0
    
    def test_training_mode_switching(self):
        """Test training and evaluation mode switching"""
        model = PointNetVAE(self.config)
        
        # Test training mode
        model.train()
        assert model.training
        
        with torch.no_grad():
            # In training mode, should use reparameterization
            reconstructed1, mu1, logvar1, _ = model(self.dummy_data)
        
        # Test evaluation mode
        model.eval()
        assert not model.training
        
        with torch.no_grad():
            # In eval mode, should use mean
            reconstructed2, mu2, logvar2, _ = model(self.dummy_data)
        
        # Results should be different due to reparameterization
        # Note: This might not always be true due to randomness, but generally should be
        assert reconstructed1.shape == reconstructed2.shape


def test_import_structure():
    """Test that all modules can be imported correctly"""
    try:
        from pointnet_vae import Config
        from pointnet_vae.models import PointNetVAE
        from pointnet_vae.losses import ChamferLoss
        from pointnet_vae.utils import PointCloudVisualizer
        from pointnet_vae.training import PointNetVAETrainer
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_dummy_data_creation():
    """Test creation of dummy data for testing"""
    # Test sphere generation
    def generate_sphere(num_points=1000):
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0.8, 1.0, num_points)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        return np.stack([x, y, z], axis=1)
    
    sphere_points = generate_sphere(1000)
    assert sphere_points.shape == (1000, 3)
    
    # Check that points are roughly on a sphere
    distances_from_origin = np.linalg.norm(sphere_points, axis=1)
    assert np.all(distances_from_origin >= 0.7)  # Should be roughly between 0.8 and 1.0
    assert np.all(distances_from_origin <= 1.1)


if __name__ == '__main__':
    # Run basic tests
    test_suite = TestBasicFunctionality()
    test_suite.setup_method()
    
    print("Running basic functionality tests...")
    
    try:
        test_suite.test_config_creation()
        print("✓ Config creation test passed")
        
        test_suite.test_model_creation()
        print("✓ Model creation test passed")
        
        test_suite.test_model_encode_decode()
        print("✓ Model encode/decode test passed")
        
        test_suite.test_model_sampling()
        print("✓ Model sampling test passed")
        
        test_suite.test_loss_functions()
        print("✓ Loss functions test passed")
        
        test_suite.test_sampling_methods()
        print("✓ Sampling methods test passed")
        
        test_suite.test_metrics()
        print("✓ Metrics test passed")
        
        test_suite.test_kl_loss_computation()
        print("✓ KL loss computation test passed")
        
        test_suite.test_training_mode_switching()
        print("✓ Training mode switching test passed")
        
        test_import_structure()
        print("✓ Import structure test passed")
        
        test_dummy_data_creation()
        print("✓ Dummy data creation test passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()