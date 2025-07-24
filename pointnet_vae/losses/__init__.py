"""
Loss functions package for PointNet VAE
"""

from .chamfer import (
    ChamferLoss, MultiScaleChamferLoss, SymmetricChamferLoss, 
    RobustChamferLoss, chamfer_distance_squared
)
from .emd import (
    EMDLoss, ApproximateEMDLoss, MultiScaleEMDLoss, 
    CombinedEMDChamferLoss, earth_movers_distance
)
from .feature_matching import (
    FeatureMatchingLoss, PerceptualLoss, LocalFeatureMatchingLoss,
    HierarchicalFeatureMatchingLoss, AdversarialFeatureLoss
)
from .multi_scale import (
    MultiScaleLoss, PyramidLoss, AdaptiveMultiScaleLoss,
    ProgressiveMultiScaleLoss
)
from .density import (
    DensityRegularizationLoss, VoronoiDensityLoss, RepulsionLoss,
    CoverageAndDensityLoss, AdaptiveDensityLoss
)

__all__ = [
    # Chamfer Distance losses
    'ChamferLoss', 'MultiScaleChamferLoss', 'SymmetricChamferLoss', 
    'RobustChamferLoss', 'chamfer_distance_squared',
    
    # Earth Mover's Distance losses
    'EMDLoss', 'ApproximateEMDLoss', 'MultiScaleEMDLoss', 
    'CombinedEMDChamferLoss', 'earth_movers_distance',
    
    # Feature matching losses
    'FeatureMatchingLoss', 'PerceptualLoss', 'LocalFeatureMatchingLoss',
    'HierarchicalFeatureMatchingLoss', 'AdversarialFeatureLoss',
    
    # Multi-scale losses
    'MultiScaleLoss', 'PyramidLoss', 'AdaptiveMultiScaleLoss',
    'ProgressiveMultiScaleLoss',
    
    # Density losses
    'DensityRegularizationLoss', 'VoronoiDensityLoss', 'RepulsionLoss',
    'CoverageAndDensityLoss', 'AdaptiveDensityLoss'
]