from setuptools import setup, find_packages

setup(
    name="pointnet-vae",
    version="0.1.0",
    description="Advanced PointNet VAE for Point Cloud Reconstruction",
    author="ShuleZhao",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "open3d>=0.13.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "einops>=0.4.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)