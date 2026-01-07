from setuptools import setup, find_packages

setup(
    name="neural_mmot",
    version="1.0.0",
    description="Neural Network-Based Multi-Period Martingale Optimal Transport Solver",
    author="ML Engineering Team",
    author_email="team@example.com",
    url="https://github.com/your-org/neural-mmot",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "jax>=0.4.13",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "tensorboard>=2.13.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
