# geospatial-neural-adapter

A Python package for neural spatial modeling with low-rank approximations.

## Prerequisites

### System Requirements
- **Operating System**: Unix-like system (Linux, macOS)
- **Python**: 3.10 or higher
- **Conda**: Miniconda or Anaconda installed
- **C++ Compiler**: GCC (Linux) or Clang (macOS)
- **CMake**: 3.18.0 or higher

### Install Conda (if not already installed)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or for macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

## Installation

### Development installation
```bash
# Clone the repository
git clone https://github.com/egpivo/geospatial-neural-adapter.git
cd geospatial-neural-adapter

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from geospatial_neural_adapter import SpatialBasisLearner, TrendModel, SpatialNeuralAdapter

# Create spatial basis learner
spatial_learner = SpatialBasisLearner(
    input_dim=10,
    hidden_dim=50,
    output_dim=25
)

# Create trend model
trend_model = TrendModel(
    input_dim=10,
    hidden_layers=[100, 50, 25]
)

# Initialize trainer
trainer = SpatialNeuralAdapter(
    spatial_learner=spatial_learner,
    trend_model=trend_model,
    learning_rate=0.001,
    epochs=100
)

# Train model
history = trainer.train(
    train_features=X_train,
    train_labels=y_train,
    valid_features=X_val,
    valid_labels=y_val
)
```

## Development

### Setup development environment
```bash
# Set up Conda environment
make conda-env
```

### Run tests
```bash
make test
```

### Run tests with coverage
```bash
make test-cov
```

### Format code
```bash
poetry run black .
poetry run isort .
```

### Type checking
```bash
poetry run mypy geospatial_neural_adapter
```

### Build C++ extensions
```bash
make build-cpp
```

## License

MIT License - see LICENSE file for details.
