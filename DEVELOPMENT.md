# Development Guide

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Make (optional, for convenience commands)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/echoloc-nn.git
   cd echoloc-nn
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   make install-dev
   # Or manually:
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   python -c "import echoloc_nn; print(echoloc_nn.__version__)"
   ```

## Development Workflow

### Code Quality

This project maintains high code quality standards:

```bash
# Format code
make format

# Run all quality checks
make all-checks

# Individual checks
make lint          # Linting with flake8
make type-check    # Type checking with mypy
make security-check # Security scanning
make test          # Run test suite
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (detect-secrets)
- Trailing whitespace removal
- YAML/TOML validation

To run manually:
```bash
pre-commit run --all-files
```

## Project Structure

```
echoloc-nn/
├── echoloc_nn/              # Main package
│   ├── __init__.py
│   ├── models/              # Neural network models
│   ├── hardware/            # Hardware interfaces
│   ├── signal_processing/   # Signal processing utilities
│   ├── simulation/          # Acoustic simulation
│   ├── training/            # Training utilities
│   └── visualization/       # Visualization tools
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Usage examples
├── hardware_designs/        # PCB designs and schematics
├── firmware/                # Microcontroller firmware
└── benchmarks/             # Performance benchmarks
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/ -m "not slow"      # Skip slow tests
python -m pytest tests/ -m integration     # Integration tests only
python -m pytest tests/ -m hardware        # Hardware tests only

# Run with coverage
python -m pytest --cov=echoloc_nn --cov-report=html
```

### Test Categories

Tests are marked with categories:
- `slow`: Long-running tests (model training, large datasets)
- `integration`: Tests requiring multiple components
- `hardware`: Tests requiring physical hardware
- `unit`: Fast unit tests (default)

### Writing Tests

```python
import pytest
from echoloc_nn.models import EchoLocModel

@pytest.mark.unit
def test_model_initialization():
    model = EchoLocModel(n_sensors=4)
    assert model.n_sensors == 4

@pytest.mark.hardware
def test_ultrasonic_array_connection():
    # Tests requiring actual hardware
    pass

@pytest.mark.slow
def test_full_training_pipeline():
    # Long-running training tests
    pass
```

## Architecture Guidelines

### Code Organization

1. **Models** (`echoloc_nn/models/`)
   - Neural network architectures
   - CNN, Transformer, and hybrid implementations
   - Pre-trained model loading and saving

2. **Hardware** (`echoloc_nn/hardware/`)
   - Ultrasonic array interfaces
   - Arduino/Raspberry Pi communication
   - Sensor calibration and configuration

3. **Signal Processing** (`echoloc_nn/signal_processing/`)
   - Echo preprocessing and filtering
   - Feature extraction algorithms
   - Beamforming and array processing

4. **Training** (`echoloc_nn/training/`)
   - Training loops and optimization
   - Data augmentation strategies
   - Self-supervised learning methods

### Design Principles

1. **Modularity**: Each component should be independently testable
2. **Configurability**: Use YAML configs for hyperparameters
3. **Hardware Abstraction**: Abstract hardware interfaces for testing
4. **Performance**: Optimize for real-time inference
5. **Extensibility**: Plugin architecture for new algorithms

### Type Hints

All code must include type hints:

```python
from typing import Tuple, Optional, List
import torch
import numpy as np

def process_echo(
    echo: np.ndarray,
    sample_rate: int,
    sensor_positions: List[Tuple[float, float]]
) -> Tuple[torch.Tensor, Optional[float]]:
    """Process ultrasonic echo for localization.
    
    Args:
        echo: Raw echo data, shape (n_sensors, n_samples)
        sample_rate: Sampling rate in Hz
        sensor_positions: List of (x, y) positions in meters
        
    Returns:
        Tuple of processed tensor and confidence score
    """
    pass
```

## Hardware Development

### Ultrasonic Array Setup

1. **Required Components**
   - 4x HC-SR04 or similar ultrasonic sensors
   - Arduino Uno or compatible microcontroller
   - Breadboard and jumper wires
   - 5V power supply

2. **Wiring Diagram**
   ```
   Sensor 1: Trig->Pin 2, Echo->Pin 3
   Sensor 2: Trig->Pin 4, Echo->Pin 5
   Sensor 3: Trig->Pin 6, Echo->Pin 7
   Sensor 4: Trig->Pin 8, Echo->Pin 9
   ```

3. **Firmware Upload**
   ```bash
   # Generate Arduino code
   python -m echoloc_nn.hardware.arduino_generator
   
   # Upload to Arduino
   arduino-cli compile --fqbn arduino:avr:uno firmware/echoloc_array
   arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno firmware/echoloc_array
   ```

### Testing Hardware

```bash
# Test sensor connectivity
make test-hardware

# Interactive sensor testing
python -m echoloc_nn.hardware.test_sensors --port /dev/ttyUSB0
```

## Model Development

### Adding New Models

1. Create model class in `echoloc_nn/models/`
2. Inherit from `EchoLocBaseModel`
3. Implement required methods:
   - `forward()`
   - `configure_optimizers()`
   - `training_step()`
   - `validation_step()`

```python
from echoloc_nn.models.base import EchoLocBaseModel
import torch.nn as nn

class MyCustomModel(EchoLocBaseModel):
    def __init__(self, n_sensors: int, **kwargs):
        super().__init__()
        self.n_sensors = n_sensors
        # Initialize layers
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement forward pass
        pass
```

### Training New Models

```python
from echoloc_nn.training import EchoLocTrainer
from echoloc_nn.data import EchoDataModule

# Setup data
data_module = EchoDataModule(
    data_dir="data/",
    batch_size=32,
    num_workers=4
)

# Initialize trainer
trainer = EchoLocTrainer(
    model=MyCustomModel,
    max_epochs=100,
    gpus=1
)

# Train model
trainer.fit(data_module)
```

## Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Clean and rebuild
make docs-clean

# Serve locally
cd docs/_build/html && python -m http.server 8000
```

### Writing Documentation

- Use Google-style docstrings
- Include type hints in function signatures
- Provide examples in docstrings
- Update README.md for major changes

```python
def calculate_position(
    echo_data: np.ndarray,
    sensor_positions: List[Tuple[float, float]],
    speed_of_sound: float = 343.0
) -> Tuple[float, float]:
    """Calculate position from ultrasonic echo data.
    
    This function processes raw echo data from multiple ultrasonic
    sensors to triangulate the position of a reflecting object.
    
    Args:
        echo_data: Raw echo amplitudes, shape (n_sensors, n_samples)
        sensor_positions: Sensor (x, y) coordinates in meters
        speed_of_sound: Sound speed in m/s at current temperature
        
    Returns:
        Estimated (x, y) position in meters
        
    Raises:
        ValueError: If sensor_positions length doesn't match echo_data
        
    Example:
        >>> positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        >>> echo = np.random.randn(4, 1024)
        >>> x, y = calculate_position(echo, positions)
        >>> print(f"Position: ({x:.2f}, {y:.2f})")
    """
    pass
```

## Performance Optimization

### Profiling

```bash
# Profile training performance
python -m cProfile -o profile.stats train_model.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
python -m memory_profiler train_model.py
```

### Optimization Targets

1. **Training Speed**: Batch processing, mixed precision
2. **Inference Latency**: Model quantization, pruning
3. **Memory Usage**: Gradient checkpointing, data loading
4. **Hardware Utilization**: Multi-GPU training, CUDA optimization

## Contributing

### Submission Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes following code quality standards
4. Add tests for new functionality
5. Update documentation as needed
6. Submit pull request with clear description

### Code Review Checklist

- [ ] Code follows style guidelines (black, isort)
- [ ] All tests pass
- [ ] Type hints included
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance impact assessed

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -e ".[dev]"  # Reinstall in development mode
   ```

2. **Hardware Not Detected**
   ```bash
   ls /dev/tty*              # List available ports
   sudo usermod -a -G dialout $USER  # Add user to dialout group
   ```

3. **CUDA Errors**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in configs
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

### Getting Help

- Check existing issues on GitHub
- Join the discussion forum
- Read the troubleshooting docs
- Contact maintainers for complex issues

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `__version__` in `__init__.py`
3. Update CHANGELOG.md
4. Create git tag: `git tag v0.1.0`
5. Push tags: `git push --tags`

### Building Release

```bash
make build
python -m twine upload dist/*
```