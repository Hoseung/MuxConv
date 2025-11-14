# MuxConv: Multiplexed Convolutional Neural Networks for FHE

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

MuxConv is a Python library implementing efficient Fully Homomorphic Encryption (FHE) based Convolutional Neural Networks using multiplexed convolution techniques. This approach optimizes homomorphic operations for CNN inference on encrypted data, enabling privacy-preserving machine learning applications.

## Key Features

- **Multiplexed Convolution**: Efficient packing and convolution operations on encrypted data
- **ResNet Support**: Implementation of ResNet architectures for FHE
- **Parallel Batch Normalization**: Optimized batch normalization for encrypted tensors
- **HEAAN Integration**: Support for HEAAN homomorphic encryption scheme
- **Approximate Activation Functions**: Sign and ReLU approximations suitable for FHE

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, PyTorch
- HEAAN library (for FHE operations)

### Install from source

```bash
git clone https://github.com/yourusername/MuxConv.git
cd MuxConv
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
MuxConv/
├── muxcnn/                  # Main package
│   ├── hecnn.py            # Core multiplexed convolution operations
│   ├── hecnn_par.py        # Parallel/batched convolution operations
│   ├── utils.py            # Utility functions for tensor operations
│   ├── comparator_heaan.py # Approximation functions for HEAAN
│   ├── resnet_muxconv.py   # ResNet implementation with MuxConv
│   ├── resnet_HEAAN.py     # ResNet with HEAAN backend
│   ├── resnet_fhe.py       # Generic FHE ResNet
│   └── models/             # Neural network model definitions
├── scripts/                 # Example scripts and notebooks
└── tests/                   # Unit tests (coming soon)
```

## Quick Start

### Basic Usage

```python
import torch
from muxcnn.resnet_muxconv import ResNet_MuxConv
from muxcnn.models import ResNet20

# Load a pre-trained PyTorch model
model = ResNet20()
model.load_state_dict(torch.load('model.pt'))

# Create MuxConv wrapper
mux_model = ResNet_MuxConv(model, alpha=12)

# Perform inference on encrypted data
img_tensor = torch.randn(1, 3, 32, 32)
result = mux_model(img_tensor)
```

### Multiplexed Convolution Example

```python
import numpy as np
from muxcnn.hecnn import MultConv, MultPack
from muxcnn.utils import get_conv_params

# Prepare input dimensions
ins = {'h': 32, 'w': 32, 'c': 3, 'k': 1, 't': 3, 'p': 4}
outs = {'h': 32, 'w': 32, 'c': 16, 'k': 1, 't': 16, 'p': 1}

# Pack input for multiplexed convolution
input_mat = np.random.randn(32, 32, 3)
ct_a = MultPack(input_mat, ins)

# Perform convolution
weights = np.random.randn(3, 3, 3, 16)
ct_out = MultConv(ct_a, weights, ins, outs)
```

## Core Concepts

### Multiplexed Convolution

The library implements multiplexed convolution, a technique that packs multiple input channels into a single ciphertext, reducing the number of homomorphic operations required for CNN inference. Key parameters:

- **k**: Multiplexing factor
- **t**: Number of packed channel groups
- **p**: Parallelization factor
- **nslots**: Number of slots in the ciphertext (default: 2^15)

### Tensor Packing

Input tensors are packed using `MultPack` or `MultParPack` functions, which arrange data in a specific layout optimized for homomorphic operations:

```python
# Standard packing
ct = MultPack(input_tensor, dims, nslots=2**15)

# Parallel packing (for batched operations)
ct = MultParPack(input_tensor, dims, nslots=2**15)
```

### Rotation-Aware Operations

The library tracks and minimizes ciphertext rotations, which are expensive operations in FHE. The `SumSlots` function efficiently aggregates values using a logarithmic number of rotations.

## Technical Details

### Supported Operations

- **Convolution**: Standard and strided convolutions with multiplexing
- **Batch Normalization**: Parallel batch normalization for encrypted data
- **Activation Functions**: Approximate ReLU and sign functions
- **Pooling**: Average pooling with ciphertext packing
- **Linear Layers**: Fully connected layers for encrypted data

### Performance Optimization

- Minimized ciphertext rotations
- Efficient slot packing strategies
- Parallel channel processing
- Optimized weight packing

## Examples

Example notebooks are available in the `scripts/` directory:

- `MuxConv_FHE.ipynb`: Basic multiplexed convolution examples
- `MuxConv_HEAAN_example.ipynb`: Integration with HEAAN
- `MuxConv_parBN_final_example.ipynb`: Parallel batch normalization
- `Validate.ipynb`: Validation and testing examples

## Citation

If you use this code in your research, please cite:

```bibtex
@software{muxconv2024,
  title={MuxConv: Multiplexed Convolutional Neural Networks for FHE},
  author={Hoseung Choi},
  year={2024},
  url={https://github.com/yourusername/MuxConv}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HEAAN library for homomorphic encryption primitives
- Research on efficient FHE-based CNNs
- PyTorch for neural network framework

## Contact

For questions and feedback:
- Email: hschoi@dinsight.ai
- Issues: [GitHub Issues](https://github.com/yourusername/MuxConv/issues)

## References

- [FHE-based CNN papers and research]
- [HEAAN documentation]
- [Related work on privacy-preserving ML]