"""
MuxConv - Multiplexed Convolutional Neural Networks for FHE

This package implements efficient homomorphic encryption-based
convolutional neural networks using multiplexed convolution techniques.
"""

__version__ = "0.0.1"

# Explicit imports from submodules
from . import hecnn
from . import hecnn_par
from . import utils
from . import comparator_heaan
from . import resnet_muxconv
from . import resnet_HEAAN
from . import resnet_fhe

__all__ = [
    'hecnn',
    'hecnn_par',
    'utils',
    'comparator_heaan',
    'resnet_muxconv',
    'resnet_HEAAN',
    'resnet_fhe',
]