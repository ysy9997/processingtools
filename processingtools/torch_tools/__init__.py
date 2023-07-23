try:
    import torch
except ImportError:
    raise ImportError('torch_tools is needed pytorch!')

from processingtools.torch_tools.torch_function import *
from processingtools.torch_tools.TorchProcess import *
