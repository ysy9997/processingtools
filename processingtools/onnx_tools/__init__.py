try:
    import onnx
except ModuleNotFoundError:
    raise ModuleNotFoundError('onnx_tools is needed onnx! (try "pip install onnx")')

try:
    import onnxruntime
except ModuleNotFoundError:
    raise ModuleNotFoundError('onnx_tools is needed onnxruntime! (try "pip install onnxruntime or pip install '
                              'onnxruntime-gpu")')

from processingtools.onnx_tools.onnx_function import *
from processingtools.onnx_tools.ONNXProcess import *
