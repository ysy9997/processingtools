import onnxruntime
import typing
import cv2
import numpy as np
import os

if typing.TYPE_CHECKING:
    import torch


class ONNXInferenceModel:
    def __init__(self, onnx_model_path: str,
                 mean: typing.Union[float, typing.Tuple[float, float, float], np.array] = np.array([0.0, 0.0, 0.0]),
                 std: typing.Union[float, typing.Tuple[float, float, float], np.array] = np.array([1.0, 1.0, 1.0])):
        """
        initialize
        :param onnx_model_path: onnx file path
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        """

        super().__init__()

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.ort_input_name = self.ort_session.get_inputs()[0].name

        self.mean = mean
        self.std = std

    def __call__(self, inputs: typing.Union['torch.Tensor', str, list]):
        if type(inputs) is str:
            return self.ort_session.run(None, {self.ort_input_name: self.normalize_image(inputs)})

        elif type(inputs) is list:
            _inputs = []
            for _input in inputs:
                _inputs.append(self.normalize_image(_input))
            return self.ort_session.run(None, {self.ort_input_name: np.concatenate(_inputs, axis=0)})

        else:  # if type of inputs torch.Tensor
            return self.ort_session.run(None, {self.ort_input_name: self.to_numpy(inputs)})

    def normalize_image(self, image_path: str) -> np.ndarray:
        """
        normalize image
        :param image_path: input image path
        :return: normalized image
        """
        try:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32') / 255.0
            image = np.transpose((image - self.mean) / self.std, (2, 0, 1))
            return np.expand_dims(image, axis=0)
        except Exception:
            raise Exception(f'Cannot read a image ({os.path.abspath(image_path)})!')

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        convert input tensor from torch tensor to numpy tensor
        :param tensor: torch input tensor
        :return: numpy tensor
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
