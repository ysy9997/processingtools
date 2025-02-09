import onnxruntime
import typing
import cv2
import numpy as np
import os

if typing.TYPE_CHECKING:
    import torch


class ONNXInferenceModel:
    def __init__(self, onnx_model_path: str, size: typing.Union[tuple, list, None] = None,
                 mean: typing.Union[float, typing.Tuple[float, float, float], np.array] = np.array([0.0, 0.0, 0.0]),
                 std: typing.Union[float, typing.Tuple[float, float, float], np.array] = np.array([1.0, 1.0, 1.0])):
        """
        initialize
        :param onnx_model_path: onnx file path
        :param size: size to which images will be resized
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        """

        super().__init__()

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.ort_input_name = self.ort_session.get_inputs()[0].name

        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, inputs: typing.Union['torch.Tensor', str, list]):
        if type(inputs) is str:
            return self.ort_session.run(None, {self.ort_input_name: self.normalize_image(inputs)})

        elif type(inputs) is list:
            out_dict = {'results': {}}
            outputs = []

            for _input in inputs:
                out = self.normalize_image(_input)[0]
                outputs.append(out)
                out_dict['results'][_input] = out[0]

            out_dict['total outputs'] = np.concatenate(outputs, axis=0)

            return out_dict

        else:  # if type of inputs torch.Tensor
            return self.ort_session.run(None, {self.ort_input_name: self.to_numpy(inputs)})

    def normalize_image(self, image_path: str) -> np.ndarray:
        """
        normalize image
        :param image_path: input image path
        :return: normalized image
        """
        try:
            image = cv2.imread(image_path)
            if self.size is not None:
                image = cv2.resize(image, self.size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = np.transpose((image - self.mean) / self.std, (2, 0, 1))
            return np.expand_dims(image, axis=0).astype('float32')
        except Exception:
            raise Exception(f'Cannot read a image ({os.path.abspath(image_path)})!')

    @staticmethod
    def to_numpy(tensor: 'torch.Tensor') -> np.ndarray:
        """
        convert input tensor from torch tensor to numpy tensor
        :param tensor: torch input tensor
        :return: numpy tensor
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
