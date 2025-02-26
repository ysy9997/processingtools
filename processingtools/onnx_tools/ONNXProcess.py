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

    def __call__(self, inputs: typing.Union['torch.Tensor', str, list], batch_size: int = 1):
        """
        Process inputs using ONNX model
        :param inputs: list of image paths, or single image path, or torch.Tensor
        :param batch_size: size of the batch for processing inputs
        """

        if isinstance(inputs, str):
            return self.ort_session.run(None, {self.ort_input_name: self.normalize_image(inputs)})

        if isinstance(inputs, list):
            # Normalize inputs
            normalize_inputs = [self.normalize_image(_input) for _input in inputs]

            # Process in batches
            batch_results = [
                self.ort_session.run(None, {self.ort_input_name: np.concatenate(normalize_inputs[i:i + batch_size], axis=0)})
                for i in range(0, len(normalize_inputs), batch_size)
            ]

            # Combine results and organize output
            total_outputs = np.concatenate(batch_results, axis=0)
            out_dict = {'results': {path: out for out, path in zip(total_outputs, inputs)}, 'total outputs': total_outputs}

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
