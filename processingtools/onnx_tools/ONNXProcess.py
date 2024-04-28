import onnxruntime


class ONNXInferenceModel:
    def __init__(self, onnx_model_path: str):
        """
        initialize
        :param onnx_model_path: onnx file path
        """

        super().__init__()

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.ort_input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, inputs):
        return self.ort_session.run(None, {self.ort_input_name: self.to_numpy(inputs)})

    @staticmethod
    def to_numpy(tensor):
        """
        convert input tensor from torch tensor to numpy tensor
        :param tensor: torch input tensor
        :return: numpy tensor
        """

        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
