import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import torch
import torchvision.transforms
import torchvision.transforms.functional
import torch.nn.functional
import typing
import warnings
import processingtools.functions
import torch.utils.data


class QueueTensor:
    def __init__(self, n: int):
        """
        :param n: queue length
        """

        self.length = n
        self.contents = [None for _ in range(n)]
        self.len = 0

    def append(self, x):
        """
        :param x: input tensor
        """

        temp = self.contents[-1]
        self.contents[1:] = self.contents[0:-1]
        self.contents[0] = x

        self.len = self.len + 1 if self.len < self.length else self.len

        return temp

    def __call__(self):
        return torch.stack([_ for _ in self.contents if _ is not None])


class WeightAveraging:
    """
    Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." arXiv preprint arXiv:1803.05407 (2018).
    """

    def __init__(self):
        """
        initial function
        """

        self.weight_list = list()

    def save(self, model):
        """
        :param model: input model for averaging
        """

        self.weight_list.append(copy.deepcopy(model.state_dict()))

    def averaging(self, model):
        """
        averaging saved model
        """

        averaging_sd = dict()

        for key in self.weight_list[0]:
            averaging_sd[key] = torch.mean(torch.stack([_[key].float() for _ in self.weight_list], dim=0), dim=0)

        model.load_state_dict(averaging_sd)


class GradCam:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    (https://arxiv.org/abs/1610.02391)
    """

    def __init__(self, model):
        """
        :param model: input model
        """
        self.feature_blobs = list()
        self.backward_feature = list()
        self.model = model

    def free_list(self):
        """
        empty member lists
        """
        self.feature_blobs = list()
        self.backward_feature = list()

    def hook_feature(self, module, _input, output):
        self.feature_blobs.append(output.cpu().data.numpy())

    def backward_hook(self, module, _input, output):
        self.backward_feature.append(output[0])

    def grad_cam(self, image, ground_truth, module_name: str) -> torch.Tensor:
        """
        :param image: input image (torch.tensor)
        :param ground_truth: ground_truth class
        :param module_name: where you want to see grad-cam
        :return: grad cam
        """

        self.model._modules.get(module_name).register_forward_hook(self.hook_feature)
        self.model._modules.get(module_name).register_full_backward_hook(self.backward_hook)

        self.model.eval()
        feature = self.model(image)

        torch.argmax(feature, dim=1)

        score = feature[:, ground_truth].squeeze()
        score.backward(retain_graph=True)

        activations = torch.Tensor(self.feature_blobs[0]).to(image.device)
        gradients = self.backward_feature[0]
        b, k = gradients.shape[:2]

        alpha = torch.mean(torch.reshape(gradients, (b, k, -1)), dim=2)
        weights = torch.reshape(alpha, (b, k, 1, 1))

        grad_cam_map = torch.sum(weights * activations, dim=1, keepdim=True)
        grad_cam_map = torch.nn.functional.relu(grad_cam_map)
        grad_cam_map = torch.nn.functional.interpolate(grad_cam_map, size=image.shape[2:], mode='bilinear', align_corners=False)
        map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
        grad_cam_map = ((grad_cam_map - map_min) / (map_max - map_min)).data

        return grad_cam_map

    def draw_cam(self, image, ground_truth, module_name: str, save_path: str, name: str, free_list: bool = True) -> bool:
        """
        :param image: input image (torch.tensor)
        :param ground_truth: ground_truth class
        :param module_name: where you want to see grad-cam
        :param save_path: path for saving grad cam image
        :param name: image name
        :param free_list: free feature_blobs and backward_feature
        :return: grad cam image
        """

        grad_cam = self.grad_cam(image, ground_truth, module_name)
        cv2.imwrite(f'{save_path}/cam_{name}.png', (image[:, :, ::-1] / 2 + grad_cam / 2))

        if free_list:
            self.free_list()

        return True


class HOA(torch.nn.Module):
    """
    High Order Attention module.
    Chen, Binghui, Weihong Deng, and Jiani Hu. "Mixed high-order attention network for person re-identification."
        Proceedings of the IEEE/CVF international conference on computer vision. 2019.
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Mixed_High-Order_Attention_Network_for_Person_Re-Identification_ICCV_2019_paper.pdf
    """

    def __init__(self, in_channels: int, r: int = 3, activation: str = 'ReLU') -> None:
        """
        initial function
        :param in_channels: the number of input channels
        :param r: order for attention module
        :param activation: activation function for MOA
        """

        super().__init__()

        conv2ds = []
        for i in range(r):
            conv2ds.append(torch.nn.ModuleList([torch.nn.Conv2d(in_channels, in_channels, 1) for _ in range(i + 1)]))
        self.conv2ds = torch.nn.ModuleList(conv2ds)

        sequentials = [torch.nn.Sequential(getattr(torch.nn, activation)(), torch.nn.Conv2d(in_channels, in_channels, 1)) for _ in range(r)]
        self.sequentials = torch.nn.ModuleList(sequentials)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        cs = []
        for conv2d, sequential in zip(self.conv2ds, self.sequentials):
            z1s = [_(x) for _ in conv2d]

            c = torch.ones_like(x)
            for z1 in z1s:
                c *= z1

            c = sequential(c)
            cs.append(c)

        outs = torch.zeros_like(x)
        for c in cs:
            outs += c

        return outs * x


def torch_img_show(img):
    """
    :param img: torch tensor image
    """

    img = (img - torch.min(img))
    img = img / torch.max(img) * 255
    plt.imshow(np.array(img.permute(1, 2, 0)).astype(int))
    plt.show()


def torch_img_denormalize(image, mean: typing.Union[tuple, list, None] = None, std: typing.Union[tuple, list, None] = None):
    """
    :param image: torch tensor
    :param mean: mean for normalization
    :param std: standard deviation for normalization
    :return: denormalized images
    """
    if (mean is None and std is not None) or (mean is not None and std is None):
        raise ValueError("Both mean and std must be provided together or both must be None")

    if mean is not None and std is not None:
        for i in range(image.shape[0]):
            image = (image * std + mean) * 255
    else:
        for i in range(image.shape[0]):
            image[i] = (image[i] - torch.min(image[i]))
            image[i] = image[i] / torch.max(image[i]) * 255

    return image


def torch_imgs_save(image, save_path: str = './', mean: typing.Union[tuple, list, None] = None, std: typing.Union[tuple, list, None] = None):
    """
    Save images in png files
    :param image: torch tensor
    :param save_path: save path
    :param mean: mean for normalization
    :param std: standard deviation for normalization
    :return: True if normal, otherwise False
    """

    if type(image) != torch.Tensor:
        print('\033[31mInput must be torch tensor images (Tensor shape must be (?, 3, ?, ?))\033[0m')
        return False

    image = torch.unsqueeze(torch.unsqueeze(image, dim=0), dim=0) if len(image.shape) == 2 else image
    image = torch.unsqueeze(image, dim=0) if len(image.shape) == 3 else image

    if image.shape[1] != 3 and image.shape[1] != 1:
        print('\033[31mInput must be torch tensor images (Tensor shape must be (?, 3, ?, ?) if color scale or (?, 1, ?, ?) if gray scale)\033[0m')
        return False

    image = image.clone()
    image = torch_img_denormalize(image, mean, std)
    zeros = int(np.log10(image.shape[0])) + 1

    for i in range(image.shape[0]):
        img = image[i].cpu().detach().clone()
        cv2.imwrite(f'{save_path}{i:0{zeros}}.png', np.array(img.permute(1, 2, 0)).astype(int)[:, :, ::-1])

    return True


def fcl(in_features: int, out_features: int, mid_features: int = 1024, layers: int = 0, drop_out=1):
    """
    Make fully connected layers.
    :param in_features: size of each input feature
    :param out_features: size of each output feature
    :param mid_features: size of each hidden layers feature (default: 1024)
    :param layers: The number of fully connected layers (default: 0)
    :param drop_out:
        It will be drop out rate if bigger than zero else fully connected layers don't have a drop out layer
    :return: fully connected layers
    """

    fc = list()
    fc.append(torch.nn.Linear(in_features, mid_features, bias=False))
    fc.append(torch.nn.BatchNorm1d(mid_features))
    fc.append(torch.nn.ReLU(inplace=True))

    for i in range(layers):
        fc.append(torch.nn.Linear(mid_features, mid_features, bias=False))
        fc.append(torch.nn.BatchNorm1d(mid_features))
        fc.append(torch.nn.ReLU(inplace=True))

    fc.append(torch.nn.Linear(mid_features, out_features, bias=False))
    fc.append(torch.nn.BatchNorm1d(out_features))
    fc.append(torch.nn.ReLU(inplace=True))

    if drop_out < 1: fc.append(torch.nn.Dropout(drop_out))

    return torch.nn.Sequential(*fc)


class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """

    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        """
        initial function
        :param epsilon: epsilon for smoothing
        :param reduction: mean or sum
        """

        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    @staticmethod
    def reduce_loss(loss, reduction: str = 'mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    @staticmethod
    def linear_combination(x, y, epsilon: float):
        """
        linear combination with epsilon
        :param x: input x
        :param y: input y
        :param epsilon: epsilon of combination
        :return:
        """

        return epsilon * x + (1 - epsilon) * y

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate label smoothing loss
        :param inputs: predicted tensor
        :param target: ground truth
        :return: label smoothing loss
        """

        n = inputs.size()[-1]
        log_input = torch.nn.functional.log_softmax(inputs, dim=-1)
        loss = self.reduce_loss(-log_input.sum(dim=-1), self.reduction)
        nll = torch.nn.functional.nll_loss(log_input, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)


class ResizeKeepRatio:
    """
    Resize, but keep ratio
    """

    def __init__(self, size, interpolation=None, corner: str = 'center', value=0):
        """
        initial function
        :param size: size of resize
        :param interpolation: interpolation mode
        :param corner: how to pad. (e.g. 'center', 'TL', 'TR', 'BL', 'BR'). 'T' means Top, 'B' means Bottom, 'L' means Left, and 'R' means Right
        :param value: padding value. you can set as string (e.g. 'zero', 'one', 'noise') or float.
        """

        self.corner = corner
        self.value = value
        self.size = size

        if interpolation is None:
            try:
                # if pytorch has torchvision.transforms.functional.InterpolationMode.BICUBIC
                self.interpolation = torchvision.transforms.functional.InterpolationMode.BICUBIC
            except AttributeError:
                # if pytorch doesn't have torchvision.transforms.functional.InterpolationMode.BICUBIC
                self.interpolation = 3
        else:
            self.interpolation = interpolation

    def __call__(self, image):
        """
        resize image, but keep image ratio
        :param image: input image
        """

        if type(image) != torch.Tensor:
            image = torchvision.transforms.ToTensor()(image)

        r_target = self.size[0] / self.size[1]
        h, w = image.shape[1:]
        r = h / w

        if self.value == 'zero' or self.value == 0:
            pad = torch.zeros((3, *self.size))
        elif self.value == 'one' or self.value == 1:
            pad = torch.ones((3, *self.size))
        elif self.value == 'noise':
            pad = torch.rand((3, *self.size))
        else:
            pad = torch.ones((3, *self.size)) * self.value

        if r > r_target:
            image = torchvision.transforms.Resize(size=(self.size[0], round(self.size[0] / r)), interpolation=self.interpolation)(image)
            if self.corner[1] == 'L': pad[:, :, :round(self.size[0] / r)] = image
            elif self.corner[1] == 'R': pad[:, :, self.size[1] - round(self.size[0] / r):] = image
            else: pad[:, :, (self.size[1] - round(self.size[0] / r)) // 2:(self.size[1] + round(self.size[0] / r)) // 2] = image
        elif r < r_target:
            image = torchvision.transforms.Resize(size=(round(self.size[1] * r), self.size[1]), interpolation=self.interpolation)(image)
            if self.corner[0] == 'T': pad[:, :round(self.size[1] * r)] = image
            elif self.corner[0] == 'B': pad[:, self.size[0] - round(self.size[1] * r):] = image
            else: pad[:, (self.size[0] - round(self.size[1] * r)) // 2:(self.size[0] + round(self.size[1] * r)) // 2] = image
        else:
            pad = torchvision.transforms.Resize(size=(self.size[0], self.size[1]), interpolation=self.interpolation)(image)

        return pad


class SecondOrderDerivative(torch.nn.Module):
    def __int__(self, model, meta_model, train_loss, valid_loss, optimizer, optimizer_meta, epoch, epsilon=1e-2, grad_clip_factor=5):
        """
        Class for second order derivative.
        :param model: model
        :param meta_model: meta-model
        :param train_loss: train loss function
        :param valid_loss: validation loss function
        :param optimizer: main optimizer
        :param optimizer_meta: meta-optimizer
        :param epoch: epoch
        :param epsilon: epsilon for meta gradient
        :param grad_clip_factor: gradient clipping factor
        """

        super().__init__()
        self.model = model
        self.meta_model = meta_model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.optimizer_meta = optimizer_meta
        self.epoch = epoch
        self.epsilon = 1e-2
        self.grad_clip_factor = grad_clip_factor

    def forward(self, inputs, inputs_valid):
        """
        use second derivative
        :param inputs: inputs
        :param inputs_valid:  for validation
        """

        with torch.no_grad():
            ori_p = [copy.deepcopy(p) for p in self.model.parameters()]

        # 1. Update for meta-model
        # update model temporally
        grad_nh = self.grad_train(inputs)
        self.optimizer_train.zero_grad()
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), grad_nh):
                w.grad = g
        self.optimizer_train.step()

        epsilon, grad_val = self.obtain_epsilon(inputs_valid)

        # get w+ gradient
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), grad_val):
                if g is not None:
                    w.copy_(w + epsilon * g)

        grad_hue1 = self.grad_train(inputs)

        # get w- gradient
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), grad_val):
                if g is not None:
                    w.copy_(w - (epsilon * g * 2))

        grad_hue2 = self.grad_train(inputs)

        # update meta model
        self.optimizer_meta.zero_grad()
        with torch.no_grad():
            for w, g1, g2 in zip(self.meta_model.parameters(), grad_hue1, grad_hue2):
                w.grad = -self.cf['lr'] * (g1 - g2) / (2 * epsilon)
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), self.grad_clip_factor)
        self.optimizer_meta.step()

        # 2. Update for main model
        # get gradient
        grad, loss = self.grad_train(inputs, param=ori_p)

        # update main model
        self.optimizer.zero_grad()
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), grad):
                w.grad = g
        self.optimizer.step()

        return loss

    def grad_val(self, valid_inputs):
        """
        validation gradient
        :param valid_inputs: valid inputs (* use validation inputs *)
        """

        outputs = self.model(valid_inputs.clone().detach())
        valid_loss = self.valid_loss(outputs)

        return torch.autograd.grad(valid_loss, self.model.parameters())

    def grad_train(self, inputs, param=None):
        """
        validation gradient
        :param inputs: inputs (* do not use validation inputs *)
        :param param: origin model parameters
        """

        if param is not None:
            with torch.no_grad():
                for w, p in zip(self.model.parameters(), param):
                    w.copy_(p)

        outputs = self.model(inputs)
        loss = self.train_loss(outputs) * self.meta_model.weight

        if param is not None:
            return torch.autograd.grad(loss, self.model.parameters()), loss
        else:
            return torch.autograd.grad(loss, self.meta_model.parameters()), loss

    def obtain_epsilon(self, valid_inputs):
        """
        calculate epsilon value
        :param valid_inputs: valid inputs (* use validation inputs *)
        """

        grad_val = self.grad_val(valid_inputs)

        flatt_grad = list()
        for x in grad_val:
            if x is not None:
                flatt_grad.append(torch.reshape(x, (-1,)))
        return self.epsilon / torch.norm(torch.cat(flatt_grad)), grad_val


class AutoInputModel(torch.nn.Module):
    """
    A PyTorch module for automatically processing and normalizing input images.

    This class wraps a given model and provides functionality to read, preprocess, and forward images through the model.
    It supports custom transformers and normalization parameters.
    """

    @processingtools.functions.custom_warning_format
    def __init__(self, model, size: typing.Union[tuple, list, None] = None, mean: typing.Union[float, list, torch.Tensor, None] = None, std: typing.Union[float, list, torch.Tensor, None] = None, transformer=None):
        """
        initialize the AutoInputModel
        :param model: model to be used
        :param size: size to which images will be resized
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        :param transformer: custom transformer for image preprocessing
        """

        if transformer is None and (mean is None or std is None):
            raise ValueError('Either transformer must be provided, or both mean and std must be specified.')

        super().__init__()

        self.model = model

        if transformer is not None and (mean is not None or std is not None):
            warnings.warn('NormalizeModel uses transformer for normalizing not (mean, std).')

        if transformer is not None:
            self.transform = transformer
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])

        self.device = None

    def image_read(self, path: str) -> torch.Tensor:
        """
        read and preprocess an image from the given path
        :param path: image file path
        :return: normalized image tensor
        """

        try:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return torch.unsqueeze(self.transform(image), dim=0)
        except Exception as e:
            raise ValueError(f'Error reading image from path {path}: {e}')

    def get_device(self) -> None:
        """
        get the device to run the model on
        :return: device
        """

        self.device = next(self.model.parameters()).device
        print(f'run on {processingtools.functions.s_text(f"{self.device}", styles=("bold",))}')

    @processingtools.functions.custom_warning_format
    def forward(self, inputs: typing.Union[str, list], batch_size: int = 1, num_workers: int = 0) -> typing.Union[torch.Tensor, dict]:
        """
        forward pass of the model
        :param inputs: string or list of strings representing image paths
        :param batch_size: batch size if input is path list
        :param num_workers: workers num for dataloader if input is path list
        :return: model outputs
        """

        if self.model.training:
            warnings.warn('model is training mode now! (if you want to change eval mode, add model.eval())')

        self.get_device()

        if isinstance(inputs, list):
            outputs = []
            out_dict = {'results': {}}

            dataset = AutoInputDataset(inputs, transformer=self.transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for data in processingtools.ProgressBar(dataloader, total=len(dataloader), finish_mark=None):
                image, paths = data
                image = image.to(self.device)
                output = self.model(image)

                outputs.append(output)

                for out, path in zip(output, paths):
                    out_dict['results'][path] = out

            print('\rInference done.')
            out_dict['total outputs'] = torch.cat(outputs, dim=0)

            return out_dict

        elif isinstance(inputs, str):
            return self.model(self.image_read(inputs))
        else:
            raise TypeError('inputs must be a string or a list of strings')


class AutoInputDataset(torch.utils.data.Dataset):
    def __init__(self, image_list: list, size: typing.Union[tuple, list, None] = None, mean: typing.Union[float, list, torch.Tensor, None] = None, std: typing.Union[float, list, torch.Tensor, None] = None, transformer=None):
        """
        initialize the AutoInputDataset
        :param image_list:
        :param size: size to which images will be resized
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        :param transformer: custom transformer for image preprocessing
        """

        if transformer is not None and (mean is not None or std is not None):
            warnings.warn('NormalizeModel uses transformer for normalizing not (mean, std).')

        if transformer is not None:
            self.transform = transformer
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])

        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.transform(image), path


def gpu_env() -> None:
    """
    check and display GPU availability and details.
    :return: None
    """

    num_gpus = torch.cuda.device_count()

    print(f'GPU usable: {"Yes" if torch.cuda.is_available() else "No"}')
    print(f'Number of detected GPUs: {num_gpus}')

    for i in range(num_gpus):
        print(f'    • GPU {i}: {torch.cuda.get_device_name(i)}')


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        """
        Initialize the EnsembleModel.
        :param models: List of models to be ensembled.
        """

        super().__init__()
        self.models = models

    def forward(self, x, option: str = 'mean', weights: typing.Union[list, tuple, None] = None) -> torch.Tensor:
        """
        Perform ensemble predictions on the input data.
        :param x: Input data.
        :param option: Ensemble method ('mean' or 'WA' for weighted average).
        :param weights: Weights for weighted average. Must be the same length as models if specified.
        :return: Ensemble prediction.
        """

        # Check if the option is valid
        if option not in ['mean', 'WA', 'sum', 'voting']:
            raise ValueError("Option must be 'mean', 'WA', 'sum', or 'voting'.")

        # Perform predictions for each model
        predictions = [model(x) for model in self.models]

        if option == 'mean':
            ensemble_predictions = torch.mean(torch.stack(predictions), dim=0)

        elif option == 'WA':
            if weights is None or len(weights) != len(self.models):
                raise ValueError("Weights must be provided and match the number of models for weighted averaging.")
            weights = torch.tensor(weights, dtype=torch.float32)
            weights /= weights.sum()  # Normalize weights

            weighted_predictions = torch.stack(predictions) * weights[:, None, None, None]
            ensemble_predictions = torch.sum(weighted_predictions, dim=0)

        elif option == 'sum':
            ensemble_predictions = torch.sum(torch.stack(predictions), dim=0)

        elif option == 'voting':
            voting_results = [torch.argmax(prediction, dim=1) for prediction in predictions]
            voting_results = torch.stack(voting_results, dim=1)

            # Vote by batch and return the most common class
            ensemble_predictions = torch.mode(voting_results, dim=1).values

        else:
            ensemble_predictions = None

        return ensemble_predictions
