import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import torch
import torchvision.transforms
import torchvision.transforms.functional
import torch.nn.functional


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


def torch_img_show(img):
    """
    :param img: torch tensor image
    """

    img = (img - torch.min(img))
    img = img / torch.max(img) * 255
    plt.imshow(np.array(img.permute(1, 2, 0)).astype(int))
    plt.show()


def torch_img_denormalize(imgs):
    """
    :param imgs: torch tensor
    :return: denormalized images
    """

    for i in range(imgs.shape[0]):
        imgs[i] = (imgs[i] - torch.min(imgs[i]))
        imgs[i] = imgs[i] / torch.max(imgs[i]) * 255

    return imgs


def torch_imgs_save(imgs, save_path: str = './'):
    """
    Save images in png files
    :param imgs: torch tensor
    :param save_path: save path
    :return: True if normal, otherwise False
    """

    if type(imgs) != torch.Tensor:
        print('\033[31mInput must be torch tensor images (Tensor shape must be (?, 3, ?, ?))\033[0m')
        return False

    imgs = torch.unsqueeze(torch.unsqueeze(imgs, dim=0), dim=0) if len(imgs.shape) == 2 else imgs
    imgs = torch.unsqueeze(imgs, dim=0) if len(imgs.shape) == 3 else imgs

    if imgs.shape[1] != 3 and imgs.shape[1] != 1:
        print('\033[31mInput must be torch tensor images (Tensor shape must be (?, 3, ?, ?) if color scale or (?, 1, ?, ?) if gray scale)\033[0m')
        return False

    imgs = imgs.clone()
    imgs = torch_img_denormalize(imgs)
    zeros = int(np.log10(imgs.shape[0])) + 1

    for i in range(imgs.shape[0]):
        img = imgs[i].cpu().detach().clone()
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
