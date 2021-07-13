import matplotlib.pyplot as plt
import numpy as np
import cv2

try:
    import torch
except ImportError:
    print('this module is needed pytorch!')
    raise ImportError


def torch_img_show(img):

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
        img = imgs[i].cpu().detach()
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
