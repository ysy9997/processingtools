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
    imgs = torch.unsqueeze(imgs, dim=0) if len(imgs.shape) == 3 else imgs
    if imgs.shape[1] != 3:
        print('\033[31mInput must be torch tensor images (Tensor shape must be (?, 3, ?, ?))\033[0m')
        return False

    imgs = imgs.clone()
    imgs = torch_img_denormalize(imgs)
    zeros = int(np.log10(imgs.shape[0])) + 1

    for i in range(imgs.shape[0]):
        img = imgs[i].cpu().detach()
        cv2.imwrite(f'{save_folder_path}{i:0{zeros}}.png', np.array(img.permute(1, 2, 0)).astype(int)[:, :, ::-1])

    return True
