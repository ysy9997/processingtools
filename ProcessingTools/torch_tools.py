import matplotlib.pyplot as plt
import numpy as np

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