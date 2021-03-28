import cv2
import numpy as np
import os
import glob
import multiprocessing as mp
import argparse
import time
import matplotlib.pyplot as plt
import itertools


class ProgressBar:
    """
    The class of progress.
    This should be defined begin for loop.
    example:
        for x in ProgressBar(100)
        for x in ProgressBar(range(0, 100))
        for x in ProgressBar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    """

    def __init__(self, in_loop, bar_length: int = 40, start_mark: str = None, finish_mark: str = 'progress finished!'):
        """
        The initial function
        :param in_loop: the input loop
        :param bar_length: bar length
        :param start_mark: print string when the progress start
        :param finish_mark: print string what you want when progress finish
        """

        if start_mark is not None: print(start_mark)

        self.take = np.zeros(10, float)
        T = time.time()
        for i in range(10): self.take[i] = T

        self.start = time.time() * 1000  # for the total take time
        self.bar_length = bar_length
        self.finish_mark = finish_mark
        self.index = 0

        if type(in_loop) == int:
            self.it = iter([i for i in range(in_loop)])
        else:
            self.it = iter(in_loop)

        self.it, copy_it = itertools.tee(self.it)
        self.length = 0
        for _ in iter(copy_it): self.length = self.length + 1

    def __iter__(self):
        return self

    def __next__(self):
        """
        the iteration phase
        :return: the consist of for loop
        """

        # when the loop finished
        if self.index == self.length:
            bar = '█' * self.bar_length
            print(
                f'\r|{bar}| \033[38;5;208m100.0%\033[0m | \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m0s\033[0m\033[0m |  ',
                end='')
            if self.finish_mark:
                print(f'\n\033[5m{self.finish_mark}\033[0m({round(time.time() * 1000 - self.start)}ms)\n')

            raise StopIteration
        else:
            progress_per = self.index / self.length * 100
            progress_per_str = str(int(progress_per * 10) / 10)
            bar = '█' * int(self.bar_length / 100 * progress_per)
            space = '░' * (self.bar_length - int(self.bar_length / 100 * progress_per))

            if self.index == 0:
                # The first loop is not finished yet, so that it cannot be calculated
                left = '...'
            else:
                take_temp = np.zeros(10, float)
                take_temp[:9] = self.take[1:10]
                take_temp[9] = time.time()

                # make time smooth
                if self.index >= 10: left = np.mean(take_temp - self.take) * (self.length - self.index)
                else: left = np.sum(take_temp - self.take) * (self.length - self.index) / self.index

                self.take = take_temp

                if left >= 3600:
                    left = f'{left / 3600:.1f}h'
                elif left >= 60:
                    left = f'{round(left / 6) / 10:.1f}m'
                else:
                    left = f'{round(left * 10) / 10:.1f}s'

            print(
                f'\r|{bar}{space}| \033[38;5;208m{progress_per_str}%\033[0m | \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m{left}\033[0m\033[0m |  ',
                end='')

            out = next(self.it)
            self.index = self.index + 1
            return out


def video2png(video_path: str, save_path: str):
    """
    video to png file
    save_path: video file directory, save_path: save png directory
    return True if the function end as normal, else False
    """

    print(f'read: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'\033[31mvideo path: {video_path} is not exist\033[0m')
        return False

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nzero = int(np.log10(length)) + 1
    zeros = f'0{nzero}d'
    for i in ProgressBar(range(length)):
        ret, frame = cap.read()
        if ret: cv2.imwrite(f'{save_path}/{i:{zeros}}.png', frame)

    return True


def video_resize(in_path: str, out_path: str, size):
    """
    video resize as size
    Args:
        in_path: input video path
        out_path: output video path
        size: resize (height, width)
    Returns: True
    """

    cap = cv2.VideoCapture(in_path)
    fourcc = round(cap.get(cv2.CAP_PROP_FOURCC))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    if cap.isOpened():
        print('video: %s loaded' % (in_path))
    else:
        print('\033[31mvideo: %s not loaded\033[0m' % (in_path))
        exit(1)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (size[1], size[0]))

    for j in range(length):
        _, frame = cap.read()
        out.write(cv2.resize(frame, (size[1], size[0])))

    return True


def create_folder(directory, warning: bool = True):
    """
    create folder when folder is not exist
    :param directory: the path which is verified exist
    :param warning: print warning when folder is not exist
    :return: True when directory is created
    """

    try:
        if not os.path.exists(directory):
            if warning: print(f'\033[31m{directory} is created. \033[0m')
            os.makedirs(directory)
            return True
        else: return False
    except OSError:
        print('Error: Creating directory. ' + directory)


def read_images(dir_path: str, img_format: str = None):
    """
    return the tuple that is all images name
    :param dir_path: the images folder
    :param img_format: images format (e.g. 'png' or 'jpg')
    :return: the tuple that is all images name
    """

    if img_format is None:
        images_png = glob.glob(f'{dir_path}/*.png')
        images_jpg = glob.glob(f'{dir_path}/*.jpg')
        return sorted(images_png + images_jpg)

    else:
        return sorted(glob.glob(f'{dir_path}/*.{img_format}'))


def multi_func(func, args: tuple, cpu_n: int = mp.cpu_count()):
    """
    Run the function as multiprocess
    :param func: the function for running multiprocess
    :param args: arguments for function
    :param cpu_n: the number of cpus number that you want use (default: the number of the all cpus)
    :return: True
    """

    if cpu_n < len(args):
        for i in range(len(args) // cpu_n):
            pro = list()
            for j in range(cpu_n):
                pro.append(mp.Process(target=func, args=args[i * cpu_n + j]))
            for mul in pro: mul.start()
            for mul in pro: mul.join()

        pro = list()
        for left in range(cpu_n * i + j + 1, len(args)):
            pro.append(mp.Process(target=func, args=args[left]))
        for mul in pro: mul.start()
        for mul in pro: mul.join()

    else:
        pro = list()
        for left in range(0, len(args)):
            pro.append(mp.Process(target=func, args=args[left]))
        for mul in pro: mul.start()
        for mul in pro: mul.join()

    return True


def png2video(images_path: str, save_path: str, fps: int = 60, fourcc: int = cv2.VideoWriter_fourcc(*'DIVX')):
    """
    make avi file using images in path
    :param images_path: directory path for images
    :param save_path: directory path for video
    :param fps: video fps (default: 60)
    :param fourcc: video fourcc (default: cv2.VideoWriter_fourcc(*'DIVX'))
    :return: True
    """

    # when run in window, should replace backslash
    images_path = images_path.replace('\\', '/')
    save_path = save_path.replace('\\', '/')

    files = glob.glob(images_path + '/*.png')
    files = sorted(files)

    # when run in window, glob return backslash so this have to do
    for n, i in enumerate(files): files[n] = i.replace('\\', '/')

    h, w, _ = cv2.imread(files[0]).shape
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for i in ProgressBar(files):
        out.write(cv2.imread(i))

    out.release()
    return True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sorted_glob(path: str, key = None):
    """
    automatically sorted glob
    :param path: glob path
    :param key: sorted key
    :return: sorted glob list
    """

    if key is None: return sorted(glob.glob(path))
    else: return sorted(glob.glob(path), key=key)


def torch_img_show(img):

    try: import torch
    except ImportError:
        print('this function is needed pytorch!')
        raise ImportError

    img = (img - torch.min(img))
    img = img / torch.max(img) * 255
    plt.imshow(np.array(img.permute(1, 2, 0)).astype(int))
    plt.show()