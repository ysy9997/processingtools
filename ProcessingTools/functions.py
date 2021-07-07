import cv2
import numpy as np
import os
import glob
import multiprocessing as mp
import argparse
import ProcessingTools.PrgressBar


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
    for i in ProcessingTools.PrgressBar.ProgressBar(range(length)):
        ret, frame = cap.read()
        cv2.imwrite(f'{save_path}/{i:{zeros}}.png', frame) if ret else None

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
            print(f'\033[31m{directory} is created. \033[0m') if warning else None
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

    for i in ProcessingTools.PrgressBar.ProgressBar(files):
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


def print_write(string: str, file):
    """
    Write text in a text file with printing in the console.
    :param string: string to write
    :param file: text file
    :return: True
    """

    print(string, file=file)
    print(string)

    return True
