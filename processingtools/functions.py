import cv2
import numpy as np
import os
import glob
import multiprocessing as mp
import argparse
import processingtools.PrgressBar
import time


class VideoTools:
    def __init__(self, video_path: str):
        print(f'Input Video: {os.path.abspath(video_path)}')

        self.video_path = video_path

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f'Video path: {os.path.abspath(video_path)} is not exist or cannot be read.')

        self.cap = None
        self.length = None
        self.fourcc = None
        self.fps = None
        self.width = None
        self.height = None
        self.video_name = None

    def initial_video_capture(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f'Video path: {os.path.abspath(self.video_path)} is cannot be read.')

        self.length = round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = round(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_name = os.path.basename(self.video_path)[:-4]

    def video2images(self, save_path: str, extension: str = 'jpg', start: float = 0, end: float = None, jump: float = 1, option: str = 'frame', size=None):
        """
        video to image files
        :param save_path: video file directory, save_path: save png directory
        :param extension: file extension
        :param start: start frame
        :param end: end frame
        :param jump: jump frame
        :param option: 'second' or 'frame'
        :param size: if size type is tuple resize (height, width), else if size type float, resize size times
        :return: True
        """

        self.initial_video_capture()

        if end is None:
            end = self.length

        if option == 'second' or option == 's':
            start, end, jump = self.second2frame(start, end, jump)

        if type(size) is float or type(size) is int:
            size = [round(self.height * size), round(self.width * size)]

        create_folder(save_path)

        for n, i in processingtools.PrgressBar.ProgressBar(enumerate(range(self.length)), total=self.length, finish_mark=None):
            ret, frame = self.cap.read()
            if start <= n and ret and n % jump == 0:
                frame = frame if size is None else cv2.resize(frame, (size[1], size[0]))
                cv2.imwrite(f'{save_path}/{self.video_name}_{zero_padding(self.length, i)}.{extension}', frame)

            if n > end:
                print('\r', end='\r')
                break

        print('\rImage writing done.')

        return True

    def video_resize(self, save_path: str, size):
        """
        video resize as size
        :param save_path: save_path path
        :param size: if size type is tuple resize (height, width), else if size type float, resize size times
        :return: True
        """

        self.initial_video_capture()

        if type(size) is float or type(size) is int:
            size = [round(self.height * size), round(self.width * size)]
        out = cv2.VideoWriter(save_path, self.fourcc, self.fps, (size[1], size[0]))

        for _ in processingtools.PrgressBar.ProgressBar(range(self.length), total=self.length, finish_mark=None):
            _, frame = self.cap.read()
            out.write(cv2.resize(frame, (size[1], size[0])))

        print('\rVideo resize done.')

        return True

    def second2frame(self, *args):
        outputs = list()

        for arg in args:
            outputs.append(arg * self.fps)

        return outputs

    def video2gif(self, save_path: str, speed: float = 1, size=1):
        try:
            import moviepy.editor
        except ModuleNotFoundError:
            raise ModuleNotFoundError('video2gif is needed moviepy! Try <pip install moviepy>.')

        if type(size) is tuple or type(size) is list:
            size = size[::-1]

        moviepy.editor.VideoFileClip(self.video_path).resize(size).speedx(speed).write_gif(save_path)


class MultiProcess:
    def __init__(self, cpu_n: int = mp.cpu_count()):
        """
        initial function
        :param cpu_n: the number of cpus number that you want use (default: the number of the all cpus)
        """

        self.cpu_n = cpu_n

    def duplicate_func(self, func, args: tuple):
        """
        Run the function as multiprocess
        :param func: the function for running multiprocess
        :param args: arguments for function
        :return: True
        """

        i = 0
        j = 0

        if self.cpu_n < len(args):
            for i in range(len(args) // self.cpu_n):
                pro = list()
                for j in range(self.cpu_n):
                    pro.append(mp.Process(target=func, args=args[i * self.cpu_n + j]))
                for mul in pro: mul.start()
                for mul in pro: mul.join()

            pro = list()
            for left in range(self.cpu_n * i + j + 1, len(args)):
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

    def multi_func(self, funcs: tuple, args: tuple):
        """
        Run the function as multiprocess
        :param funcs: the functions for running multiprocess
        :param args: arguments for function
        :return: True
        """

        i = 0
        j = 0

        if self.cpu_n < len(args):
            for i in range(len(args) // self.cpu_n):
                pro = list()
                for j in range(self.cpu_n):
                    pro.append(mp.Process(target=funcs[i * self.cpu_n + j], args=args[i * self.cpu_n + j]))
                for mul in pro: mul.start()
                for mul in pro: mul.join()

            pro = list()
            for left in range(self.cpu_n * i + j + 1, len(args)):
                pro.append(mp.Process(target=funcs[left], args=args[left]))
            for mul in pro: mul.start()
            for mul in pro: mul.join()

        else:
            pro = list()
            for left in range(0, len(args)):
                pro.append(mp.Process(target=funcs[left], args=args[left]))
            for mul in pro: mul.start()
            for mul in pro: mul.join()

        return True

    def split_list(self, *args):
        """
        split list by the number of self.cpu_n
        :param args: input lists
        :return: split list
        """

        outputs = list()
        quotient, remainder = divmod(len(args[0]), self.cpu_n)

        for arg in args:
            result = list()
            for i in range(self.cpu_n):
                start = i * quotient + min(i, remainder)
                end = (i + 1) * quotient + min(i + 1, remainder)
                result.append(arg[start:end])

            outputs.append(result)

        return outputs

    @staticmethod
    def wrapper(data, *args, **kwargs):
        try:
            import dill
        except ModuleNotFoundError:
            raise ModuleNotFoundError('wrapper is needed dill! Try <pip install dill>.')

        func = dill.loads(data)
        return func(*args, **kwargs)

    @staticmethod
    def adapt_function(function, order=False):
        """
        Example

        processor = pt.MultiProcess()
        outputs = Queue()  # outputs need to be queued.
        adapt_func = processor.adapt_function(FUNCTION, order=True)

        processor.duplicate_func(pt.wrapper, ((adapt_func, 1, outputs, (PRAMS, )), (adapt_func, 0, outputs, (PRAMS, ))))
        print([outputs.get() for _ in range(2)])

        :param function: multiple functions
        :param order: if you need to order it will be True.
        :return: dill function
        """

        try:
            import dill
        except ModuleNotFoundError:
            raise ModuleNotFoundError('adapt_function is needed moviepy! Try <pip install dill>.')

        if order:
            def adapted_function(num, outputs: mp.queues.Queue, *args, **kwargs):
                output = function(*args, **kwargs)
                outputs.put((num, output))

        else:
            def adapted_function(outputs: mp.queues.Queue, *args, **kwargs):
                outputs.put(function(*args, **kwargs))

        return dill.dumps(adapted_function)


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


def read_images_list(dir_path: str, img_format: str = None):
    """
    return the tuple that is all images name
    :param dir_path: the images' folder
    :param img_format: images format (e.g. 'png' or 'jpg')
    :return: the tuple that is all images name
    """

    if img_format is None:
        images_png = glob.glob(f'{dir_path}/*.png')
        images_jpg = glob.glob(f'{dir_path}/*.jpg')
        return sorted(images_png + images_jpg)

    else:
        return sorted(glob.glob(f'{dir_path}/*.{img_format}'))


def read_images(dir_path: str, img_format: str = None):
    """
    return the tuple that is all images name
    :param dir_path: the images' folder
    :param img_format: images format (e.g. 'png' or 'jpg')
    :return: the tuple that is all images name
    """

    if img_format is None:
        images_png = glob.glob(f'{dir_path}/*.png')
        images_jpg = glob.glob(f'{dir_path}/*.jpg')
        return [cv2.imread(_) for _ in sorted(images_png + images_jpg)]

    else:
        return [cv2.imread(_) for _ in sorted(glob.glob(f'{dir_path}/*.{img_format}'))]


def multi_func(func, args: tuple, cpu_n: int = mp.cpu_count()):
    """
    Run the function as multiprocess
    :param func: the function for running multiprocess
    :param args: arguments for function
    :param cpu_n: the number of cpus number that you want use (default: the number of the all cpus)
    :return: True
    """

    i = 0
    j = 0

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

    for i in processingtools.PrgressBar.ProgressBar(files):
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


def resize_images(dir_path: str, save_path: str, size, interpolation=None, img_format: str = None):
    """
    return the tuple that is all images name
    :param dir_path: the images' folder
    :param save_path: save path
    :param size: size for resize that you want
    :param interpolation: interpolation parameter for opencv resize function
    :param img_format: images format (e.g. 'png' or 'jpg')
    :return: True
    """

    images = read_images_list(dir_path, img_format)

    for i in images:
        img = cv2.imread(i)
        img = cv2.resize(img, size, interpolation=interpolation)
        name = os.path.basename(i)
        cv2.imwrite(f'{save_path}/{name}', img)

    return True


def write_text_image(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None):
    text_size = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    img = img.copy()

    if org[0] == 'center':
        x = (img.shape[1] - text_size[0]) // 2
    elif org[0] == 'left':
        x = 0
    elif org[0] == 'right':
        x = (img.shape[1] - text_size[0])
    else:
        x = org[0]

    if org[1] == 'center':
        y = (img.shape[0] - text_size[1]) // 2
    elif org[1] == 'bottom':
        y = (img.shape[0] - text_size[1])
    elif org[1] == 'top':
        y = text_size[1]
    else:
        y = org[1]

    return cv2.putText(img, text, (x, y), fontFace, fontScale, color, thickness=thickness, lineType=lineType,
                       bottomLeftOrigin=bottomLeftOrigin)


def timer(input_function):
    """
    count time for input function
    :param input_function: input function
    return input function output
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        output = input_function(*args, **kwargs)
        print(f'\033[95m[{input_function.__name__}]\033[0m elapsed \033[97m{time.time() - start:0.2f}\033[0m sec.')

        return output

    return wrapper


def zero_padding(max_num, present_num):
    n_zero = int(np.log10(max_num)) + 1
    zeros = f'0{n_zero}d'

    return f'{present_num:{zeros}}'
