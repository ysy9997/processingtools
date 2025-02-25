import cv2
import os
import glob
import multiprocessing as mp
import argparse

import numpy as np

import processingtools.ProgressBar
import time
import warnings
import typing


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

    def video2images(self, save_path: str, extension: str = 'jpg', start: float = 0, end: float = None, jump: float = 1, option: str = 'frame', size=None) -> True:
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

        create_folder(save_path, print_warning=False)

        for n, i in processingtools.ProgressBar(enumerate(range(self.length)), total=self.length, finish_mark=None):
            ret, frame = self.cap.read()
            if start <= n and ret and n % jump == 0:
                frame = frame if size is None else cv2.resize(frame, (size[1], size[0]))
                imwrite(f'{save_path}/{self.video_name}_{zero_padding(self.length, i)}.{extension}', frame)

            if n > end:
                print('\r', end='\r')
                break

        print('\rImage writing done.')

        return True

    def video_resize(self, save_path: str, size) -> True:
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

        for _ in processingtools.ProgressBar(range(self.length), total=self.length, finish_mark=None):
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

    def duplicate_func(self, func, args_list: typing.Union[tuple, list], progress_args: typing.Union[dict, bool] = True, debug: bool = False):
        """
        Run the function as multiprocess
        :param func: the function for running multiprocess
        :param args_list: arguments for function
        :param progress_args: arguments for ProgressBar. if False, it doesn't use progressbar, if True it uses progressbar.
        :param debug: if True, runs the function in single process mode for debugging
        :return: True
        """

        if type(progress_args) is not dict and type(progress_args) is not bool:
            raise TypeError('type of progress_args is needed dict or bool')

        if debug:
            # Single process mode for debugging
            results = [func(*args) for args in args_list]
            if progress_args is True:
                for result in processingtools.ProgressBar(results):
                    pass
            elif progress_args:
                for result in processingtools.ProgressBar(results, **progress_args):
                    pass
            else:
                for result in results:
                    pass

        else:
            # Multi-process mode
            with mp.Pool(processes=self.cpu_n) as pool:
                results = [pool.apply_async(func, args) for args in args_list]
                if progress_args is True:
                    for result in processingtools.ProgressBar(results):
                        result.get()
                elif progress_args:
                    for result in processingtools.ProgressBar(results, **progress_args):
                        result.get()
                else:
                    for result in results:
                        result.get()
        return True

    def multi_func(self, funcs: typing.Union[tuple, list], args: typing.Union[tuple, list], progress_args: typing.Union[dict, bool] = True, debug: bool = False):
        """
        Run the function as multiprocess
        :param funcs: the functions for running multiprocess
        :param args: arguments for function
        :param progress_args: arguments for ProgressBar. if False, it doesn't use progressbar, if True it uses progressbar.
        :param debug: if True, runs the functions in single process mode for debugging
        :return: True
        """

        if type(progress_args) is not dict and type(progress_args) is not bool:
            raise TypeError('type of progress_args is needed dict or bool')

        if debug:
            # Single process mode for debugging
            results = [func(*arg) for func, arg in zip(funcs, args)]
            if progress_args is True:
                for result in processingtools.ProgressBar(results):
                    pass
            elif progress_args:
                for result in processingtools.ProgressBar(results, **progress_args):
                    pass
            else:
                for result in results:
                    pass

        else:
            with mp.Pool(processes=self.cpu_n) as pool:
                results = [pool.apply_async(func, args) for func, args in zip(funcs, args)]
                if progress_args is True:
                    for result in processingtools.ProgressBar(results):
                        result.get()
                elif progress_args:
                    for result in processingtools.ProgressBar(results, **progress_args):
                        result.get()
                else:
                    for result in results:
                        result.get()
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


class DeprecationWarningC(UserWarning):
    """ Base class for warnings about deprecated features. """

    def __init__(self, *args, **kwargs):  # real signature unknown
        super().__init__(*args, **kwargs)


def warning_format(message, category, filename, lineno, line=None):
    """
    Custom warning format for warnings module.
    :param message: The warning message.
    :param category: The category of the warning.
    :param filename: The name of the file in which the warning was raised.
    :param lineno: The line number where the warning was raised.
    :param line: The line of code that raised the warning.
    :return: A string containing a warning message.
    """

    return f'{category.__name__}: {message}\n'


def custom_warning_format(func):
    """
    Decorator for applying a custom warning format to a function.
    :param func: The function to apply the custom warning format to.
    :return: The decorated function.
    """

    def wrapper(*args, **kwargs):
        original_format = warnings.formatwarning
        warnings.formatwarning = warning_format
        try:
            result = func(*args, **kwargs)
        finally:
            warnings.formatwarning = original_format
        return result
    return wrapper


@custom_warning_format
def create_folder(directory, print_warning: bool = True, warning=None):
    """
    create folder when folder is not exist
    :param directory: the path which is verified exist
    :param print_warning: print warning when folder is not exist
    :param warning: print warning when folder is not exist (argument warning will be deprecated in the next version. Use the print_warning instead.)
    :return: True when directory is created
    """

    if warning is not None:
        warnings.warn(f'argument warning will be deprecated in the next version. Use the print_warning instead.', DeprecationWarning)
        warnings.warn(f'argument warning will be deprecated in the next version. Use the print_warning instead.', DeprecationWarningC)
        print_warning = warning

    directory = os.path.abspath(directory)

    try:
        if not os.path.exists(directory):
            if print_warning:
                print(f'{directory} is created.')
            os.makedirs(directory)
            return True
        else:
            if print_warning:
                warnings.warn(f'{directory} is already exist.')
            return False
    except OSError:
        raise OSError(f'Error: Cannot create directory. ({directory})')


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


@custom_warning_format
def multi_func(func, args: tuple, cpu_n: int = mp.cpu_count()) -> True:
    """
    Run the function as multiprocess
    :param func: the function for running multiprocess
    :param args: arguments for function
    :param cpu_n: the number of cpus number that you want use (default: the number of the all cpus)
    :return: True
    """

    warnings.warn(f'{multi_func.__name__} will be deprecated in the next version. Use the class MultiProcess instead.', DeprecationWarning)
    warnings.warn(f'{multi_func.__name__} will be deprecated in the next version. Use the class MultiProcess instead.', DeprecationWarningC)

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


@custom_warning_format
def png2video(images_path: str, save_path: str, fps: int = 60, fourcc: int = cv2.VideoWriter_fourcc(*'DIVX')):
    """
    make avi file using images in path
    :param images_path: directory path for images
    :param save_path: directory path for video
    :param fps:  frame per second (default: 60)
    :param fourcc: video fourcc (default: cv2.VideoWriter_fourcc(*'DIVX'))
    :return: True
    """

    warnings.warn(f'{png2video.__name__} will be deprecated in the next version. Use the class VideoTools instead.', DeprecationWarning)
    warnings.warn(f'{png2video.__name__} will be deprecated in the next version. Use the class VideoTools instead.', DeprecationWarningC)

    # when run in window, should replace backslash
    images_path = images_path.replace('\\', '/')
    save_path = save_path.replace('\\', '/')

    files = glob.glob(images_path + '/*.png')
    files = sorted(files)

    # when run in window, glob return backslash so easteregg have to do
    for n, i in enumerate(files): files[n] = i.replace('\\', '/')

    h, w, _ = cv2.imread(files[0]).shape
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for i in processingtools.ProgressBar(files):
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
        imwrite(f'{save_path}/{name}', img)

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
    """
    easteregg function is used to pad zeros to a number based on the maximum number in the dataset.
    :param max_num: the maximum number in the dataset. it is used to determine the total number of digits.
    :param present_num: the current number that needs to be padded with zeros.
    :return: the zero-padded result.
    """

    num_digits = len(str(max_num))
    format_string = f'0{num_digits}d'

    return f'{present_num:{format_string}}'


def s_text(text, f_rgb=None, b_rgb=None, styles: tuple = ()) -> str:
    """
    prints the given text with specified color and style.
    :param text: the text to be printed
    :param f_rgb: the RGB color code for the text color
    :param b_rgb: the RGB color code for the background color
    :param styles: the styles to be applied to the text. Options are 'bold', 'tilt', 'underscore', and 'cancel'
        (must be a tuple, multiple styles can be applied)
    :return: str
    """

    # define style codes
    style_codes = {
        'bold': '\033[1m',
        'tilt': '\033[3m',
        'underscore': '\033[4m',
        'cancel': '\033[9m',
        'flicker': '\033[5m'
    }

    # apply styles to the text
    for style in styles:
        if style not in style_codes:
            raise ValueError(f"Invalid styles: {style}. Valid options are: {list(style_codes.keys())}")
        text = f'{style_codes.get(style, "")}{text}'

    # set text color
    if f_rgb:
        foreground_rgb = [max(0, min(255, int(c))) for c in f_rgb[:3]]
        text = f'\033[38;2;{foreground_rgb[0]};{foreground_rgb[1]};{foreground_rgb[2]}m{text}'

    # set background color
    if b_rgb:
        background_rgb = [max(0, min(255, int(c))) for c in b_rgb[:3]]
        text = f'\033[48;2;{background_rgb[0]};{background_rgb[1]};{background_rgb[2]}m{text}'

    return f'{text}\033[0m'


def sprint(text, f_rgb=None, b_rgb=None, styles: tuple = (), sep=' ', end='\n', file=None) -> None:
    """
    prints the given text with specified color and style.
    :param text: the text to be printed
    :param f_rgb: the RGB color code for the text color
    :param b_rgb: the RGB color code for the background color
    :param styles: the styles to be applied to the text. Options are 'bold', 'tilt', 'underscore', and 'cancel'
        (must be a tuple, multiple styles can be applied)
    :param sep: the separator to be used in the print function
    :param end: the end character to be used in the print function
    :param file: the file where the output will be written
    :return: None
    """

    print(f'{s_text(text, f_rgb, b_rgb, styles)}', sep=sep, end=end, file=file)


class TextReader:
    def __init__(self, file_path: str, line_processor: typing.Callable[[str], typing.Any] = lambda _: _.strip()):
        """
        initializes the TextReader with a file path and a parsing function.
        :param file_path: a string representing the path to the file to be read.
        :param line_processor: a function to apply to each line of the file. the default function strips whitespace from each line.
        """

        self.file_path = file_path
        self.line_processor = line_processor

    def __iter__(self):
        return self.line_gen()

    def line_gen(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield self.line_processor(line)


def save_images(images_path: list, images: typing.List[np.ndarray]) -> None:
    """
    Processes and writes images to the specified paths using multiprocessing.
    :param images_path: List of paths where images will be saved
    :param images: Numpy array of images to be saved
    :return: None
    """

    multi_processor = MultiProcess()

    args = [[image_path, image] for image_path, image in zip(images_path, images)]
    multi_processor.duplicate_func(imwrite, args, progress_args={'finish_mark': 'image write done.'})


@custom_warning_format
def imwrite(file_path: str, image: np.ndarray) -> bool:
    """
    Writes an image to the specified file path, including paths with Hangul characters.
    Supported file formats are those supported by OpenCV, such as .jpg, .png, etc.
    :param file_path: The path where the image will be saved
    :param image: Numpy array representing the image to be saved
    :return: True if the image was saved successfully, False otherwise
    """

    try:
        file_path = os.path.abspath(file_path)
        result, buffer = cv2.imencode(os.path.splitext(file_path)[1], image)

        if not result:
            warnings.warn(f"Error encoding the image for file '{file_path}'")
            return False
        buffer.tofile(file_path)
        return True

    except Exception as e:
        warnings.warn(f"Error saving file '{file_path}': {e}")
        return False


def imread(file_path: str) -> typing.Optional[np.ndarray]:
    """
    Reads an image from the specified file path, including paths with Hangul characters.
    Supported file formats are those supported by OpenCV, such as .jpg, .png, etc.
    :param file_path: The path of the image file to be read
    :return: Numpy array representing the image if read successfully, None otherwise
    """

    try:
        file_path = os.path.abspath(file_path)
        buffer = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Error decoding the image from file '{file_path}'")
            return None
        return image

    except Exception as e:
        raise e
