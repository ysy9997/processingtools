import cv2
import glob

class ProgressBar:
    """
    The class of progress.
    This should be defined before the for loop.
    """

    def __init__(self, in_loop, bar_length: int = 40, finish_mark: str = 'progress finished!'):
        """
        The initial function
        :param length: the number of total progress
        :param bar_length: bar length
        :param finish_mark: print string what you want when progress finish
        """

        import time
        self.take = time.time()
        self.start = self.take * 1000
        self.bar_length = bar_length
        self.finish_mark = finish_mark
        self.index = 0

        if type(in_loop) == int: self.in_list = [i for i in range(in_loop)]
        else: self.in_list = in_loop

        self.length = len(self.in_list)

    def __iter__(self): return self

    def __next__(self):
        import time

        if self.index == self.length:
            bar = '█' * self.bar_length
            print(f'\r|{bar}| \033[38;5;208m100.0%\033[0m | \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m0s\033[0m\033[0m |  ', end='')
            print(f'\n{self.finish_mark}\033[5m({round(time.time() * 1000 - self.start)}ms)\033[0m\n')

            raise StopIteration
        else:
            progress_per = self.index / self.length * 100
            progress_per_str = str(int(progress_per * 10) / 10)
            bar = '█' * int(self.bar_length / 100 * progress_per)
            space = '░' * (self.bar_length - int(self.bar_length / 100 * progress_per))
            if self.index == 0: left = '...'
            else:
                left = (time.time() - self.take) * (self.length - self.index)
                if left >= 3600: left = f'{left / 3600:.1f}h'
                elif left >= 60: left = f'{round(left / 60)}m'
                else: left = f'{round(left)}s'
            print(f'\r|{bar}{space}| \033[38;5;208m{progress_per_str}%\033[0m | \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m{left}\033[0m\033[0m |  ', end='')
            self.take = time.time()

            out = self.in_list[self.index]
            self.index = self.index + 1
            return out


def png2video(images_path: str, save_path: str, fps: int = 60):
    """
    make avi file using images in path
    :param images_path: directory path for images
    :param save_path: directory path for video
    :param fps: video fps (default: 60)
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
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    length = len(files)

    for n, i in enumerate(files):
        out.write(cv2.imread(i))
        progress_bar(n, length, finish_mark='make finish')

    out.release()
    return True

def video2png(video_path: str, save_path: str):
    """
    video to png file
    :param video_path: video file directory
    :param save_path: save png directory
    :return: True
    """

    video_path = video_path.replace('\\', '/')
    save_path = save_path.replace('\\', '/')

    print('read: %s' % (video_path))
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        progress_bar(i, length, finish_mark=video_path + ' to png finish!')
        frame = cap.read()[1]
        cv2.imwrite(save_path + '_%d.png' % (i), frame)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
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
    print('video: %s loaded' % (in_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (size[1], size[0]))

    for j in range(length):
        _, frame = cap.read()
        out.write(cv2.resize(frame, (size[1], size[0])))
        
        
