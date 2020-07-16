import cv2
import glob

ef progress_bar(progress: int, length: int, bar_length: int = 50, finish_mark: str='progress finished!'):
    """
    print progress
    :param progress: the number of present progress
    :param length: the number of total progress
    :param bar_length: bar length
    :param finish_mark: print string what you want when progress finish
    :return: return: True
    """

    progress = progress + 1
    progress_per = progress / length * 100
    progress_per_str = str(int(progress_per * 10) / 10)
    bar = '█' * int(bar_length / 100 * progress_per)
    space = '░' * (bar_length - int(bar_length / 100 * progress_per))

    print('|\r%s%s|    %s%%    %d/%d' % (bar, space, progress_per_str, progress, length), end='')
    if progress == length: print('\n' + finish_mark)

    return True


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
        
        
