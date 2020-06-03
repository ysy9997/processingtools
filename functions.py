def progress_bar(progress: int, length: int, bar_length: int = 50, finish_mark: str='progress finish!'):
    """
    print progress

    progress: present progress, length: total progress, bar_length: bar length,
    finish_mark: print string what you want when progress finish

    return: True
    """

    progress = progress + 1
    progress_per = progress / length * 100
    progress_per_str = str(int(progress_per * 10) / 10)
    bar = '█' * int(bar_length / 100 * progress_per)
    space = '░' * (bar_length - int(bar_length / 100 * progress_per))

    print('\r%s%s    %s%%    %d/%d' % (bar, space, progress_per_str, progress, length), end='')
    if progress == length: print('\n' + finish_mark)

    return True


def make_video(str: path):
    """
    make avi file using images in path
    path: directory path for images
    return True
    """
    import cv2
    import glob
    #from tqdm import tqdm
    import tqdm
    
    files = glob.glob(path + '/*.png')
    files = sorted(files)


    h, w, _ = cv2.imread(files[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 60, (w,h))

    for i in tqdm.tqdm(files):
        out.write(cv2.imread(i))
     
    out.release()
    return True
