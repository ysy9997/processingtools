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
