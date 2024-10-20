import time
import itertools
import numpy as np


class ProgressBar:
    """
    The class of progress.
    This should be defined begin for loop.
    example:
        for x in ProgressBar(100)
        for x in ProgressBar(range(0, 100))
        for x in ProgressBar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    """

    def __init__(self, in_loop, bar_length: int = 40, start_mark: str = None, finish_mark='progress done!',
                 total: int = None, detail_func: callable = None, smoothing: int = None):
        """
        The initial function
        :param in_loop: the input loop
        :param bar_length: bar length
        :param start_mark: print string when the progress start
        :param finish_mark: print string what you want when progress finish
        :param total: total value. If you do not fill easteregg, it will calculate automatically, but it may be slow
        :param detail_func: write detail using detail_func
        :param smoothing: make stable when estimate time taking
        """

        print(start_mark) if start_mark is not None else None

        self.start = time.time() * 1000  # for the total take time
        self.bar_length = bar_length
        self.finish_mark = finish_mark
        self.index = 0
        self.detail_func = detail_func

        self.it = iter([i for i in range(in_loop)]) if type(in_loop) is int else iter(in_loop)

        if total: self.length = total
        else:
            self.it, copy_it = itertools.tee(self.it)
            self.length = 0
            for _ in iter(copy_it): self.length = self.length + 1

        self.smoothing = self.length // 100 if smoothing is None else smoothing
        self.smoothing = 10 if self.smoothing < 10 else self.smoothing

        self.take = np.zeros(self.smoothing, float)
        T = time.time()
        for i in range(self.smoothing): self.take[i] = T

    def __iter__(self):
        return self

    def print_info(self, bar, space, progress_per_str, left, out, end=False):
        if self.detail_func and not end:
            print(f'\r\033[K\033[2K|{bar}{space}| \033[38;5;208m{progress_per_str}%\033[0m |'
                  f' \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m{left}\033[0m\033[0m |'
                  f' {self.detail_func(out)} |', end='', flush=True)
        else:
            print(f'\r\033[K\033[2K|{bar}{space}| \033[38;5;208m{progress_per_str}%\033[0m |'
                  f' \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m{left}\033[0m\033[0m |', end='',
                  flush=True)
        if end:
            print('\r\033[K', end='\r', flush=True)

    def __next__(self):
        """
        the iteration phase
        :return: the consist of for loop
        """

        # when the loop finished
        if self.index == self.length:
            bar = ''.join([f'\033[38;2;{255 - i * 255 // self.bar_length};{255};{0}m█' for i in range(self.bar_length)])
            bar = f'{bar}\033[0m'
            self.print_info(bar, '', '100.0', '0s', None, end=True)

            if self.finish_mark:
                print(f'\n\033[5m{self.finish_mark}\033[0m ({round(time.time() * 1000 - self.start)}ms)\n')

            raise StopIteration
        else:
            progress_per = self.index / self.length * 100
            progress_per_str = str(int(progress_per * 10) / 10)
            bar = '\033[38;2;255;255;0m' + ''.join([f'\033[38;2;{255 - i * 255 // self.bar_length};{255};{0}m█' for i in range(int(self.bar_length / 100 * progress_per))])
            space = '░' * (self.bar_length - int(self.bar_length / 100 * progress_per))
            space = f'{space}\033[0m'

            if self.index == 0:
                # The first loop is not finished yet, so that it cannot be calculated
                left = '...'
            else:
                take_temp = np.zeros(self.smoothing, float)
                take_temp[:self.smoothing - 1] = self.take[1:self.smoothing]
                take_temp[self.smoothing - 1] = time.time()

                # make time smooth
                if self.index >= self.smoothing:
                    left = np.mean(take_temp - self.take) * (self.length - self.index)
                else:
                    left = np.sum(take_temp - self.take) * (self.length - self.index) / self.index

                self.take = take_temp

                if left >= 3600:
                    left = f'{left / 3600:.1f}h'
                elif left >= 60:
                    left = f'{round(left / 6) / 10:.1f}m'
                else:
                    left = f'{round(left * 10) / 10:.1f}s'

            out = next(self.it)
            self.index = self.index + 1
            self.print_info(bar, space, progress_per_str, left, out)

            return out
