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

    def __init__(self, in_loop, bar_length: int = 40, start_mark: str = None, finish_mark='progress finished!', max=False):
        """
        The initial function
        :param in_loop: the input loop
        :param bar_length: bar length
        :param start_mark: print string when the progress start
        :param finish_mark: print string what you want when progress finish
        :param max: max value. If you do not fill this, it will calculate automatically but it may has memory leak
        """

        print(start_mark) if start_mark is not None else None

        self.take = np.zeros(10, float)
        T = time.time()
        for i in range(10): self.take[i] = T

        self.start = time.time() * 1000  # for the total take time
        self.bar_length = bar_length
        self.finish_mark = finish_mark
        self.index = 0

        self.it = iter([i for i in range(in_loop)]) if type(in_loop) == int else iter(in_loop)

        if max: self.length = max
        else:
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
                if self.index >= 10:
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

            print(
                f'\r|{bar}{space}| \033[38;5;208m{progress_per_str}%\033[0m | \033[38;5;177m{self.index}/{self.length}\033[0m | \033[38;5;43m{left}\033[0m\033[0m |  ',
                end='')

            out = next(self.it)
            self.index = self.index + 1
            return out
