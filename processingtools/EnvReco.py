import os
import shutil
import json
import time
import datetime
import functions


class EnvReco:
    """
    The class of Environments recorder.
    """
    def __init__(self, save_path: str, project_root_path: str = None, space: str = '\n', varify_exist: bool = True):
        """
        The initial function
        :param save_path: save path for logs
        :param project_root_path: project root path
        """

        self.save_path = os.path.abspath(save_path)

        if varify_exist and os.path.exists(self.save_path):
            raise OSError(f'{self.save_path} already exist.')
        if not os.path.exists(self.save_path):
            functions.create_folder(self.save_path, warning=False)

        self.logs = open(f'{save_path}/logs.txt', 'a')
        self.project_root_path = os.path.dirname(os.path.abspath(__file__)) if project_root_path is None else project_root_path
        self.timer = time.time()
        self.present = datetime.datetime
        self.__start = True
        self.space = space

        self.args = None
        self.gpu = None
        self.os = None

        print(f'Record in the \"{self.save_path}\".')

    def record_code(self, folder_name: str = 'snapshot') -> True:
        """
        record project code
        :param folder_name:
        :return: True
        """

        if self.project_root_path in self.save_path:
            raise OSError('[record_code] will save the current folder. '
                          'Thus, the save path must not include the current path.')

        shutil.copytree(f'{self.project_root_path}', f'{self.save_path}/{folder_name}/')

        return True

    def record_arg(self, args, save_type: str = 'txt', print_console: bool = True):
        """

        :param args: input arguments
        :param print_console:
        :param save_type:
        :return: True
        """

        args = self.arg2abs(args)

        self.put_space(print_console)
        self.print_if_true('Args: ', print_console)

        if save_type not in ['txt', 'text', 'json']:
            raise ValueError('save_type must be \'txt\' or \'text\' or \'json\'')

        if save_type == 'json':
            with open(f'{self.save_path}/args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=4)
        else:
            args_dict = args.__dict__
            print('{', file=self.logs)
            for key in args_dict:
                print(f'    {key}: {args_dict[key]}', file=self.logs)
            print('}', file=self.logs)

        if print_console:
            args_dict = args.__dict__
            print('{')
            for key in args_dict:
                print(f'    {key}: {args_dict[key]}')
            print('}')

        self.args = args.__dict__

        return self.args, args

    def record_os(self, keys: list = None, print_console: bool = True):

        self.put_space(print_console)
        self.print_if_true('OS Env: ', print_console)

        os_env = os.environ
        if keys:
            self.print_if_true('{', print_console)
            for key in keys:
                self.print_if_true(f'    {key}: {os_env[key]}', print_console)
            self.print_if_true('}', print_console)

        self.os = os_env

        return self.os

    def record_gpu(self, print_console: bool = True):
        self.put_space(print_console)
        self.print_if_true('GPU Info: ', print_console)

        try:
            import torch

            gpu = {'cuda': torch.cuda.is_available(),
                   'num': torch.cuda.device_count(),
                   'names': [torch.cuda.get_device_name(_) for _ in range(torch.cuda.device_count())]}

        except Exception:
            raise ImportError('this function is needed pytorch!')

        self.gpu = gpu
        return self.gpu

    def record_present(self, log: str) -> True:
        """
        write and print log with present time
        :param log: log
        return True
        """

        now = self.present.now()
        functions.print_write(f'\033[95m[{now.year}-{now.month}-{now.date} '
                              f'{now.hour}:{now.minute}:{now.second}.{now.microsecond:0.2f}]\033[0m: {log}', self.logs)

        return True

    def put_space(self, print_console: bool = True) -> bool:
        if self.__start:
            self.__start = True
            return False

        else:
            self.print_if_true(self.space, print_console)

        return True

    def print_if_true(self, contents: str, print_console: bool):
        if print_console:
            functions.print_write(contents, self.logs)
        else:
            print(contents, file=self.logs)

    def logging(self, contents: str, space: bool = False, print_console: bool = True):
        if space:
            self.put_space()

        self.print_if_true(contents, print_console)

        return True

    @staticmethod
    def arg2abs(args):
        for attr in dir(args):
            value = getattr(args, attr)
            if type(value) == str and os.path.exists(value):
                setattr(args, attr, os.path.abspath(value))

        return args


if __name__ == '__main__':
    env = EnvReco('./save')
    env.record_code()
