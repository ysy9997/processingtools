import os
import shutil
import json
import time
import datetime
import processingtools.functions


def file_opener(input_function):
    """
    file opener
    :param input_function: input function
    return input function output
    """

    def wrapper(self, *args, **kwargs):
        if self.logs.closed:
            with open(f'{self.save_path}/logs.txt', 'a') as self.logs:
                output = input_function(self, *args, **kwargs)
        else:
            output = input_function(self, *args, **kwargs)
        return output

    return wrapper


class EnvReco:
    """
    The class of Environments recorder.
    """
    def __init__(self, save_path: str, project_root_path: str = None, space: str = '', varify_exist: bool = True):
        """

        :param save_path: save path for logs
        :param project_root_path: project root path
        :param space: space between logs
        :param varify_exist: if False ignore save pate already exist or not
        """

        self.save_path = os.path.abspath(save_path)

        if varify_exist and os.path.exists(self.save_path):
            raise OSError(f'{self.save_path} already exist.')
        if not os.path.exists(self.save_path):
            processingtools.functions.create_folder(self.save_path, print_warning=False)

        self.logs = open(f'{save_path}/logs.txt', 'a')
        self.project_root_path = os.path.dirname(os.path.abspath(__file__)) if project_root_path is None else project_root_path
        self.timer = time.time()
        self.present = datetime.datetime
        self.space = space

        self.args = None
        self.gpu = None
        self.os = None

        print(f'Record in the \"{self.save_path}\".')
        self.logs.close()

    def record_code(self, folder_name: str = 'snapshot') -> True:
        """
        record project code
        :param folder_name:
        :return: True
        """

        with open(f'{self.save_path}/logs.txt', 'a') as self.logs:
            if self.project_root_path in self.save_path:
                raise OSError('[record_code] will save the current folder. '
                              'Thus, the save path must not include the current path.')

            shutil.copytree(f'{self.project_root_path}', f'{self.save_path}/{folder_name}/')

        return True

    @file_opener
    def record_arg(self, args, save_type: str = 'txt', print_console: bool = True):
        """
        record input arguments
        :param args: input arguments
        :param print_console: if True you can see logs in the console as well
        :param save_type: if 'json', it will be attached the logs file
        :return: args dictionary and absolute path arg
        """

        args = self.arg2abs(args)

        if save_type not in ['txt', 'text', 'json']:
            raise ValueError('save_type must be \'txt\' or \'text\' or \'json\'')

        if save_type == 'json':
            with open(f'{self.save_path}/args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=4)
        else:
            self.print_if_true('Args: ', print_console)
            args_dict = args.__dict__
            self.print_if_true('{', print_console)
            for key in args_dict:
                self.print_if_true(f'    {key}: {args_dict[key]}', print_console)
            self.print_if_true('}', print_console)
            self.put_space(print_console)

        self.args = args.__dict__

        return self.args, args

    @file_opener
    def record_os(self, keys: list = None, print_console: bool = True):
        """
        record os environments
        :param keys: if you insert key values, it will record only key environments
        :param print_console: if True you can see logs in the console as well
        :return: os dictionary
        """

        self.print_if_true('OS Env: ', print_console)

        os_env = os.environ
        self.log_dict(os_env, keys, print_console)

        self.os = os_env
        self.put_space(print_console)

        return self.os

    @file_opener
    def record_gpu(self, keys: list = None, print_console: bool = True) -> dict:
        """
        record gpu environments
        :param keys: if you insert key values, it will record only key environments
        :param print_console: if True you can see logs in the console as well
        :return: gpu dictionary
        """

        self.print_if_true('GPU Info: ', print_console)

        try:
            import torch

            gpu = {'cuda': torch.cuda.is_available(),
                   'num': torch.cuda.device_count(),
                   'names': [torch.cuda.get_device_name(_) for _ in range(torch.cuda.device_count())]}

        except Exception:
            raise ImportError('easteregg function is needed pytorch!')

        self.log_dict(gpu, keys, print_console)
        self.gpu = gpu
        self.put_space(print_console)

        return self.gpu

    @file_opener
    def print(self, log: str, console: bool = True, file: bool = True, time: bool = True) -> None:
        """
        write and print log with present time
        :param log: log
        :param console: if True, print in the console
        :param file: if True, write in the logs file
        :param time: if True, write in the present time before the log
        return True
        """

        now = self.present.now()
        time_info = f'[{now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d}:{now.second:02d}.' \
                    f'{now.microsecond // 10000:02d}]: ' if time else ''

        if console:
            print(f'\033[32m{time_info}\033[0m{log}')
        if file:
            print(f'{time_info}{log}', file=self.logs)

    @file_opener
    def put_space(self, print_console: bool = True) -> bool:
        """
        insert space in the logs
        :param print_console: if True you can see logs in the console as well
        :return: True or False
        """

        if self.logs.closed:
            self.logs = open(f'{self.save_path}/logs.txt', 'a')

        self.print_if_true(self.space, print_console)

        return True

    @file_opener
    def print_if_true(self, contents: str, print_console: bool) -> None:
        """
        if print_console true, print in the console as well
        :param contents: logs contents
        :param print_console: if True you can see logs in the console as well
        :return: None
        """

        if print_console:
            processingtools.functions.print_write(contents, self.logs)
        else:
            print(contents, file=self.logs)

    @staticmethod
    def arg2abs(args):
        """
        if args have path, convert absolute path
        :param args: input arguments
        :return: converted args
        """

        for attr in dir(args):
            value = getattr(args, attr)
            if type(value) == str and os.path.exists(value):
                setattr(args, attr, os.path.abspath(value))

        return args

    def log_dict(self, input_dict, keys: list = None, print_console: bool = True) -> True:
        """
        write dictionary in the logs file
        :param input_dict: input dictionary
        :param keys: key for dictionary
        :param print_console: if True you can see logs in the console as well
        :return: True
        """

        keys = input_dict.keys() if keys is None else keys
        self.print_if_true('{', print_console)

        for key in keys:
            self.print_if_true(f'    {key}: {input_dict[key]}', print_console)
        self.print_if_true('}', print_console)

        return True

    def record_default(self, comment: bool = True, os_key: list = None, gpu: bool = True, args=None) -> True:
        """
        write basic environment
        :param comment: True if you want to write comment
        :param os_key: insert key for recording os
        :param gpu: if True, record gpu information
        :param args: write args information if feed args
        :return:
        """

        if comment:
            self.print(f'Comments: {input("Leave comments for the logs file: ")}', console=False, time=False)
            self.put_space()
        if os_key is not None:
            self.record_os(os_key)
        if gpu:
            self.record_gpu()
        if args:
            args = self.arg2abs(args)
            self.record_arg(args)

        return True
