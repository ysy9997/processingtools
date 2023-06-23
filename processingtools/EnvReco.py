import os
import shutil
import json
import sys


class EnvReco:
    """

    """
    def __init__(self, save_path: str):
        """

        """

        self.save_path = save_path
        self.logs = open(f'{save_path}/logs.txt', 'a')

    def record_code(self, folder_name: str = 'snapshot') -> True:
        shutil.copytree('./', f'{self.save_path}/{folder_name}/')
        return True

    def record_env(self, args, _print: bool = True, save_file: str = None, save_type: str = 'txt') -> True:
        if save_type not in ['txt', 'text', 'json']:
            raise ValueError('save_type must be \'txt\' or \'text\' or \'json\'')

        if save_type == 'json':
            with open(f'{self.save_path}/args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        else:
            self.logs.write('\n'.join(sys.argv[1:]))

        # todo add save_file

        return True

    def record_arg(self):
        # todo
        return

    def print_gpu(self):
        try:
            import torch
            # todo
        except ImportError:
            import tensorflow as tf
            # todo
        except Exception:
            raise ImportError('this function is needed pytorch or tensorflow!')
