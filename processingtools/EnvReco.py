import os


class EnvReco:
    """

    """
    def __init__(self, save_path: str):
        """

        """

        self.save_path = save_path

    def save_code(self):
        # todo
        return

    def present_env(self):
        # todo
        return

    def present_arg(self):
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
            print('this function is needed pytorch or tensorflow!')
            raise ImportError
