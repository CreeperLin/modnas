import os
import time


class ExpManager():
    def __init__(self, root_dir, name, subdir_timefmt='%Y-%m-%d_%H-%M'):
        if subdir_timefmt is None:
            root_dir = os.path.join(root_dir, name)
        else:
            root_dir = os.path.join(root_dir, name, time.strftime(subdir_timefmt, time.localtime()))
        self.root_dir = os.path.realpath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def subdir(self, *args):
        subdir = os.path.join(self.root_dir, *args)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def join(self, *args):
        return os.path.join(self.subdir(*args[:-1]), args[-1])

    @property
    def logs_path(self):
        return self.subdir('logs')

    @property
    def writer_path(self):
        return self.subdir('writer')

    @property
    def save_path(self):
        return self.subdir('chkpt')

    @property
    def output_path(self):
        return self.subdir('output')

    @property
    def plot_path(self):
        return self.subdir('plot')

    @property
    def config_path(self):
        return self.join('config.yaml')
