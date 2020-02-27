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

    def exp_subdir(self, name):
        subdir = os.path.join(self.root_dir, name)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def join(self, subdir, filename):
        return os.path.join(self.exp_subdir(subdir), filename)

    @property
    def logs_path(self):
        return self.exp_subdir('logs')

    @property
    def writer_path(self):
        return self.exp_subdir('writer')

    @property
    def save_path(self):
        return self.exp_subdir('chkpt')

    @property
    def output_path(self):
        return self.exp_subdir('output')

    @property
    def plot_path(self):
        return self.exp_subdir('plot')

    @property
    def config_path(self):
        return self.join('', 'config.yaml')
