import json
import os

class ExpManager():
    def __init__(self, root_dir, name):
        self.root_dir = os.path.realpath(os.path.join(root_dir, name))
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
    
    def save_model(self, model, path):
        pass