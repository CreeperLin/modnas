import pytest
import os
import yaml
from functools import partial
import combo_nas
from combo_nas.utils.routine import search
from combo_nas.utils.wrapper import init_all_search
from combo_nas.utils.routine import augment
from combo_nas.utils.wrapper import init_all_augment
import testnet

class TestAuto():
    @staticmethod
    def create_testcases():
        root_dir = os.path.dirname(__file__)
        for root, dirs, files in os.walk(os.path.join(root_dir, 'config'), topdown=True):
            for name in files:
                config_path = os.path.join(root, name)
                testname, ext = os.path.splitext(name)
                if ext != '.yaml': continue
                print(os.path.join(root, name), testname)
                def case_fn(casename, conf_path, *args, **kwargs):
                    exp_root_dir = 'exp'
                    conf = yaml.load(open(conf_path, 'r'), Loader=yaml.Loader)
                    if 'search' in conf:
                        print('name: {} search'.format(casename))
                        gt_file = os.path.join(root_dir, 'genotype', 'search', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        else: print('gt_file: {}'.format(gt_file))
                        search_kwargs = init_all_search(conf_path, casename, exp_root_dir, None, None, genotype=gt_file)
                        best_top1, best_genotype, genotypes = search(**search_kwargs)
                    if 'augment' in conf:
                        print('name: {} augment'.format(casename))
                        gt_file = os.path.join(root_dir, 'genotype', 'augment', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        else: print('gt_file: {}'.format(gt_file))
                        augment_kwargs = init_all_augment(conf_path, casename, exp_root_dir, None, None, genotype=gt_file)
                        best_top1 = augment(**augment_kwargs)
                case_runner = partial(case_fn, testname, config_path)
                fn_name = 'test_'+testname
                case_runner.__name__ = fn_name
                setattr(TestAuto, fn_name, case_runner)

TestAuto.create_testcases()