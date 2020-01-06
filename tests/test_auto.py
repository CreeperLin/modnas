import pytest
import os
import yaml
from functools import partial
import combo_nas
from combo_nas.utils.wrapper import run_search, run_augment, run_hptune
import testnet

class TestAuto():
    @staticmethod
    def create_testcases():
        root_dir = os.path.dirname(__file__)
        for root, dirs, files in os.walk(os.path.join(root_dir, 'config', 'auto'), topdown=True):
            for name in files:
                config_path = os.path.join(root, name)
                testname, ext = os.path.splitext(name)
                if ext != '.yaml': continue
                print(os.path.join(root, name), testname)
                def case_fn(casename, conf_path, *args, **kwargs):
                    exp = 'exp'
                    conf = yaml.load(open(conf_path, 'r'), Loader=yaml.Loader)
                    if 'search' in conf:
                        print('name: {} search'.format(casename))
                        gt_file = os.path.join(root_dir, 'genotype', 'search', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        else: print('gt_file: {}'.format(gt_file))
                        run_search(conf_path, casename, exp, None, None, genotype=gt_file)
                    if 'augment' in conf:
                        print('name: {} augment'.format(casename))
                        gt_file = os.path.join(root_dir, 'genotype', 'augment', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        else: print('gt_file: {}'.format(gt_file))
                        run_augment(conf_path, casename, exp, None, None, genotype=gt_file)
                    if 'hptune' in conf:
                        print('name: {} hptune'.format(casename))
                        run_hptune(conf_path, casename, exp, None, None, measure_fn=None)
                case_runner = partial(case_fn, testname, config_path)
                fn_name = 'test_'+testname
                case_runner.__name__ = fn_name
                setattr(TestAuto, fn_name, case_runner)

TestAuto.create_testcases()