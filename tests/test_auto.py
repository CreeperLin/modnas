import pytest
import os
import yaml
from functools import partial
from combo_nas.utils.wrapper import run_search, run_augment, run_hptune, run_pipeline
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
                    best_gt = None
                    if 'search' in conf:
                        gt_file = os.path.join(root_dir, 'genotype', 'search', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        ret = run_search(conf_path, casename, exp, None, None, genotype=gt_file)
                        best_gt = ret.get('best_gt', None)
                    if 'augment' in conf:
                        gt_file = os.path.join(root_dir, 'genotype', 'augment', casename+'.gt')
                        if not os.path.isfile(gt_file):
                            gt_file = None
                        run_augment(conf_path, casename, exp, None, None, genotype=gt_file)
                        if not best_gt is None:
                            run_augment(conf_path, casename, exp, None, None, genotype=best_gt)
                    if 'hptune' in conf:
                        run_hptune(conf_path, casename, exp, None, None, measure_fn=None)
                    if 'pipeline' in conf:
                        run_pipeline(conf_path, casename, exp)
                case_runner = partial(case_fn, testname, config_path)
                fn_name = 'test_'+testname
                case_runner.__name__ = fn_name
                setattr(TestAuto, fn_name, case_runner)

TestAuto.create_testcases()