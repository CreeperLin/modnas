import sys
import time
from functools import wraps
import numpy as np
from .. import backend

ttable = {}
mtable = {}


def get_cputime():
    return time.perf_counter() * 1000


def seqstat(arr):
    a = np.array(arr[min(len(arr) - 1, 1):])
    return '[ {:.3f}({:.3f}) / {:.3f} / {:.3f} / {:.3f} ]'.format(np.mean(a), np.mean(a[len(a) // 2:]), np.min(a), np.max(a),
                                                                  np.std(a))


t0 = 0


def report_time(msg=''):
    global t0
    t1 = get_cputime()
    fr = sys._getframe(1)
    lat = t1 - t0
    if msg in ttable:
        ttable[msg].append(lat)
    else:
        ttable[msg] = [lat]
    print("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} / {} ms".format(msg.center(20, ' '), fr.f_code.co_name, fr.f_lineno, t1,
                                                                    lat, seqstat(ttable[msg])))
    t0 = get_cputime()


m0 = 0


def report_mem(msg=''):
    global m0
    m1 = backend.get_dev_mem_used()
    fr = sys._getframe(1)
    fp = m1 - m0
    if msg in mtable:
        mtable[msg].append(fp)
    else:
        mtable[msg] = [fp]
    print("GPU Mem: {} {} @ {} : {:.3f} {:.3f} dt: {:.3f} MB".format(msg.center(20, ' '), fr.f_code.co_name, fr.f_lineno, m1,
                                                                     fp, seqstat(mtable[msg])))
    m0 = m1


def profile_mem(function):
    @wraps(function)
    def gpu_mem_profiler(*args, **kwargs):
        m1 = backend.get_dev_mem_used()
        result = function(*args, **kwargs)
        m2 = backend.get_dev_mem_used()
        fp = m2 - m1
        fname = function.__name__
        if fname in mtable:
            mtable[fname].append(fp)
        else:
            mtable[fname] = [fp]
        print("GPU Mem: {}: {:.3f} / {:.3f} / {:.3f} / {} MB".format(fname.center(20, ' '), m1, m2, fp,
                                                                     seqstat(mtable[fname])))
        return result

    return gpu_mem_profiler


def profile_time(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t1 = get_cputime()
        result = function(*args, **kwargs)
        t2 = get_cputime()
        lat = t2 - t1
        fname = function.__name__
        if fname in ttable:
            ttable[fname].append(lat)
        else:
            ttable[fname] = [lat]
        print("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} / {} ms".format(fname.center(20, ' '), t1, t2, lat,
                                                                      seqstat(ttable[fname])))
        return result

    return function_timer


class TimeProfiler(object):
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.table = {}
        self.acc_table = {}

    def reset(self):
        self.table.clear()
        self.acc_table.clear()

    def timer_start(self, iid):
        if not self.enabled:
            return
        tic = get_cputime()
        if iid not in self.table:
            self.table[iid] = np.array([-tic])
        else:
            arr = self.table[iid]
            self.table[iid] = np.append(arr, -tic)

    def timer_stop(self, iid):
        if not self.enabled:
            return
        t1 = get_cputime()
        self.table[iid][-1] += t1

    def stat(self, iid, logger=None):
        if not self.enabled or iid not in self.table:
            return
        arr = self.table[iid]
        msg = 'Time {}: {} / {:.3f} / {} ms'.format(iid.center(10, ' '), len(arr), arr[-1], seqstat(arr))
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    def stat_all(self, logger=None):
        for i in self.table:
            self.stat(i, logger)

    def begin_acc_item(self, cid):
        if cid not in self.acc_table:
            self.acc_table[cid] = np.array([0.])
        else:
            arr = self.acc_table[cid]
            self.acc_table[cid] = np.append(arr, [0.])

    def add_acc_item(self, cid, iid):
        arr = self.acc_table[cid]
        item = self.table[iid]
        arr[-1] += item[-1]

    def clear_acc_item(self, cid):
        arr = self.acc_table[cid]
        arr.clear()

    def stat_acc(self, cid, logger=None):
        if cid not in self.acc_table:
            return
        arr = self.acc_table[cid]
        msg = 'AccTime {}: {} / {:.3f} / {} ms'.format(cid.center(10, ' '), len(arr), arr[-1], seqstat(arr))
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    def avg(self, iid):
        return 0 if iid not in self.table else np.mean(self.table[iid])
