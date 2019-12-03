# -*- coding: utf-8 -*-
import sys
import time
import torch
from functools import wraps
import numpy as np

ttable = {}
mtable = {}

def get_gpumem():
    return torch.cuda.memory_allocated() / 1024. / 1024.

def get_cputime():
    return time.perf_counter() * 1000

def seqstat(arr):
    a = np.array(arr[min(len(arr)-1,1):])
    return '[ {:.3f}({:.3f}) / {:.3f} / {:.3f} / {:.3f} ]'.format(
        np.mean(a), np.mean(a[len(a)//2:]), np.min(a), np.max(a), np.std(a))

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
    print("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} / {} ms".format(
        msg.center(20,' '), fr.f_code.co_name, fr.f_lineno, t1, lat, seqstat(ttable[msg])))
    t0 = get_cputime()

m0 = 0
def report_mem(msg=''):
    global m0
    m1 = get_gpumem()
    fr = sys._getframe(1)
    fp = m1 - m0
    if msg in mtable:
        mtable[msg].append(fp)
    else:
        mtable[msg] = [fp]
    print("GPU Mem: {} {} @ {} : {:.3f} dt: {:.3f} MB".format(
        msg.center(20,' '), fr.f_code.co_name, fr.f_lineno, m1, fp, seqstat(mtable[msg])))
    m0 = m1

def profile_mem(function):
    @wraps(function)
    def gpu_mem_profiler(*args, **kwargs):
        m1 = get_gpumem()
        result = function(*args, **kwargs)
        m2 = get_gpumem()
        fp = m2 - m1
        fname = function.__name__
        if fname in mtable:
            mtable[fname].append(fp)
        else:
            mtable[fname] = [fp]
        print("GPU Mem: {}: {:.3f} / {:.3f} / {:.3f} / {} MB".format(
            fname.center(20,' '), m1, m2, fp, seqstat(mtable[fname])))
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
        print("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} / {} ms".format(
            fname.center(20,' '), t1, t2, lat, seqstat(ttable[fname])))
        return result
    return function_timer

class profile_ctx():
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.m0 = get_gpumem()
        self.t0 = get_cputime()
    
    def __exit__(self):
        self.t1 = get_cputime()
        self.m1 = get_gpumem()
        lat = self.t1 - self.t0
        mem = self.m1 - self.m0
        fname = self.name
        if fname in ttable:
            ttable[fname].append(lat)
        else:
            ttable[fname] = [lat]
        if fname in mtable:
            mtable[fname].append(fp)
        else:
            mtable[fname] = [fp]
        print("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} / {} ms".format(
            fname.center(20,' '), self.t0, self.t1, lat, seqstat(ttable[fname])))
        print("GPU Mem: {}: {:.3f} / {:.3f} / {:.3f} / {} MB".format(
            fname.center(20,' '), self.m0, self.m1, mem, seqstat(mtable[fname])))
    
    def report(self):
        t1 = get_cputime()
        fr = sys._getframe(1)
        print("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} ms".format(
        self.name.center(20,' '), fr.f_code.co_name, fr.f_lineno, t1, t1 - self.t0))
        m1 = get_gpumem()
        print("GPU Mem: {} {} @ {} : {:.3f} dt: {:.3f} MB".format(
        self.name.center(20,' '), fr.f_code.co_name, fr.f_lineno, m1, m1 - self.m0))


class TimeProfiler(object):
    def __init__(self):
        self.table = {}
        self.acc_table = {}
        self.offset = 0
        self.timer_start('ofs')
        self.timer_stop('ofs')
        self.offset = self.table['ofs'][0]
    
    def timer_start(self, id):
        t0 = get_cputime()
        if not id in self.table:
            self.table[id] = np.array([-t0])
        else:
            arr = self.table[id]
            self.table[id] = np.append(arr, -t0)
    
    def timer_stop(self, id):
        t1 = get_cputime()
        self.table[id][-1] += t1 - self.offset

    def print_stat(self, id):
        if not id in self.table: return
        arr = self.table[id]
        print('Time {}: {} / {:.3f} / {} ms'.format(
            id.center(10,' '), len(arr), arr[-1], seqstat(arr)))
    
    def stat_all(self):
        for i in self.table:
            self.print_stat(i)
    
    def begin_acc_item(self, cid):
        if not cid in self.acc_table:
            self.acc_table[cid] = np.array([0.])
        else:
            arr = self.acc_table[cid]
            self.acc_table[cid] = np.append(arr, [0.])

    def add_acc_item(self, cid, id):
        arr = self.acc_table[cid]
        item = self.table[id]
        arr[-1] += item[-1]

    def clear_acc_item(self, cid):
        arr = self.acc_table[cid]
        arr.clear()

    def stat_acc(self, cid):
        if not cid in self.acc_table: return
        arr = self.acc_table[cid]
        print('AccTime {}: {} / {:.3f} / {} ms'.format(
            cid.center(10,' '), len(arr), arr[-1], seqstat(arr)))

    def avg(self, id):
        return 0 if not id in self.table else np.mean(self.table[id])

tprof = TimeProfiler()