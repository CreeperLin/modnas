#!/usr/bin/env python
import os
from setuptools import setup, find_packages
root_dir = os.path.abspath(os.path.dirname(__file__))
readme = open(os.path.join(root_dir, 'README.md')).read()
requirements = [name.rstrip() for name in open(os.path.join(root_dir, 'requirements.txt')).readlines()]
VERSION = '0.0.1'

setup(
    name = 'combo_nas',
    version = VERSION,
    author = 'CreeperLin',
    author_email = 'linyunfeng@sjtu.edu.cn',
    url = 'https://github.com/CreeperLin/combo_nas',
    description = 'combo_nas: a nas framework for rapid experiment, development and tuning',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    license = license,
    packages = find_packages(exclude=('test')),
    install_requires = requirements,
    classifiers = [
        'Programming Language :: Python :: 3',
    ],
)
