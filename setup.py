#!/usr/bin/env python

import setuptools
from distutils.core import setup

print("Installing leidenfrost")

setup(
    name='leidenfrost',
    version='0.9',
    author='Thomas A Caswell',
    author_email='tcaswell@uchicago.edu',
    url='https://github.com/tacaswell/leidenfrost',
    packages=["leidenfrost", "leidenfrost.gui", 'cine',
              'find_peaks'],
    requires=['numpy', 'six', 'scipy', 'pymongo', 'h5py',
              'networkx', 'parse', 'ipython', 'pyzmq', 'pyside'],
    )
