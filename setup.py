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
    packages=["leidenfrost"],
    install_requires=['numpy', 'six', 'scipy', 'trackpy']
    )
