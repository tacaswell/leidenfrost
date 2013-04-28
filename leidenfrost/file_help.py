#Copyright 2013 Thomas A Caswell
#tcaswell@uchicago.edu
#http://jfi.uchicago.edu/~tcaswell
#
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
#
#This program is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, see <http://www.gnu.org/licenses>.

import os
from leidenfrost import FilePath


def get_h5_lst(base_path, search_path):
    '''Recursively returns all h5 files below base_path/search_path'''
    h5names = []
    for dirpath, dirnames, fnames in os.walk(base_path + '/' + search_path):
        h5names.extend([FilePath(base_path, dirpath[len(base_path) + 1:], f) for f in fnames if 'h5' in f])
    h5names.sort(key=lambda x: x[-1])

    return h5names
