from __future__ import print_function
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
import cine
from leidenfrost import FilePath
import shutil
from collections import defaultdict
import parse


def get_h5_lst(base_path, search_path):
    '''Recursively returns all h5 files below base_path/search_path'''
    h5names = []
    for dirpath, dirnames, fnames in os.walk(base_path + '/' + search_path):
        h5names.extend([FilePath(base_path, dirpath[len(base_path) + 1:], f)
                        for f in fnames if 'h5' in f])
    h5names.sort(key=lambda x: x[-1])

    return h5names


def get_cine_hashes(base_path, search_path):
    '''returs all paths and cine hash values under the search path'''
    cine_fnames = []
    for dirpath, dirnames, fnames in os.walk(base_path + '/' + search_path):
        cine_fnames.extend([FilePath(base_path, dirpath[len(base_path)+1:], f)
                        for f in fnames if 'cine' in f])
    cine_fnames.sort(key=lambda x: x[-1])
    cine_hash = [cine.Cine(cn.format).hash for cn in cine_fnames]

    return zip(cine_fnames, cine_hash)


def copy_file(fname_src, fname_dest):
    '''
    Copies file given by `fname_src` to `fname_dest`.

    makes sure all needed directories exist in between

    Does not copy if file exists at `fname_dest`
    Parameters
    ----------
    fname_src : FilePath
        source file

    fname_dest: FilePath
        destination file

    '''
    src_path = os.path.abspath(fname_src.format)
    dest_path = os.path.abspath(fname_dest.format)
    if (src_path == dest_path):
        raise Exception("can not buffer to self!!")
    ensure_path_exists(os.path.join(*fname_dest[:2]))
    if not os.path.exists(dest_path):
        shutil.copy2(src_path, dest_path)
    else:
        print("file exists, copy failed")


def ensure_path_exists(path):
    '''ensures that a given path exists, throws error if
    path points to a file'''
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isfile(path):
        raise Exception("there is a file where you think there is a path!")


def get_split_rm(base_path):
    res_dict = defaultdict(list)
    for dirpath, dirnames, fnames in os.walk(base_path):
        for ff in fnames:
            if ff[:2] == 'RM':
                rr = parse.parse('RM_{key}_{start_f:d}-{end_f:d}_{name}.h5', ff)

                res_dict[rr['key']].append((rr['start_f'], rr['end_f'],
                                              FilePath(base_path,
                                              dirpath[len(base_path) + 1:]
                                              , ff)))

    for k, v in res_dict.items():
        v.sort()

    return res_dict
