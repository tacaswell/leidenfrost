# Copyright 2013 Thomas A Caswell
# tcaswell@uchicago.edu
# http://jfi.uchicago.edu/~tcaswell
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import collections
import os.path
import copy


class FilePath(collections.namedtuple('FilePath',
                                      ['base_path', 'path', 'fname'])):
    '''
    Class for carrying around path information in what seemed like a
    convenient way when I wrote this.  The segmentation of the path into
    two parts is to make dealing with data on multiple external hard
    drives easier.  The idea is basically the same as chroot.
    '''
    __slots__ = ()

    @property
    def format(self):
        '''
        Formats the tuple -> string for handing to file operations
        '''
        return os.path.join(*self)

    @classmethod
    def from_db_dict(cls, in_dict, disk_dict):
        """
        Construct a FilePath object from a dict, converting the base_path with
        disk_dict

        Parameters
        ----------
        in_dict : dict
             a dict with the keys {'disk', 'path', 'fname'}
        disk_dict : dict
             A dict that maps disk number -> a path
        """
        in_dict = copy.copy(in_dict)
        base_path = disk_dict[in_dict.pop('disk')]
        return cls(base_path, **in_dict)

    def __new__(self, *args, **kwargs):
        args = tuple(a.decode('utf-8') if isinstance(a, bytes) else a
                     for a in args)
        for k, v in kwargs.items():
            if isinstance(v, bytes):
                kwargs[k] = v.decode('utf-8')
        return super().__new__(self, *args, **kwargs)


def convert_base_path(in_file_path, disk_dict):
    '''
    Returns a new FilePath object with the converted base_path.  Defaults
    to `base_path=''` if no mapping in disk_dict

    Parameters
    ----------
    in_file_path : FilePath
         the FilePath to return modified version of

    disk_dict : dict
        Dict that maps old value of base_path -> new values of base_path
    '''
    return in_file_path._replace(
        base_path=disk_dict.get(in_file_path.base_path, None))
