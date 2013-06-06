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

from __future__ import division

import os

# do this to make me learn where stuff is and to make it easy to
# switch to PyQt later
import PySide.QtCore as QtCore
import PySide.QtGui as QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from common import directory_selector

from collections import defaultdict

import numpy as np

import leidenfrost.infra as infra
import leidenfrost.backends as backends


class LFReaderWriter(QtCore.QObject):

    frame_loaded = QtCore.Signal(bool, bool)
    file_loaded = QtCore.Signal(bool, bool)
    file_params = QtCore.Signal(dict)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.hdf_backend = None
        self.process_backend = None

        self.mbe = None
        self.cur_frame = -1

    @QtCore.Slot(int)
    def read_frame(self, ind):
        if self.hdf_backend is not None:
            if self.cur_frame != ind:
                self.cur_frame = ind

                if self.hdf_backend.contains_frame(ind):
                    self.mbe = self.hdf_backend.get_frame(ind,
                                                      raw=True,
                                                      get_img=True)
                    self.frame_loaded.emit(True, True)
                else:
                    self.mbe = None
                    self.frame_loaded.emit(False, True)




    @QtCore.Slot(int, infra.SplineCurve)
    def proc_frame(self, ind, curve):

        if self.process_backend is not None:
            self.mbe, self.next_curve = self.process_backend.process_frame(ind,
                                                                           curve)
            self.frame_loaded.emit(True, True)

    @QtCore.slot()
    def get_mbe(self):
        return self.mbe

    @QtCore.slot(int, tuple, dict)
    def get_img(self, ind, *args, **kwargs):
        if self.process_backend is not None:
            tmp = self.process_backend.get_image(ind, *args, **kwargs)
            return np.asarray(tmp, dtype=np.float)
        return None

    @QtCore.slot()
    def clear(self):
        self.mbe = None
        del self.hdf_backend
        self.hdf_backend = None
        self.process_backend = None
        self.next_curve = None

    def __len__(self):
        if self.cine_backend:
            return len(self.cine_backend)
        else:
            return 0

    @QtCore.slot()
    def save_frame(self):
        if self.mbe is None:
            print 'no frame to save'
            return
        if self.contains_frame(self.cur_frame):
            print 'current frame exists in file, not saving'
        self.mbe.write_to_hdf(self.hdf_backend.file)

    @QtCore.Slot(backends.FilePath, backends.FilePath, dict, dict)
    def reset_fnames(self,
                    hdf_fname, cine_fname,
                    params,
                    kwargs):

        # clear
        self.clear()
        # set up the hdf backend
        self.backend = backends.HdfBackend(hdf_fname,
                                           cine_base_path=cine_fname.basepath,
                                           **kwargs)

        # set up the process backend
        self.process_backend = backends.ProcessBackend.from_args(cine_fname,
                                                                 bck_img=None,
                                                                 **params)
        # emit a whole bunch of stuff
        self.file_loaded.emit(True, True)
        self.file_params.emit(self.backend.proc_prams)
        self.read_frame(0)
