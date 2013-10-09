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

# do this to make me learn where stuff is and to make it easy to
# switch to PyQt later
import PySide.QtCore as QtCore
import PySide.QtGui as QtGui
import os.path
from itertools import izip

from leidenfrost import FilePath


class directory_selector(QtGui.QWidget):
    '''
    A widget class deal with selecting and displaying path names
    '''

    selected = QtCore.Signal(str)

    def __init__(self, caption, path='', parent=None):
        QtGui.QWidget.__init__(self, parent)
        print caption
        self.cap = caption

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QtGui.QLabel(caption))

        hlayout = QtGui.QHBoxLayout()
        layout.addLayout(hlayout)

        self.label = QtGui.QLabel(path)
        hlayout.addWidget(self.label)
        hlayout.addStretch()
        button = QtGui.QPushButton('')
        button.setIcon(QtGui.QIcon.fromTheme('folder'))
        button.clicked.connect(self.select_path)
        hlayout.addWidget(button)

    @QtCore.Slot(str)
    def set_path(self, path):
        if os.path.isdir(path):
            self.label.setText(path)
            self.selected.emit(path)
        else:
            raise Exception("path does not exst")

    @QtCore.Slot()
    def select_path(self):
        cur_path = self.path
        if len(cur_path) == 0:
            cur_path = None
        path = QtGui.QFileDialog.getExistingDirectory(self,
                                                      caption=self.cap,
                                                      dir=cur_path)

        if len(path) > 0:
            self.path = path
            return path
        else:
            path = None
        return path

    @property
    def path(self):
        return self.label.text()

    @path.setter
    def path(self, in_path):
        self.set_path(in_path)


class numbered_paths(QtGui.QWidget):
    def __init__(self, N, parent=None):
        """
        Parameters
        ----------
        N : int
           Initial number of path to keep track of
        """
        QtGui.QWidget.__init__(self, parent)
        self.path_widgets = []
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        for j in xrange(N):
            tmp = directory_selector('Disk {}'.format(j))
            self.path_widgets.append(tmp)
            layout.addWidget(tmp)

    @property
    def path_dict(self):
        return {j: path for j, path in enumerate(fs.path for fs in self.path_widgets)}


class base_path_path_selector(QtGui.QWidget):
    def __init__(self, caption, parent=None):
        """
        Parameters
        ----------
        N : int
           Initial number of path to keep track of
        """
        QtGui.QWidget.__init__(self, parent)
        self.base_widget = directory_selector('{} base path'.format(caption))
        self.path_widget = directory_selector('{} path'.format(caption))
        self.path_widget.selected.connect(self.validate_path)

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.base_widget)
        layout.addWidget(self.path_widget)

    @property
    def base_path(self):
        return self.base_widget.path

    @property
    def path(self):
        return self.path_widget.path

    @property
    def path_template(self):
        bp = self.base_path
        return FilePath(bp, self.path[len(bp)+1:], '')

    @QtCore.Slot()
    def validate_path(self, in_path):
        bp = self.base_path
        if in_path[:len(bp)] != bp:
            self.path_widget.path = bp


class frame_range_selector(QtGui.QWidget):
    frame_range = QtCore.Signal(int, int)
    updated = QtCore.Signal()

    def __init__(self, spinner, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self._spinner = spinner
        self._start = 0
        self._end = 1
        # global layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        # top layer of labels
        top_layer = QtGui.QHBoxLayout()
        self._end_lab = QtGui.QLabel('-')
        self._start_lab = QtGui.QLabel('-')
        top_layer.addWidget(self._start_lab)
        top_layer.addWidget(self._end_lab)

        # in/out buttons
        mid_layer = QtGui.QHBoxLayout()

        start_button = QtGui.QPushButton('in')
        start_button.clicked.connect(self._set_start)
        mid_layer.addWidget(start_button)

        end_button = QtGui.QPushButton('out')
        end_button.clicked.connect(self._set_end)
        mid_layer.addWidget(end_button)

        # submit button
        submit = QtGui.QPushButton('submit')
        submit.clicked.connect(
            lambda: self.frame_range.emit(self._start, self._end))
        layout.addLayout(top_layer)
        layout.addLayout(mid_layer)
        layout.addWidget(submit)
        self.updated.connect(self._update)

    def _set_end(self):
        trial_val = self._spinner.value() + 1
        if trial_val < self._start:
            return
        self._end = trial_val
        self.updated.emit()

    def _set_start(self):
        trial_val = self._spinner.value()
        if trial_val >= self._end:
            return
        self._start = trial_val
        self.updated.emit()

    @QtCore.Slot(int, int)
    def set_limit(self, in_val, out_val):
        self._start = in_val
        self._end = out_val
        self.updated.emit()

    @QtCore.Slot()
    def _update(self):
        for val, lab in izip((self._start, self._end), (self._start_lab, self._end_lab)):
            if val is None:
                lab.setText('-')
            else:
                lab.setNum(val)
