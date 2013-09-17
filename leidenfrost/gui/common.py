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
        pass

    @QtCore.Slot()
    def select_path(self):
        path = QtGui.QFileDialog.getExistingDirectory(self,
                                                      caption=self.cap,
                                                      dir=None)

        if len(path) > 0:
            self.label.setText(path)
            self.selected.emit(path)
            return path
        else:
            path = None
        self.path = path
        return path

    @property
    def path(self):
        return self.label.text()


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
