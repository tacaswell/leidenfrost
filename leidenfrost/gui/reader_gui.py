#Copyright 2012 Thomas A Caswell
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

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from collections import defaultdict

import numpy as np

import leidenfrost.infra as infra
import leidenfrost.backends as backends

from common import directory_selector


class LFReader(QtCore.QObject):
    frame_loaded = QtCore.Signal(bool, bool)
    file_loaded = QtCore.Signal(bool, bool)
    file_params = QtCore.Signal(dict)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.backend = None

        self.mbe = None
        self.cur_frame = None

    @QtCore.Slot(int)
    def read_frame(self, ind):
        if self.backend is not None:
            if self.cur_frame != ind:
                self.mbe = self.backend.get_frame(ind,
                                                  raw=True,
                                                  get_img=True)
                self.cur_frame = ind

            self.frame_loaded.emit(True, True)

    def get_mbe(self):
        return self.mbe

    def clear(self):
        self.mbe = None
        self.backend = None

    def __len__(self):
        if self.backend:
            return len(self.backend)
        else:
            return 0

    @QtCore.Slot(backends.FilePath, str, dict)
    def set_fname(self, fname, cbp, kwargs):
        print 'entered set_fname'
        self.clear()

        print fname, cbp, kwargs
        self.backend = backends.HdfBackend(fname,
                                           cine_base_path=cbp,
                                           **kwargs)
        self.mbe = self.backend.get_frame(0,
                                          raw=True,
                                          get_img=True)
        self.cur_frame = 0

        self.file_loaded.emit(True, True)
        self.file_params.emit(self.backend.proc_prams)


class LFReaderGui(QtGui.QMainWindow):
    read_request_sig = QtCore.Signal(int)
    open_file_sig = QtCore.Signal(backends.FilePath, str, dict)
    kill_thread = QtCore.Signal()
    redraw_sig = QtCore.Signal(bool, bool)
    draw_done_sig = QtCore.Signal()
    cap_lst = ['hdf base path',
               'cine base path',
               'cine cache path',
               'hdf cache path']

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Fringe Display')

        self.draw_fringes = False
        self.all_fringes_flg = False

        self.reader = LFReader()
        self.thread = QtCore.QThread(parent=self)
        self.reader.moveToThread(self.thread)

        self.bin_search = None

        self.fringe_lines = []

        self.paths_dict = defaultdict(lambda: None)

        self.create_main_frame()
        self.create_actions()
        self.create_menu_bar()
        self.create_diag()
        self.create_status_bar()

        self.read_request_sig.connect(self.reader.read_frame)
        self.kill_thread.connect(self.thread.quit)
        self.open_file_sig.connect(self.reader.set_fname)
        self.redraw_sig.connect(self.on_draw)
        self.draw_done_sig.connect(self._play)

        self.reader.frame_loaded.connect(self.on_draw)
        self.reader.file_loaded.connect(self.redraw)
        self.reader.file_params.connect(self.update_param_labels)
        self.show()

        self.thread.start()

        self.label_block = None

        QtGui.qApp.exec_()

    def set_fringes_visible(self, i):
        self.draw_fringes = bool(i)

        self.redraw_sig.emit(True, False)

    def set_all_friges(self, i):
        self.all_fringes_flg = bool(i)
        self.redraw_sig.emit(True, False)

    @QtCore.Slot(bool, bool)
    def on_draw(self, refresh_lines=True, refresh_img=True):
        """ Redraws the figure
        """
        mbe = self.reader.get_mbe()
        if mbe is None:
            #            self.clear()
            return

        if refresh_lines:
            # clear the lines we have
            for ln in self.fringe_lines:
                ln.remove()
            # nuke all those objects
            self.fringe_lines = []

            if self.draw_fringes:
                # if we should draw new ones, do so
                # grab new mbe from thread object
                # grab new next curve

                if self.draw_fringes and mbe is not None:
                    self.fringe_lines.extend(
                        mbe.ax_plot_tracks(self.axes,
                                           min_len=0,
                                           all_tracks=self.all_fringes_flg)
                    )
                    self.fringe_lines.extend(
                        mbe.ax_draw_center_curves(self.axes))

        if refresh_img:
            img = mbe.img
            if img is not None and self.im is not None:
                self.im.set_data(img)

        self.canvas.draw()
        self.draw_done_sig.emit()
        pass

    @QtCore.Slot(bool, bool)
    def redraw(self, draw_img, draw_tracks):
        if self.im is not None:
            self.im.remove()

        # clear the lines we have
        for ln in self.fringe_lines:
            ln.remove()

        self.axes = None
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        #       self.fig.tight_layout(.3, None, None, None)
        # nuke all those objects
        self.fringe_lines = []
        self.im = None
        mbe = self.reader.get_mbe()

        img = mbe.img
        if img is None:
            img = 1.5 * np.ones((1, 1))

        self.im = self.axes.imshow(img,
                                   cmap='gray',
                                   origin='lower',
                                   interpolation='nearest')
        self.im.set_clim([.5, 1.5])
        self.axes.set_aspect('equal')

        if self.draw_fringes:
            # if we should draw new ones, do so
            # grab new mbe from thread object
            # grab new next curve

            if self.draw_fringes and mbe is not None:
                self.fringe_lines.extend(
                    mbe.ax_plot_tracks(self.axes,
                                       min_len=0,
                                       all_tracks=self.all_fringes_flg)
                )

        #self.status_text.setText(label)

        self.frame_spinner.setRange(0, len(self.reader) - 1)
        self.frame_spinner.setValue(0)
        self.max_frame_label.setText(str(len(self.reader) - 1))
        self.redraw_sig.emit(False, False)

    @QtCore.Slot(int)
    def set_cur_frame(self, i):
        self.read_request_sig.emit(i)

    def show_cntrls(self):
        self.diag.show()

    def open_file(self):

        if self.paths_dict['hdf base path'] is None:
            self.directory_actions['hdf base path'].trigger()
        hdf_bp = self.paths_dict['hdf base path']

        if self.paths_dict['cine base path'] is not None:
            cine_bp = self.paths_dict['cine base path']
        else:
            cine_bp = hdf_bp

        fname, _ = QtGui.QFileDialog.getOpenFileName(self,
                                                     caption='Select File',
                                                     filter='hdf (*.h5 *.hdf)')
        if len(fname) == 0:
            return

        while not hdf_bp == fname[:len(hdf_bp)]:
            print 'please set base_dir'
            self.directory_actions['hdf base path'].trigger()
            hdf_bp = self.paths_dict['hdf base path']

        path_, fname_ = os.path.split(fname[(len(hdf_bp) + 1):])
        new_hdf_fname = backends.FilePath(hdf_bp, path_, fname_)

        tmp_dict = {'cine_cache_dir': self.paths_dict['cine cache path'],
                    'hdf_cache_dir': self.paths_dict['hdf cache path']}

        self.fname_text.setText(fname_)

        self.open_file_sig.emit(new_hdf_fname, cine_bp, tmp_dict)

    @QtCore.Slot(dict)
    def update_param_labels(self, prams):
        diag_layout = self.diag.widget().layout()
        if self.label_block is not None:
            diag_layout.removeWidget(self.label_block)
            self.label_block.setVisible(False)

        param_form_layout = QtGui.QFormLayout()
        ignore_lst = ['tck0', 'tck1', 'tck2', 'center',
                      'cine_path', 'cine_fname', 'cine_hash']
        for k, v in prams.iteritems():
            if k in ignore_lst:
                continue
            param_form_layout.addRow(QtGui.QLabel(k + ':'), QtGui.QLabel(str(v)))

        def print_parameters():
            '''
            Print the parameters out to stdout in a form that will play nice
            with org-mode

            '''
            ignore_lst = ['tck0', 'tck1', 'tck2', 'center']
            for k, v in prams.iteritems():
                if k in ignore_lst:
                    continue
                print "| {key} | {val} |".format(key=k, val=v)

        print_button = QtGui.QPushButton('Print')
        print_button.pressed.connect(print_parameters)

        self.label_block = QtGui.QGroupBox("Parameters")
        lb_layout = QtGui.QVBoxLayout()
        lb_layout.addLayout(param_form_layout)
        lb_layout.addWidget(print_button)
        self.label_block.setLayout(lb_layout)
        diag_layout.addWidget(self.label_block)

    def create_diag(self):

        # frame number lives on top
        self.frame_spinner = QtGui.QSpinBox()
        self.frame_spinner.setRange(0, len(self.reader) - 1)
        self.frame_spinner.valueChanged.connect(self.set_cur_frame)
        self.frame_spinner.setWrapping(True)
        frame_selector_group = QtGui.QVBoxLayout()
        fs_form = QtGui.QHBoxLayout()
        fs_form.addWidget(QtGui.QLabel('frame #'))
        fs_form.addWidget(self.frame_spinner)
        fs_form.addWidget(QtGui.QLabel(' of '))
        self.max_frame_label = QtGui.QLabel(str(len(self.reader) - 1))
        fs_form.addWidget(self.max_frame_label)
        fs_stepbox = QtGui.QGroupBox("Frame step")
        fs_sb_rb = QtGui.QHBoxLayout()
        for j in [1, 10, 100, 1000, 10000]:
            tmp_rdo = QtGui.QRadioButton(str(j))
            tmp_rdo.toggled.connect(lambda x, j=j: self.frame_spinner.setSingleStep(j) if x else None)
            fs_sb_rb.addWidget(tmp_rdo)
            if j == 1:
                tmp_rdo.toggle()
            pass
        fs_stepbox.setLayout(fs_sb_rb)
        frame_selector_group.addLayout(fs_form)
        frame_selector_group.addWidget(fs_stepbox)

        # box for setting if the fringes should be drawn
        self.fringe_grp_bx = QtGui.QGroupBox("Draw Fringes")
        self.fringe_grp_bx.setCheckable(True)
        self.fringe_grp_bx.setChecked(self.draw_fringes)
        self.fringe_grp_bx.toggled.connect(self.set_fringes_acc.setChecked)
        self.set_fringes_acc.toggled.connect(self.fringe_grp_bx.setChecked)
        all_fringe_rb = QtGui.QRadioButton('All Fringes')
        valid_fringes_rb = QtGui.QRadioButton('Valid Fringes')
        rb_vbox = QtGui.QVBoxLayout()
        rb_vbox.addWidget(valid_fringes_rb)
        rb_vbox.addWidget(all_fringe_rb)
        self.fringe_grp_bx.setLayout(rb_vbox)
        all_fringe_rb.toggled.connect(self.set_all_fringes_acc.setChecked)
        all_fringe_rb.setChecked(False)
        valid_fringes_rb.setChecked(True)

        def rb_sync(flg):
            if flg:
                all_fringe_rb.setChecked(True)
            else:
                valid_fringes_rb.setChecked(True)
        self.set_all_fringes_acc.toggled.connect(rb_sync)

        # box for path information
        path_box = QtGui.QGroupBox("paths")
        pb_layout = QtGui.QVBoxLayout()
        path_box.setLayout(pb_layout)
        for c in self.cap_lst:
            ds = directory_selector(caption=c)
            pb_layout.addWidget(ds)
            self.directory_actions[c].triggered.connect(ds.select_path)
            ds.selected.connect(lambda x, c=c: self.paths_dict.__setitem__(c, x))

        # set up over-all layout
        self.diag = QtGui.QDockWidget('controls', parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.diag)
        diag_widget = QtGui.QWidget(self.diag)
        self.diag.setWidget(diag_widget)
        diag_layout = QtGui.QVBoxLayout()
        diag_widget.setLayout(diag_layout)

        # play button
        play_button = QtGui.QPushButton('Play')
        self.play_button = play_button
        play_button.setCheckable(True)
        self.play_button.pressed.connect(self.frame_spinner.stepUp)
        # track pickers
        self.graphs_window = GraphDialog((2, 1))
        self.show_track_graphs.toggled.connect(self.graphs_window.setVisible)

        # add everything to the layout
        diag_layout.addLayout(frame_selector_group)
        diag_layout.addWidget(self.fringe_grp_bx)
        diag_layout.addWidget(path_box)
        diag_layout.addStretch()
        diag_layout.addWidget(play_button)

    @QtCore.Slot()
    def _play(self):
        if self.play_button.isChecked():
            QtCore.QTimer.singleShot(30, self.frame_spinner.stepUp)

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        # create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #

        self.fig = Figure((24, 24))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        def track_plot(axes_lst, trk):
            q, phi, v = zip(*[(p.q, p.phi, p.v) for p in trk.points])

            axes_lst[0].cla()
            axes_lst[0].plot(q, phi)

            axes_lst[1].cla()
            axes_lst[1].plot(v)

        def picker_fun(trk):
            if isinstance(trk, infra.lf_Track):
                if self.graphs_window:
                    self.graphs_window.update_axes(track_plot, trk)

            else:
                print type(trk)

        self.picker = PickerHandler(self.canvas, picker_fun)
        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        #        self.fig.tight_layout(.3, None, None, None)

        self.im = None

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Other GUI controls
        #
        # lay out main panel
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.canvas)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_status_bar(self):
        self.status_text = QtGui.QLabel('')
        self.fname_text = QtGui.QLabel('')
        self.fname_text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.statusBar().addWidget(self.status_text)
        self.prog_bar = QtGui.QProgressBar()
        self.prog_bar.setRange(0, 0)
        self.prog_bar.hide()
        self.statusBar().addWidget(self.prog_bar, 1)
        self.statusBar().addPermanentWidget(self.fname_text)

    def closeEvent(self, ce):
        self.kill_thread.emit()

        self.diag.close()
        QtGui.QMainWindow.closeEvent(self, ce)

    def create_actions(self):

        def set_dir(cap, d):
            print cap
            base_dir = QtGui.QFileDialog.getExistingDirectory(self,
                                                              caption=cap,
                                                              dir=d[cap])
            if len(base_dir) > 0:
                d[cap] = base_dir

        self.show_cntrl_acc = QtGui.QAction(u'show controls', self)
        self.show_cntrl_acc.triggered.connect(self.show_cntrls)

        cap_lst = self.cap_lst
        cta_lst = ['Select ' + x for x in cap_lst]

        self.directory_actions = {}
        for cap, cta in zip(cap_lst, cta_lst):
            tmp_acc = QtGui.QAction(cta, self)
            self.directory_actions[cap] = tmp_acc

        self.open_file_acc = QtGui.QAction(u'Open &File', self)
        self.open_file_acc.triggered.connect(self.open_file)

        self.set_fringes_acc = QtGui.QAction(u'&Display Fringes', self)
        self.set_fringes_acc.setCheckable(True)
        self.set_fringes_acc.setChecked(self.draw_fringes)
        self.set_fringes_acc.toggled.connect(self.set_fringes_visible)

        self.set_all_fringes_acc = QtGui.QAction(u'Display &All Fringes', self)
        self.set_all_fringes_acc.setCheckable(True)
        self.set_all_fringes_acc.setChecked(self.all_fringes_flg)
        self.set_all_fringes_acc.toggled.connect(self.set_all_friges)

        self.show_track_graphs = QtGui.QAction(u'Display &Track graph', self)
        self.show_track_graphs.setCheckable(True)
        self.show_track_graphs.setChecked(False)

        self.bin_search_acc = QtGui.QAction(u'Binary Search', self)
        self.bin_search_acc.triggered.connect(self.create_binary_search)

        for cap, cta in zip(cap_lst, cta_lst):
            tmp_acc = QtGui.QAction(cta, self)
            self.directory_actions[cap] = tmp_acc

    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.open_file_acc)
        for ac in self.directory_actions:
            fileMenu.addAction(ac)

        fringeMenu = menubar.addMenu('Fringes')
        fringeMenu.addAction(self.set_fringes_acc)
        fringeMenu.addAction(self.set_all_fringes_acc)
        fringeMenu.addAction(self.bin_search_acc)

        graphMenu = menubar.addMenu('Graphs')
        graphMenu.addAction(self.show_track_graphs)

    def create_binary_search(self):
        self.bin_search = BinaryFrameSearch(len(self.reader) - 1)
        self.bin_search.change_frame.connect(self.frame_spinner.setValue)
        self.bin_search.reset()
        self.bin_search.show()


class PickerHandler(object):
    def __init__(self, canv, fun=None):
        canv.mpl_connect('pick_event', self.on_pick)
        self.fun = fun

    def on_pick(self, event):
        art = event.artist
        # if not a Line2D, don't want to deal with it
        if not isinstance(art, matplotlib.lines.Line2D):
            return

        payload = art.payload()
        if payload is not None:
            self.fun(payload)
        else:
            print 'fail type 2'


class GraphDialog(QtGui.QDialog):
    def __init__(self, grid_size=(1, 1), parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.fig = Figure((10, 10))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui. QSizePolicy.Expanding)
        self.gs = gridspec.GridSpec(*grid_size)
        self.axes_list = [self.fig.add_subplot(s) for s in self.gs]
        self.gs.tight_layout(self.fig, rect=[0, 0, 1, 1])

    def update_axes(self, fun, args):
        fun(self.axes_list, args)
        self.gs.tight_layout(self.fig)
        self.canvas.draw()

    def resizeEvent(self, re):
        QtGui.QDialog.resizeEvent(self, re)
        self.canvas.resize(re.size().width(), re.size().height())
        self.gs.tight_layout(self.fig, rect=[0, 0, 1, 1], pad=0)


class BinaryFrameSearch(QtGui.QDialog):
    change_frame = QtCore.Signal(int)

    def __init__(self, max_number, min_number=0, parent=None):
        QtGui.QDialog.__init__(self, parent)
        main_layout = QtGui.QVBoxLayout()

        info_layout = QtGui.QHBoxLayout()

        info_layout.addWidget(QtGui.QLabel('cur frame: '))
        self.cur_frame_label = QtGui.QLabel('')
        info_layout.addWidget(self.cur_frame_label)
        info_layout.addStretch()

        info_layout.addWidget(QtGui.QLabel('bottom: '))
        self.bottom_label = QtGui.QLabel(str(min_number))
        info_layout.addWidget(self.bottom_label)
        info_layout.addStretch()

        info_layout.addWidget(QtGui.QLabel('top: '))
        self.top_label = QtGui.QLabel(str(max_number))
        info_layout.addWidget(self.top_label)

        button_layout = QtGui.QHBoxLayout()

        self.good_button = QtGui.QPushButton('good')
        self.good_button.clicked.connect(self.good_jump)
        button_layout.addWidget(self.good_button)

        self.bad_button = QtGui.QPushButton('bad')
        self.bad_button.clicked.connect(self.bad_jump)
        button_layout.addWidget(self.bad_button)

        self.reset_button = QtGui.QPushButton('reset')
        self.reset_button.clicked.connect(self.reset)
        button_layout.addWidget(self.reset_button)

        main_layout.addLayout(info_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self._min = min_number
        self._max = max_number

        self.bottom = min_number
        self.top = max_number
        self.cur = 0

    @QtCore.Slot()
    def good_jump(self):
        self.bottom = self.cur
        self.cur = self.cur + int((self.top - self.cur) // 2)

        self._update()

    @QtCore.Slot()
    def bad_jump(self):
        self.top = self.cur
        self.cur = self.cur + int((self.bottom - self.cur) // 2)
        self._update()

    @QtCore.Slot()
    def reset(self):
        self.top = self._max
        self.bottom = self._min
        self.cur = self.bottom
        self._update()

    def _update(self):
        self.change_frame.emit(self.cur)
        self.cur_frame_label.setText(str(self.cur))
        self.top_label.setText(str(self.top))
        self.bottom_label.setText(str(self.bottom))
