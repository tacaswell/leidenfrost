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

from collections import defaultdict

import numpy as np

import leidenfrost.infra as infra
import cine


class LFWorker(QtCore.QObject):
    frame_proced = QtCore.Signal(bool, bool)
    file_loaded = QtCore.Signal(bool, bool)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.process_backend = None

        self.mbe = None
        self.next_curve = None

    @QtCore.Slot(int, infra.SplineCurve)
    def proc_frame(self, ind, curve):

        if self.process_backend is not None:
            self.mbe, self.next_curve = self.process_backend.process_frame(ind,
                                                                           curve)
            self.frame_proced.emit(True, True)

    def get_mbe(self):
        return self.mbe

    def get_next_curve(self):
        return self.next_curve

    def update_param(self, key, val):
        if self.process_backend is not None:
            self.process_backend.update_param(key, val)

    def get_frame(self, ind):
        if self.process_backend is not None:
            tmp = self.process_backend.get_frame(ind)
            return np.asarray(tmp, dtype=np.float)
        return None

    def clear(self):
        self.mbe = None
        self.next_curve = None

    def __len__(self):
        if self.process_backend is not None:
            return len(self.process_backend)
        return 0

    @QtCore.Slot(infra.FilePath, dict)
    def set_new_fname(self, cine_fname, params):
        print cine_fname, params
        self.clear()
        self.process_backend = infra.ProcessBackend.from_args(cine_fname,
                                                              bck_img=None,
                                                              **params)
        self.file_loaded.emit(True, True)


class LFGui(QtGui.QMainWindow):
    proc = QtCore.Signal(int, infra.SplineCurve)
    open_file_sig = QtCore.Signal(infra.FilePath, dict)
    kill_thread = QtCore.Signal()
    redraw_sig = QtCore.Signal(bool, bool)
    spinner_lst = [
        {'name': 's_width',
         'min': 0,
         'max': 50,
         'step': .5,
         'prec': 1,
         'type': np.float,
         'default': 10},
         {'name': 's_num',
          'min': 1,
          'max': 500,
          'step': 1,
          'type': np.int,
          'default': 100},
        {'name': 'search_range',
         'min': .001,
         'max': 2 * np.pi,
         'step': .005,
         'prec': 3,
         'type': np.float,
         'default': .01},
        {'name': 'memory',
         'min': 0,
         'max': 500,
         'step': 1,
         'type': np.int,
         'default': 0},
        {'name': 'pix_err',
         'min': 0,
         'max': 5,
         'step': .1,
         'prec': 1,
         'type': np.float,
         'default': 2},
        {'name': 'mix_in_count',
         'min': 0,
         'max': 100,
         'step': 1,
         'type': np.int,
         'default': 10}
         ]

    cap_lst = ['hdf base path','cine base directory','cine cache path','hdf cache path']
    
    def __init__(self, cine_fname=None, bck_img=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Fringe Finder')
        if cine_fname is not None:
            self.cine_fname = cine_fname
            self.base_dir = cine_fname[0]
        else:
            self.cine_fname = None
            self.base_dir = None
        self.cur_frame = 0

        self.draw_fringes = False

        default_params = dict((d['name'], d['default']) for
                              d in LFGui.spinner_lst)

        

        self.thread = QtCore.QThread(parent=self)

        self.worker = LFWorker(parent=None)
        self.worker.moveToThread(self.thread)

        self.cur_curve = None
        self.next_curve = None

        self.fringe_lines = []

        self.all_fringes_flg = False
        self.param_spin_dict = {}
        self.create_main_frame()
        self.create_actions()
        self.create_menu_bar()
        self.create_diag()
        self.create_status_bar()

        self.on_draw(True, True)
        self.worker.frame_proced.connect(self.on_draw)
        self.worker.file_loaded.connect(self.redraw)
        self.proc.connect(self.worker.proc_frame)
        self.open_file_sig.connect(self.worker.set_new_fname)
        self.redraw_sig.connect(self.on_draw)

        self.kill_thread.connect(self.thread.quit)

        self.show()
        self.thread.start()
        QtGui.qApp.exec_()

    def grab_sf_curve(self):
        try:
            self.cur_curve = self.sf.return_SplineCurve()
            if self.cur_curve is not None:
                self.proc_this_frame_acc.setEnabled(True)
                self.save_param_acc.setEnabled(True)
            else:
                print 'no spline!?'
        except:
            print 'spline fitter not ready'
            self.cur_curve = None

    def update_param(self, key, val):
        self.worker.update_param(key, val)
        if self.draw_fringes:
            self._proc_this_frame()

    def _proc_this_frame(self):

        print 'entered _proc_this_frame'
        self.prog_bar.show()
        self.diag.setEnabled(False)
        self.draw_fringes = True
        self.proc_next_frame_acc.setEnabled(True)
        self.iterate_button.setEnabled(True)

        self.proc.emit(self.cur_frame, self.cur_curve)

    def _proc_next_frame(self):

        self.frame_spinner.setValue(self.cur_frame + 1)
        self.diag.setEnabled(False)
        #self._proc_this_frame()

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
        self.fringe_grp_bx.setChecked(self.draw_fringes)

        # update the image
        if refresh_img and self.im is not None:
            img = self.worker.get_frame(self.cur_frame)
            if img is not None:
                self.im.set_data(img)
            
        # if we need to update the lines
        if refresh_lines:
            # clear the lines we have
            for ln in self.fringe_lines:
                ln.remove()
            # nuke all those objects
            self.fringe_lines = []

            if self.draw_fringes:
                # if we should draw new ones, do so

                # grab new mbe from thread object
                mbe = self.worker.get_mbe()
                # grab new next curve
                self.next_curve = self.worker.get_next_curve()
                if self.draw_fringes and mbe is not None:
                    self.fringe_lines.extend(
                        mbe.ax_plot_tracks(self.axes,
                                           min_len=0,
                                           all_tracks=self.all_fringes_flg)
                        )
                    self.fringe_lines.extend(
                        mbe.ax_draw_center_curves(self.axes))

        self.canvas.draw()
        self.status_text.setNum(self.cur_frame)
        self.prog_bar.hide()
        self.diag.setEnabled(True)

    def clear_mbe(self):
        self.cur_curve = None
        self.next_curve = None
        self.proc_this_frame_acc.setEnabled(False)
        self.proc_next_frame_acc.setEnabled(False)
        self.save_param_acc.setEnabled(False)
        self.iterate_button.setEnabled(False)
        self.refresh_lines_flg = True
        self.fringe_grp_bx.setChecked(False)

    def set_spline_fitter(self, i):
        if i:
            self.sf.connect_sf()
            # if we can screw with it, make it visible
            self.sf_show.setChecked(True)
        else:
            self.sf.disconnect_sf()
        self.redraw_sig.emit(False, False)

    def set_spline_fitter_visible(self, i):
        self.sf.set_visible(i)
        # if we can't see it, don't let is screw with it
        if not bool(i):
            self.sf_check.setChecked(False)
        self.redraw_sig.emit(False, False)

    def set_cur_frame(self, i):
        old_frame = self.cur_frame
        self.cur_frame = i
        self.refresh_lines_flg = True
        self.refresh_img = True
        if old_frame == self.cur_frame - 1:
            self.cur_curve = self.next_curve
            if self.draw_fringes:
                self._proc_this_frame()
            else:
                self.redraw_sig.emit(False, True)

        else:
            self.draw_fringes = False
            self.worker.clear()
            self.redraw_sig.emit(True, True)

    def iterate_frame(self):
        self.cur_curve = self.next_curve
        self._proc_this_frame()

    def clear_spline(self):
        self.sf.clear()

    def show_cntrls(self):
        self.diag.show()

    def save_config(self):
        fname, _ = QtGui.QFileDialog.getSaveFileName(self,
                                                     caption='Save File',
                                                     dir=self.base_dir)
        if len(fname) > 0:
                self.worker.process_backend.gen_stub_h5(fname, self.cur_curve)

    def set_base_dir(self):
        base_dir = QtGui.QFileDialog.getExistingDirectory(self,
                                                          caption='Base Directory',
                                                          dir=self.base_dir)
        if len(base_dir) > 0:
            self.base_dir = base_dir

    def open_file(self):

        
        fname, _ = QtGui.QFileDialog.getOpenFileName(self,
                                                     caption='Save File',
                                                     dir=self.base_dir)
        if len(fname) == 0:
            return
        
        self.fringe_grp_bx.setChecked(False)        
        while self.base_dir is None or (not (self.base_dir == fname[:len(self.base_dir)])):
            print 'please set base_dir'
            self.set_base_dir()

        self.prog_bar.show()
        self.diag.setEnabled(False)

        path_, fname_ = os.path.split(fname[(len(self.base_dir) + 1):])
        new_cine_fname = infra.FilePath(self.base_dir, path_, fname_)
        self.fname_text.setText('/'.join(new_cine_fname[1:]))
        self.clear_mbe()
        default_params = dict((d['name'], d['default']) for
                              d in LFGui.spinner_lst)

        # reset spinners to default values
        for p in self.spinner_lst:
            self.param_spin_dict[p['name']].setValue(p['default'])
        self.open_file_sig.emit(new_cine_fname, default_params)

    def create_diag(self):
        # make top level stuff
        self.diag = QtGui.QDockWidget('controls', parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.diag)
        diag_widget = QtGui.QWidget(self.diag)
        self.diag.setWidget(diag_widget)
        diag_layout = QtGui.QVBoxLayout()
        diag_widget.setLayout(diag_layout)

        # frame number lives on top
        self.frame_spinner = QtGui.QSpinBox()
        self.frame_spinner.setRange(0, len(self.worker) - 1)
        self.frame_spinner.valueChanged.connect(self.set_cur_frame)
        fs_form = QtGui.QFormLayout()
        fs_form.addRow(QtGui.QLabel('frame #'), self.frame_spinner)

        diag_layout.addLayout(fs_form)

        # tool box for all the controls
        diag_tool_box = QtGui.QToolBox()
        diag_layout.addWidget(diag_tool_box)

        # section for dealing with fringe finding

        # the widget to shove into the toolbox
        fringe_cntrls_w = QtGui.QWidget()
        # vbox layout for this panel
        fc_vboxes = QtGui.QVBoxLayout()
        # set the widget layout
        
        fringe_cntrls_w.setLayout(fc_vboxes)
        # add to the tool box
        diag_tool_box.addItem(fringe_cntrls_w, "Fringe Finding Settings")

        # form layout to hold the spinners
        fringe_cntrls_spins = QtGui.QFormLayout()
        # add spinner layout
        fc_vboxes.addLayout(fringe_cntrls_spins)

        # fill the spinners
        for spin_prams in LFGui.spinner_lst:

            s_type = np.dtype(spin_prams['type']).kind

            if s_type == 'i':
                spin_box = QtGui.QSpinBox(parent=self)
            elif s_type == 'f':
                spin_box = QtGui.QDoubleSpinBox(parent=self)
                spin_box.setDecimals(spin_prams['prec'])
            else:
                print s_type
                continue

            spin_box.setRange(spin_prams['min'], spin_prams['max'])
            spin_box.setSingleStep(spin_prams['step'])
            spin_box.setValue(spin_prams['default'])
            name = spin_prams['name']

            spin_box.valueChanged.connect(self._gen_update_closure(name))
            fringe_cntrls_spins.addRow(QtGui.QLabel(spin_prams['name']),
                                       spin_box)
            self.param_spin_dict[name] = spin_box
            
        # button to grab initial spline
        grab_button = QtGui.QPushButton('Grab Spline')
        grab_button.clicked.connect(self.grab_sf_curve)
        fc_vboxes.addWidget(grab_button)
        # button to process this frame

        ptf_button = QtGui.QPushButton('Process This Frame')
        ptf_button.clicked.connect(self.proc_this_frame_acc.trigger)
        ptf_button.setEnabled(self.proc_this_frame_acc.isEnabled())
        self.proc_this_frame_acc.changed.connect(
            lambda: ptf_button.setEnabled(self.proc_this_frame_acc.isEnabled()))

        fc_vboxes.addWidget(ptf_button)

        # button to process next frame
        pnf_button = QtGui.QPushButton('Process Next Frame')
        pnf_button.clicked.connect(self.proc_next_frame_acc.trigger)
        pnf_button.setEnabled(self.proc_next_frame_acc.isEnabled())
        self.proc_next_frame_acc.changed.connect(
            lambda: pnf_button.setEnabled(self.proc_next_frame_acc.isEnabled()))

        fc_vboxes.addWidget(pnf_button)

        # nuke tracking data

        clear_mbe_button = QtGui.QPushButton('Clear fringes')
        clear_mbe_button.clicked.connect(self.clear_mbe)
        fc_vboxes.addWidget(clear_mbe_button)

        self.fringe_grp_bx = QtGui.QGroupBox("Draw Fringes")
        self.fringe_grp_bx.setCheckable(True)
        self.fringe_grp_bx.setChecked(False)
        self.fringe_grp_bx.toggled.connect(self.set_fringes_visible)

        #        self.proc_this_frame_acc.triggered.connect(
        #   lambda:self.fringe_grp_bx.setChecked(True))

        all_fringe_rb = QtGui.QRadioButton('All Fringes')
        valid_fringes_rb = QtGui.QRadioButton('Valid Fringes')
        rb_vbox = QtGui.QVBoxLayout()
        rb_vbox.addWidget(valid_fringes_rb)
        rb_vbox.addWidget(all_fringe_rb)
        self.fringe_grp_bx.setLayout(rb_vbox)
        all_fringe_rb.toggled.connect(self.set_all_friges)
        all_fringe_rb.setChecked(False)
        valid_fringes_rb.setChecked(True)
        fc_vboxes.addWidget(self.fringe_grp_bx)

        iterate_button = QtGui.QPushButton('Iterate fringes')
        iterate_button.clicked.connect(self.iterate_frame)
        iterate_button.setEnabled(False)
        fc_vboxes.addWidget(iterate_button)
        self.iterate_button = iterate_button

        save_param_bttn = QtGui.QPushButton('Save Configuration')
        save_param_bttn.clicked.connect(self.save_param_acc.trigger)

        save_param_bttn.setEnabled(self.save_param_acc.isEnabled())
        self.save_param_acc.changed.connect(
            lambda: save_param_bttn.setEnabled(self.save_param_acc.isEnabled()))

        fc_vboxes.addWidget(save_param_bttn)
        fc_vboxes.addStretch()
        
        # section for making spline fitting panel
        spline_cntrls = QtGui.QVBoxLayout()
        spline_cntrls_w = QtGui.QWidget()
        spline_cntrls_w.setLayout(spline_cntrls)

        self.sf_check = QtGui.QCheckBox('enabled input')
        self.sf_check.stateChanged.connect(self.set_spline_fitter)
        spline_cntrls.addWidget(self.sf_check)

        self.sf_show = QtGui.QCheckBox('display fitter')
        self.sf_show.stateChanged.connect(self.set_spline_fitter_visible)
        spline_cntrls.addWidget(self.sf_show)

        clear_spline_button = QtGui.QPushButton('Clear Spline')
        clear_spline_button.clicked.connect(self.clear_spline)
        spline_cntrls.addWidget(clear_spline_button)
        
        spline_cntrls.addStretch()
        
        diag_tool_box.addItem(spline_cntrls_w, "Manual Spline Fitting")

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        # create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #

        self.fig = Figure((24, 24))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.axes = self.fig.add_subplot(111)
        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.im = None

        #        tmp = self.worker.get_frame(self.cur_frame)

        #        self.im = self.axes.imshow(tmp, cmap='gray', interpolation='nearest')
        #        self.axes.set_aspect('equal')
        #        self.im.set_clim([.5, 1.5])

        self.sf = infra.spline_fitter(self.axes)
        self.sf.disconnect_sf()

        #        self.axes.set_xlim(left=0)
        #        self.axes.set_ylim(top=0)         # this is because images are plotted upside down
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

    @QtCore.Slot(bool,bool)
    def redraw(self,draw_img,draw_tracks):
        if self.im is not None:
            self.im.remove()
        self.im = None
        print 'entered redraw'

        # clear the lines we have
        for ln in self.fringe_lines:
            ln.remove()
        self.fringe_lines = []


        mbe = self.worker.get_mbe()
        img = self.worker.get_frame(0)

        if img is None:
            img = 1.5 * np.ones((1,1))

        extent = [0,img.shape[1],0,img.shape[0]]

        self.im = self.axes.imshow(img,
                                   cmap='gray',
                                   extent=extent,
                                   origin='lower',
                                   interpolation='nearest')
        self.im.set_clim([.5, 1.5])
        self.axes.set_aspect('equal')


        if self.draw_fringes and mbe is not None:
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

        self.frame_spinner.setRange(0,len(self.worker)-1)
        self.frame_spinner.setValue(0)

        self.redraw_sig.emit(False,False)


    def _gen_update_closure(self, name):
        return lambda x: self.update_param(name, x)

    def create_status_bar(self):
        self.status_text = QtGui.QLabel(str(self.cur_frame))
        
        self.fname_text = QtGui.QLabel('')
        self.statusBar().addWidget(self.status_text)
        self.prog_bar = QtGui.QProgressBar()
        self.prog_bar.setRange(0, 0)
        self.prog_bar.hide()
        self.statusBar().addWidget(self.prog_bar, 1)
        self.statusBar().addPermanentWidget(self.fname_text)

    def closeEvent(self, ce):
        self.kill_thread.emit()
        #        QtGui.qApp.quit()
        #        self.thread.quit()
        self.diag.close()
        QtGui.QMainWindow.closeEvent(self, ce)

    def create_actions(self):
        self.show_cntrl_acc = QtGui.QAction(u'show controls', self)
        self.show_cntrl_acc.triggered.connect(self.show_cntrls)

        self.save_param_acc = QtGui.QAction(u'Save Parameters', self)
        self.save_param_acc.setEnabled(False)
        self.save_param_acc.triggered.connect(self.save_config)

        self.set_base_dir_acc = QtGui.QAction(u'Select base dir', self)
        self.set_base_dir_acc.triggered.connect(self.set_base_dir)

        self.open_file_acc = QtGui.QAction(u'Open &File', self)
        self.open_file_acc.triggered.connect(self.open_file)

        self.proc_this_frame_acc = QtGui.QAction('Process this Frame', self)
        self.proc_this_frame_acc.setEnabled(False)
        self.proc_this_frame_acc.triggered.connect(self._proc_this_frame)

        self.proc_next_frame_acc = QtGui.QAction('Process next Frame', self)
        self.proc_next_frame_acc.setEnabled(False)
        self.proc_next_frame_acc.triggered.connect(self._proc_next_frame)

    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.show_cntrl_acc)
        fileMenu.addAction(self.save_param_acc)
        fileMenu.addAction(self.set_base_dir_acc)
        fileMenu.addAction(self.open_file_acc)

        procMenu = menubar.addMenu('&Process')
        procMenu.addAction(self.proc_this_frame_acc)
        procMenu.addAction(self.proc_next_frame_acc)


class LFReader(QtCore.QObject):
    frame_loaded = QtCore.Signal(bool, bool)
    file_loaded = QtCore.Signal(bool,bool)

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

            self.frame_loaded.emit(True,True)



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

    @QtCore.Slot(infra.FilePath,str,dict)
    def set_fname(self, fname,cbp,kwargs):
        print 'entered set_fname'
        self.clear()

        print fname,cbp,kwargs
        self.backend = infra.HdfBackend(fname,
                                                cine_base_path=cbp,
                                                **kwargs)
        self.mbe = self.backend.get_frame(0,
                                          raw=True,
                                          get_img=True)
        self.cur_frame = 0

        self.file_loaded.emit(True,True)




class LFReaderGui(QtGui.QMainWindow):
    read_request_sig = QtCore.Signal(int)
    open_file_sig = QtCore.Signal(infra.FilePath,str, dict)
    kill_thread = QtCore.Signal()
    redraw_sig = QtCore.Signal(bool, bool)
    cap_lst = ['hdf base path','cine base directory','cine cache path','hdf cache path']

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Fringe Display')

        self.draw_fringes = False
        self.all_fringes_flg = False

        self.reader = LFReader()
        self.thread = QtCore.QThread(parent=self)
        self.reader.moveToThread(self.thread)



        self.fringe_lines = []

        self.paths_dict = defaultdict(lambda :None)


        self.create_main_frame()
        self.create_actions()
        self.create_menu_bar()
        self.create_diag()
        self.create_status_bar()

        self.read_request_sig.connect(self.reader.read_frame)
        self.kill_thread.connect(self.thread.quit)
        self.open_file_sig.connect(self.reader.set_fname)
        self.redraw_sig.connect(self.on_draw)

        self.reader.frame_loaded.connect(self.on_draw)
        self.reader.file_loaded.connect(self.redraw)

        self.show()

        self.thread.start()

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
        print 'on_draw'

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



            pass

        if refresh_img:
            img = mbe.img
            if img is not None and self.im is not None:
                self.im.set_data(img)




        self.canvas.draw()
        pass


    @QtCore.Slot(bool,bool)
    def redraw(self,draw_img,draw_tracks):
        if self.im is not None:
            self.im.remove()

        print 'entered redraw'

        # clear the lines we have
        for ln in self.fringe_lines:
            ln.remove()

        self.axes = None
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        # nuke all those objects
        self.fringe_lines = []
        self.im = None
        mbe = self.reader.get_mbe()

        img = mbe.img
        if img is None:
            img = 1.5 * np.ones((1,1))

        extent = mbe.get_extent()

        self.im = self.axes.imshow(img,
                                   cmap='gray',
                                   extent=extent,
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

        self.frame_spinner.setRange(0,len(self.reader)-1)
        self.frame_spinner.setValue(0)

        self.redraw_sig.emit(False,False)


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
                                                     caption='Save File')
        if len(fname) == 0:
            return

        while not hdf_bp == fname[:len(hdf_bp)]:
            print 'please set base_dir'
            self.directory_actions['hdf base path'].trigger()
            hdf_bp = self.paths_dict['hdf base path']

        path_, fname_ = os.path.split(fname[(len(hdf_bp) + 1):])
        new_hdf_fname = infra.FilePath(hdf_bp, path_, fname_)

        tmp_dict = {'cine_cache_dir':self.paths_dict['cine cache path'],
                    'hdf_cache_dir': self.paths_dict['hdf cache path']}

        print new_hdf_fname
        print '/'.join(new_hdf_fname)
        self.open_file_sig.emit(new_hdf_fname,cine_bp,tmp_dict)

    def create_diag(self):


        self.diag = QtGui.QDockWidget('controls',parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,self.diag)
        diag_widget = QtGui.QWidget(self.diag)
        self.diag.setWidget(diag_widget)
        diag_layout = QtGui.QVBoxLayout()
        diag_widget.setLayout(diag_layout)


        # frame number lives on top
        self.frame_spinner = QtGui.QSpinBox()
        self.frame_spinner.setRange(0,len(self.reader)-1)
        self.frame_spinner.valueChanged.connect(self.set_cur_frame)
        fs_form = QtGui.QFormLayout()
        fs_form.addRow(QtGui.QLabel('frame #'),self.frame_spinner)



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

        diag_layout.addLayout(fs_form)

        diag_layout.addWidget(self.fringe_grp_bx)
        path_box = QtGui.QGroupBox("paths")
        pb_layout = QtGui.QVBoxLayout()
        path_box.setLayout(pb_layout)
        for c in self.cap_lst:
            ds = directory_selector(caption=c)
            pb_layout.addWidget(ds)
            self.directory_actions[c].triggered.connect(ds.select_path)
            ds.selected.connect(lambda x,c=c : self.paths_dict.__setitem__(c,x))

        diag_layout.addWidget(path_box)
        diag_layout.addStretch()
        pass

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        # create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #

        self.fig = Figure((24, 24))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)


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

        def set_dir(cap,d):
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
        for cap,cta in zip(cap_lst,cta_lst):
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


    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.open_file_acc)
        for ac in self.directory_actions:
            fileMenu.addAction(ac)

        fringeMenu = menubar.addMenu('Fringes')
        fringeMenu.addAction(self.set_fringes_acc)
        fringeMenu.addAction(self.set_all_fringes_acc)


class directory_selector(QtGui.QWidget):
    '''
    A widget class deal with selecting and displaying path names
    '''

    selected =  QtCore.Signal(str)

    def __init__(self,caption,path='',parent=None):
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

        button = QtGui.QPushButton('')
        button.setIcon(QtGui.QIcon.fromTheme('folder'))
        button.clicked.connect(self.select_path)
        hlayout.addWidget(button)




    @QtCore.Slot(str)
    def set_path(self,path):
        pass

    @QtCore.Slot()
    def select_path(self):
        path = QtGui.QFileDialog.getExistingDirectory(self,
                                                      caption=self.cap,
                                                      dir=None)

        if len(path) > 0:
            self.selected.emit(path)
            self.label.setText(path)
            self.selected.emit(path)
            return path
        else:
            path = None
        self.path = path
        return path
