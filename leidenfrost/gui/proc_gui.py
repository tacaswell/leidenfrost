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
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import str

import os
import copy

# do this to make me learn where stuff is and to make it easy to
# switch to PyQt later
import PySide.QtCore as QtCore
import PySide.QtGui as QtGui


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .common import directory_selector, numbered_paths, base_path_path_selector
from .common import dict_display
from collections import defaultdict

import numpy as np

import leidenfrost
import leidenfrost.infra as infra
import leidenfrost.proc
import leidenfrost.backends as backends
import leidenfrost.db as ldb

from IPython.parallel import Client


class LFWorker(QtCore.QObject):
    frame_proced = QtCore.Signal(bool, bool)
    file_loaded = QtCore.Signal(bool, bool)
    md_loadad = QtCore.Signal(dict)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.process_backend = None

        self.mbe = None
        self.next_curve = None

        self.db = None
        try:
            self.db = ldb.LFmongodb()
        except Exception as e:
            print(e)
            print('no database for you!')

    @QtCore.Slot(int, infra.SplineCurve)
    def proc_frame(self, ind, curve):

        if self.process_backend is not None:
            try:
                self.mbe, self.next_curve = self.process_backend.process_frame(
                    ind, curve)
                self.frame_proced.emit(True, True)
            except Exception as e:
                print(e)
                print('something is borked')
                self.frame_proced.emit(False, False)

    @QtCore.Slot()
    def set_useful(self):
        if self.db is not None and self.process_backend is not None:
            self.db.set_cine_useful(self.process_backend.cine_.hash, True)
            self.md_loadad.emit(self.get_cine_md())

    @QtCore.Slot()
    def set_useless(self):
        if self.db is not None and self.process_backend is not None:
            self.db.set_cine_useful(self.process_backend.cine_.hash, False)
            self.md_loadad.emit(self.get_cine_md())

    def get_cine_md(self):
        if self.db is not None and self.process_backend is not None:
            print('trying to grab md')
            md_dict = self.db.get_movie_md(self.process_backend.cine_.hash)
            print('trying to grabed md')
            return md_dict
        print('md not attempted to grab from DB')
        return {}

    def get_mbe(self):
        return self.mbe

    def get_next_curve(self):
        return self.next_curve

    @QtCore.Slot(dict)
    def update_all_params(self, params):

        if self.process_backend is not None:
            self.process_backend.update_all_params(params)

    def get_frame(self, ind, *args, **kwargs):
        if self.process_backend is not None:
            tmp = self.process_backend.get_image(ind, *args, **kwargs)
            return np.asarray(tmp, dtype=np.float)
        return None

    def clear(self):
        self.mbe = None
        self.next_curve = None

    def __len__(self):
        if self.process_backend is not None:
            return len(self.process_backend)
        return 0

    @QtCore.Slot(backends.FilePath, dict)
    def set_new_fname(self, cine_fname, params):

        self.clear()
        self.process_backend = backends.ProcessBackend.from_args(cine_fname,
                                                                 bck_img=None,
                                                                 **params)
        self.file_loaded.emit(True, True)
        self.md_loadad.emit(self.get_cine_md())

    def start_comp(self, seed_curve, name_template, cur_frame, disk_dict):
        # make the connection to the ether
        lb_view = Client(profile='vpn',
                         sshserver='10.8.0.1').load_balanced_view()
        # grab a copy of the paramters
        proc_prams = copy.copy(self.process_backend.params)
        # put in the start_frame
        if cur_frame is not None:
            proc_prams['start_frame'] = cur_frame
        name_template = leidenfrost.convert_base_path(name_template, disk_dict)
        # push to ether and hope!
        as_res = lb_view.apply_async(leidenfrost.proc.proc_cine_to_h5,
                            self.process_backend.cine_.hash,
                            name_template, proc_prams, seed_curve)
        print('fired: {}'.format(as_res.msg_id))


class LFGui(QtGui.QMainWindow):
    proc = QtCore.Signal(int, infra.SplineCurve)
    open_file_sig = QtCore.Signal(backends.FilePath, dict)
    kill_thread = QtCore.Signal()
    redraw_sig = QtCore.Signal(bool, bool)
    draw_done_sig = QtCore.Signal()
    spinner_lst = [
        {'name': 's_width',
         'min': 0,
         'max': 99,
         'step': .5,
         'prec': 1,
         'type': np.float,
         'default': 10,
         'togglable': False},
        {'name': 's_num',
         'min': 1,
         'max': 9999,
         'step': 1,
         'type': np.int,
         'default': 100,
         'togglable': False},
        {'name': 'search_range',
         'min': .001,
         'max': 2 * np.pi,
         'step': .005,
         'prec': 3,
         'type': np.float,
         'default': .01,
         'togglable': False},
        {'name': 'memory',
         'min': 0,
         'max': 999,
         'step': 1,
         'type': np.int,
         'default': 0,
         'togglable': False},
        {'name': 'pix_err',
         'min': 0,
         'max': 9,
         'step': .1,
         'prec': 1,
         'type': np.float,
         'default': 0.5,
         'togglable': False},
        {'name': 'mix_in_count',
         'min': 0,
         'max': 100,
         'step': 1,
         'type': np.int,
         'default': 0,
         'togglable': False},
        {'name': 'min_tlen',
         'min': 0,
         'max': 999999,
         'step': 1,
         'type': np.int,
         'default': 15,
         'togglable': True,
         'default_state': True},
        {'name': 'fft_filter',
         'min': 0,
         'max': 999,
         'step': 1,
         'type': np.int,
         'default': 10,
         'togglable': True,
         'default_state': True},
        {'name': 'min_extent',
         'min': 0,
         'max': 999,
         'step': 1,
         'type': np.int,
         'default': 10,
         'togglable': True,
         'default_state': False,
         'tooltip': 'The minimum extent in q[pixel] of a track to be valid'},
        {'name': 'max_gap',
         'min': 0,
         'max': np.pi,
         'step': np.pi / 100,
         'type': np.float,
         'default': np.pi / 6,
         'prec': 3,
         'togglable': True,
         'default_state': True,
         'tooltip': 'The maximum gap (in rad) before the' +
                      ' alternate gap handling is used'},
         {'name': 'max_circ_change',
          'min': 0,
          'max': 1,
          'step': .0005,
          'prec': 4,
          'type': np.float,
          'default': .002,
          'togglable': True,
          'default_state': True,
          'tooltip': """The maximum fractional change in the circumference of
successive rims.  If exceeded, the previous seed-curve is re-used"""},
         {'name': 'end_frame',
          'min': -1,
          'max': 2**31 - 1,
          'step': 1,
          'type': np.int,
          'default': -1,
          'togglable': True,
          'default_state': False,
          'tooltip': """The last frame to process to"""},

    ]

    toggle_lst = [
        {'name': 'straddle',
         'default': True,
         'tooltip': 'If checked, then tracks must cross' +
                     ' the seed line to be valid'}]

    cap_lst = ['cine base path']

    @QtCore.Slot()
    def unlock_gui(self):
        """
        Forcable un-locks the gui
        """
        self.prog_bar.hide()
        self.diag.setEnabled(True)

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Fringe Finder')

        self.cine_fname = None

        self.cur_frame = 0

        self.draw_fringes = False

        default_params = dict((d['name'], d['default']) for
                              d in self.spinner_lst
                              if ('default_state' not in d or
                                    d['default_state']))
        for tog in self.toggle_lst:
            default_params[tog['name']] = tog['default']

        self.thread = QtCore.QThread(parent=self)

        self.worker = LFWorker(parent=None)
        self.worker.moveToThread(self.thread)

        self.cur_curve = None
        self.next_curve = None

        self.fringe_lines = []

        self.all_fringes_flg = False

        self.paths_dict = defaultdict(lambda: None)
        self.directory_actions = {}
        self.param_spin_dict = {}
        self.param_checkbox_dict = {}

        self.create_actions()
        self.create_main_frame()
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

        self.draw_done_sig.connect(self._play)
        self.draw_done_sig.connect(self.unlock_gui)

        self.show()
        self.thread.start()
        QtGui.qApp.exec_()

    def grab_sf_curve(self):
        try:
            self.cur_curve = self.sf.return_SplineCurve()
            if self.cur_curve is not None:
                self.proc_this_frame_acc.setEnabled(True)
                self.start_comp_acc.setEnabled(True)
            else:
                print('no spline!?')
        except:
            print('spline fitter not ready')
            self.cur_curve = None

    # def update_param(self, key, val):
    #     self.worker.update_param(key, val)

    def _proc_this_frame(self):

        print('entered _proc_this_frame')
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
        self.draw_done_sig.emit()

    def clear_mbe(self):
        self.cur_curve = None
        self.next_curve = None
        self.proc_this_frame_acc.setEnabled(False)
        self.proc_next_frame_acc.setEnabled(False)
        self.start_comp_acc.setEnabled(False)
        self.iterate_button.setEnabled(False)
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
        old_frame, self.cur_frame = self.cur_frame, i
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

    def start_comp(self):
        self.worker.start_comp(self.cur_curve,
                               self.output_template.path_template,
                               self.cur_frame, self.i_disk_dict)

    def open_file(self):

        fname, _ = QtGui.QFileDialog.getOpenFileName(self,
                    caption='Select cine',
                    dir=self.paths_dict['cine base path'],
                    filter="cine (*.cine)")
        if len(fname) == 0:
            return

        self.fringe_grp_bx.setChecked(False)
        while (self.paths_dict['cine base path'] is None or
                (not (self.paths_dict['cine base path'] ==
                      fname[:len(self.paths_dict['cine base path'])]))):
            print('please set base_dir')
            self.directory_actions['cine base path'].trigger()

        self.prog_bar.show()

        tmp_params = self._get_cur_parametrs()
        self.diag.setEnabled(False)

        path_, fname_ = os.path.split(
            fname[(len(self.paths_dict['cine base path']) + 1):])

        new_cine_fname = backends.FilePath(
            self.paths_dict['cine base path'], path_, fname_)

        self.fname_text.setText('/'.join(new_cine_fname[1:]))
        self.clear_mbe()
        # reset spinners to default values
        # for p in self.spinner_lst:
        #     self.param_spin_dict[p['name']].setValue(p['default'])

        self.fname_text.setText(new_cine_fname[-1])
        self.open_file_sig.emit(new_cine_fname, tmp_params)

    def _get_cur_parametrs(self):
        tmp_dict = {}
        # get parameters out of spin boxes
        for key, sb in self.param_spin_dict.items():

            if sb.isEnabled():
                tmp_dict[key] = sb.value()

        # get toggle switch values
        for key, cb in self.param_checkbox_dict.items():

            tmp_dict[key] = bool(cb.checkState())

        return tmp_dict

    def update_all_params(self):
        tmp_dict = self._get_cur_parametrs()
        # shove them into the worker
        self.worker.update_all_params(tmp_dict)

        # re-draw
        if self.draw_fringes:
            self._proc_this_frame()

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

        frame_selector_group = QtGui.QVBoxLayout()
        fs_form = QtGui.QHBoxLayout()
        fs_form.addWidget(QtGui.QLabel('frame #'))
        fs_form.addWidget(self.frame_spinner)
        fs_form.addWidget(QtGui.QLabel(' of '))
        self.max_cine_label = QtGui.QLabel(str(len(self.worker) - 1))
        fs_form.addWidget(self.max_cine_label)
        fs_stepbox = QtGui.QGroupBox("Frame step")
        fs_sb_rb = QtGui.QHBoxLayout()
        for j in [1, 10, 100, 1000, 10000]:
            tmp_rdo = QtGui.QRadioButton(str(j))
            tmp_rdo.toggled.connect(lambda x, j=j:
                                    self.frame_spinner.setSingleStep(j)
                                    if x else None)
            fs_sb_rb.addWidget(tmp_rdo)
            if j == 1:
                tmp_rdo.toggle()
            pass
        fs_stepbox.setLayout(fs_sb_rb)
        frame_selector_group.addLayout(fs_form)
        frame_selector_group.addWidget(fs_stepbox)
        diag_layout.addLayout(frame_selector_group)

        # play button
        play_button = QtGui.QPushButton('Play')
        self.play_button = play_button
        play_button.setCheckable(True)
        play_button.setChecked(False)
        self.play_button.pressed.connect(self.frame_spinner.stepUp)
        diag_layout.addWidget(play_button)

        meta_data_group = QtGui.QVBoxLayout()

        useful_button = QtGui.QPushButton('useful')
        useful_button.clicked.connect(self.worker.set_useful)
        useless_button = QtGui.QPushButton('useless')
        useless_button.clicked.connect(self.worker.set_useless)
        use_level = QtGui.QHBoxLayout()
        use_level.addWidget(useful_button)
        use_level.addWidget(useless_button)
        meta_data_group.addLayout(use_level)

        md_dict_disp = dict_display('cine md')
        self.worker.md_loadad.connect(md_dict_disp.update)
        meta_data_group.addWidget(md_dict_disp)

        diag_layout.addLayout(meta_data_group)

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
                print(s_type)
                continue

            # properties of spin box
            spin_box.setRange(spin_prams['min'], spin_prams['max'])
            spin_box.setSingleStep(spin_prams['step'])
            spin_box.setValue(spin_prams['default'])
            name = spin_prams['name']
            spin_box.setKeyboardTracking(False)

            # connect it to an action
            spin_box.valueChanged.connect(self.update_params_acc.trigger)

            l_label = QtGui.QLabel(spin_prams['name'])
            if 'tooltip' in spin_prams:
                l_label.setToolTip(spin_prams['tooltip'])
            # if it can be turned on or off
            if spin_prams['togglable']:
                l_checkbox = QtGui.QCheckBox('enable')
                l_checkbox.stateChanged.connect(spin_box.setEnabled)
                l_checkbox.setChecked(spin_prams['default_state'])
                spin_box.setEnabled(spin_prams['default_state'])
                l_checkbox.stateChanged.connect(self.update_params_acc.trigger)
                l_h_layout = QtGui.QHBoxLayout()
                l_h_layout.addWidget(spin_box)
                l_h_layout.addWidget(l_checkbox)
                fringe_cntrls_spins.addRow(l_label, l_h_layout)
            # if it can't
            else:
                fringe_cntrls_spins.addRow(l_label,
                                           spin_box)
            # add the spin box
            self.param_spin_dict[name] = spin_box

        for cb_param in self.toggle_lst:
            l_label = QtGui.QLabel(cb_param['name'])
            if 'tooltip' in cb_param:
                l_label.setToolTip(cb_param['tooltip'])

            l_checkbox = QtGui.QCheckBox('enable')
            l_checkbox.setChecked(cb_param['default'])
            self.param_checkbox_dict[cb_param['name']] = l_checkbox
            l_checkbox.stateChanged.connect(self.update_params_acc.trigger)
            fringe_cntrls_spins.addRow(l_label, l_checkbox)

        # button to grab initial spline
        grab_button = QtGui.QPushButton('Grab Spline')
        grab_button.clicked.connect(self.grab_sf_curve)
        fc_vboxes.addWidget(grab_button)
        # button to process this frame

        ptf_button = QtGui.QPushButton('Process This Frame')
        ptf_button.clicked.connect(self.proc_this_frame_acc.trigger)
        ptf_button.setEnabled(self.proc_this_frame_acc.isEnabled())
        self.proc_this_frame_acc.changed.connect(
            lambda: ptf_button.setEnabled(
                self.proc_this_frame_acc.isEnabled()))

        fc_vboxes.addWidget(ptf_button)

        # button to process next frame
        pnf_button = QtGui.QPushButton('Process Next Frame')
        pnf_button.clicked.connect(self.proc_next_frame_acc.trigger)
        pnf_button.setEnabled(self.proc_next_frame_acc.isEnabled())
        self.proc_next_frame_acc.changed.connect(
            lambda: pnf_button.setEnabled(
                self.proc_next_frame_acc.isEnabled()))

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

        start_comp_bttn = QtGui.QPushButton('Start Computation')
        start_comp_bttn.clicked.connect(self.start_comp_acc.trigger)

        start_comp_bttn.setEnabled(self.start_comp_acc.isEnabled())
        self.start_comp_acc.changed.connect(
            lambda: start_comp_bttn.setEnabled(
                self.start_comp_acc.isEnabled()))

        fc_vboxes.addWidget(start_comp_bttn)
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

        # section for making spline fitting panel
        paths_layout = QtGui.QVBoxLayout()
        path_w = QtGui.QWidget()
        path_w.setLayout(paths_layout)

        for c in self.cap_lst:
            ds = directory_selector(caption=c)
            paths_layout.addWidget(ds)
            self.directory_actions[c].triggered.connect(ds.select_path)
            ds.selected.connect(lambda x, c=c:
                                self.paths_dict.__setitem__(c, x))

        self.disk_widget = numbered_paths(2, self)
        paths_layout.addWidget(self.disk_widget)

        self.output_template = base_path_path_selector('output template')
        paths_layout.addWidget(self.output_template)

        paths_layout.addStretch()
        diag_tool_box.addItem(path_w, "Paths")

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        # create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #

        self.fig = Figure((24, 24), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.axes = self.fig.add_subplot(111,
                                         adjustable='datalim',
                                         aspect='equal')
        #      self.fig.tight_layout(.3, None, None, None)
        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.im = None

        #        tmp = self.worker.get_frame(self.cur_frame)

        #self.im = self.axes.imshow(tmp, cmap='gray', interpolation='nearest')
        #        self.axes.set_aspect('equal')
        #        self.im.set_clim([.5, 1.5])

        self.sf = infra.spline_fitter(self.axes)
        self.sf.disconnect_sf()

        #        self.axes.set_xlim(left=0)
        #self.axes.set_ylim(top=0)
        # this is because images are plotted upside down
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

    @QtCore.Slot(bool, bool)
    def redraw(self, draw_img, draw_tracks):
        if self.im is not None:
            self.im.remove()
        self.im = None
        print('entered redraw')

        # clear the lines we have
        for ln in self.fringe_lines:
            ln.remove()
        self.fringe_lines = []

        mbe = self.worker.get_mbe()
        img = self.worker.get_frame(0)

        if img is None:
            img = 1.5 * np.ones((1, 1))

        extent = [0, img.shape[1], 0, img.shape[0]]

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

        self.frame_spinner.setRange(0, len(self.worker) - 1)
        self.max_cine_label.setText(str(len(self.worker) - 1))
        self.frame_spinner.setValue(0)

        self.redraw_sig.emit(False, False)

    def create_status_bar(self):
        self.status_text = QtGui.QLabel(str(self.cur_frame))

        self.fname_text = QtGui.QLabel('')
        self.fname_text.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
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

        self.start_comp_acc = QtGui.QAction(u'Save Parameters', self)
        self.start_comp_acc.setEnabled(False)
        self.start_comp_acc.triggered.connect(self.start_comp)

        self.open_file_acc = QtGui.QAction(u'Open &File', self)
        self.open_file_acc.triggered.connect(self.open_file)

        self.proc_this_frame_acc = QtGui.QAction('Process this Frame', self)
        self.proc_this_frame_acc.setEnabled(False)
        self.proc_this_frame_acc.triggered.connect(self._proc_this_frame)

        self.proc_next_frame_acc = QtGui.QAction('Process next Frame', self)
        self.proc_next_frame_acc.setEnabled(False)
        self.proc_next_frame_acc.triggered.connect(self._proc_next_frame)

        self.directory_actions = {}
        cta_lst = ['Select ' + x for x in self.cap_lst]
        for cap, cta in zip(self.cap_lst, cta_lst):
            tmp_acc = QtGui.QAction(cta, self)
            self.directory_actions[cap] = tmp_acc

        self.update_params_acc = QtGui.QAction('Update Parameters', self)
        self.update_params_acc.triggered.connect(self.update_all_params)

    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.show_cntrl_acc)
        fileMenu.addAction(self.start_comp_acc)

        fileMenu.addAction(self.open_file_acc)
        pathMenu = menubar.addMenu('&Path')
        for ac in self.directory_actions:
            pathMenu.addAction(ac)

        procMenu = menubar.addMenu('&Process')
        procMenu.addAction(self.proc_this_frame_acc)
        procMenu.addAction(self.proc_next_frame_acc)

    @QtCore.Slot()
    def _play(self):
        QtGui.qApp.processEvents()        # make sure all pending
                                          # events are cleaned up if we
                                          # don't do this, this gets
                                          # hit before the button is
                                          # marked as checked
        if self.play_button.isChecked():
            QtCore.QTimer.singleShot(30, self.frame_spinner.stepUp)

    @property
    def disk_dict(self):
        return self.disk_widget.path_dict

    @property
    def i_disk_dict(self):
        return {v: k for k, v in self.disk_widget.path_dict.items()}
