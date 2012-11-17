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

# do this to make me learn where stuff is and to make it easy to switch to PyQt later
import PySide.QtCore as QtCore
import PySide.QtGui as QtGui

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np

import leidenfrost.infra as infra
import cine






class LFWorker(QtCore.QObject):
    frame_proced = QtCore.Signal()
    
    def __init__(self,cine_fname,bck_img,params,parent=None):
        QtCore.QObject.__init__(self,parent)

        self.process_backend = infra.ProcessBackend.from_args(cine_fname,bck_img = bck_img,**params)
        
        self.mbe = None
        self.next_curve = None

    @QtCore.Slot(int,infra.SplineCurve)
    def proc_frame(self,ind,curve):
        
        self.mbe,self.next_curve = self.process_backend.process_frame(ind,curve)
        self.frame_proced.emit()

    def get_mbe(self):
        return self.mbe

    def get_next_curve(self):
        return self.next_curve





    def update_param(self,key,val):
        self.process_backend.update_param(key,val)

    def get_frame(self,ind):
        tmp = self.process_backend.get_frame(ind)
        return np.asarray(tmp,dtype=np.float)
    def clear(self):
        self.mbe = None
        self.next_curve = None
    def __len__(self):
        return len(self.process_backend)

    @QtCore.Slot(infra.FilePath,dict)
    def set_new_fname(self,cine_fname,params):
        print cine_fname,params
        self.clear()
        self.process_backend = infra.ProcessBackend.from_args(cine_fname,bck_img = None,**params)
        self.frame_proced.emit()

    
class LFGui(QtGui.QMainWindow):
    proc = QtCore.Signal(int,infra.SplineCurve)
    open_file_sig  = QtCore.Signal(infra.FilePath,dict)
    kill_thread = QtCore.Signal()
    spinner_lst = [
               {'name':'s_width','min':0,'max':50,'step':.5,'prec':1,'type':np.float,'default':10},
               {'name':'s_num','min':1,'max':500,'step':1,'type':np.int,'default':100},
               {'name':'search_range','min':.001,'max':2*np.pi,'step':.005,'prec':3,'type':np.float,'default':.01},
               {'name':'memory','min':0,'max':500,'step':1,'type':np.int,'default':0},
               {'name':'pix_err','min':0,'max':5,'step':.1,'prec':1,'type':np.float,'default':2},
               {'name':'mix_in_count','min':0,'max':100,'step':1,'type':np.int,'default':10},
               ]
    
    def __init__(self,cine_fname,bck_img=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Fringe Finder')
        self.cine_fname = cine_fname
        self.base_dir = cine_fname[0]
        self.cur_frame = 0

        self.refresh_lines_flg = True
        self.refresh_img = True        
        self.draw_fringes = False
        
        default_params = dict((d['name'],d['default']) for d in LFGui.spinner_lst)

        self.base_path = cine_fname[0]

        self.thread = QtCore.QThread(parent=self)


        self.worker = LFWorker(cine_fname,bck_img,default_params,parent=None)
        self.worker.moveToThread(self.thread)

        

        self.cur_curve = None
        self.next_curve = None


        self.fringe_lines = []



        self.create_main_frame()
        self.create_actions()
        self.create_menu_bar()
        self.create_diag()
        self.create_status_bar()

        self.on_draw()
        self.worker.frame_proced.connect(self.on_draw)        
        self.proc.connect(self.worker.proc_frame)    
        self.open_file_sig.connect(self.worker.set_new_fname)    
        self.kill_thread.connect(self.thread.quit)

        self.show()
        self.thread.start()
        #        self.thread.exec_()
        QtGui.qApp.exec_()
        

    def grab_sf_curve(self):
        try:
            self.cur_curve = self.sf.return_SplineCurve()
            self.ptf_button.setEnabled(True)
            self.save_param_acc.setEnabled(True)
        except:
            print 'spline fitter not ready'
            self.cur_curve = None
        
    def update_param(self,key,val):
        self.worker.update_param(key,val)
        if self.draw_fringes:
            self._proc_this_frame()
            
    def _proc_this_frame(self):

        
        self.prog_bar.show()
        self.diag.setEnabled(False)
        self.draw_fringes = True
                
        self.pnf_button.setEnabled(True)
        self.iterate_button.setEnabled(True)
        self.refresh_lines_flg = True


        self.proc.emit(self.cur_frame,self.cur_curve)

        
    def _proc_next_frame(self):
        self.frame_spinner.setValue(self.cur_frame + 1)
        self.diag.setEnabled(False)
        self._proc_this_frame()

    def set_fringes_visible(self,i):
        self.draw_fringes = bool(i)
        self.refresh_lines_flg = True
        self.on_draw()
        
    def on_draw(self,refresh_lines=True,refresh_img=True):
        """ Redraws the figure
        """
        self.draw_fringes_ck.setChecked(self.draw_fringes)
        print 'entered on_draw'
        print refresh_lines,refresh_img,self.draw_fringes
        # update the image
        if refresh_img:
            self.im.set_data(self.worker.get_frame(self.cur_frame))
            self.refresh_img = False
        # if we need to update the lines
        if refresh_lines:
            # clear the lines we have
            for ln in self.fringe_lines:
                ln.remove()
            # nuke all those objects
            self.fringe_lines = []
        
            if self.draw_fringes:
                # if we should draw new ones, do so
                
                mbe = self.worker.get_mbe()   # grab new mbe from thread object
                self.next_curve = self.worker.get_next_curve()   # grab new next curve
                if self.draw_fringes and mbe is not None:
                    self.fringe_lines.extend(mbe.ax_plot_tracks(self.axes,min_len = 15,all_tracks = False))
                    self.fringe_lines.extend(mbe.ax_draw_center_curves(self.axes))

        self.canvas.draw()
        self.status_text.setNum(self.cur_frame)
        self.prog_bar.hide()
        self.diag.setEnabled(True)

    def clear_mbe(self):
        self.cur_curve = None
        self.next_curve = None
        self.pnf_button.setEnabled(False)
        self.ptf_button.setEnabled(False)
        self.save_param_acc.setEnabled(False)
        self.iterate_button.setEnabled(False)
        self.refresh_lines_flg = True
        self.draw_fringes_ck.setChecked(False)

        
    def set_spline_fitter(self,i):
        if i:
            self.sf.connect_sf()
            # if we can screw with it, make it visible
            self.sf_show.setChecked(True)
        else:
            self.sf.disconnect_sf()
            
        self.on_draw()
            
    def set_spline_fitter_visible(self,i):
        self.sf.set_visible(i)
        # if we can't see it, don't let is screw with it
        if not bool(i):
            self.sf_check.setChecked(False)
        self.on_draw()
                
    def set_cur_frame(self,i):
        old_frame = self.cur_frame
        self.cur_frame = i
        self.refresh_lines_flg = True        
        self.refresh_img = True
        if old_frame  == self.cur_frame - 1:
            self.cur_curve = self.next_curve
            if self.draw_fringes:
                self._proc_this_frame()
            else:
                self.on_draw()
        else:
            self.refresh_img = True
            self.draw_fringes = False
            self.worker.clear()
            self.on_draw()
            
        

    def iterate_frame(self):
        self.cur_curve = self.next_curve
        self._proc_this_frame()
        

    def clear_spline(self):
        self.sf.clear()


    def show_cntrls(self):
        self.diag.show()


    def save_config(self):
        fname,_ = QtGui.QFileDialog.getSaveFileName(self,caption='Save File',dir = '/'.join(self.worker.process_backend.cine_fname[:1]))
        if len(fname) > 0:
                self.worker.process_backend.gen_stub_h5(fname,self.cur_curve)

    def set_base_dir(self):
        base_dir = QtGui.QFileDialog.getExistingDirectory(self,caption='Base Directory',dir = self.base_dir)
        if len(base_dir) > 0:
            self.base_dir = base_dir

            
    def open_file(self):
        fname,_ = QtGui.QFileDialog.getOpenFileName(self,caption='Save File',dir = self.base_dir)
        if len(fname) == 0:
            return
        
        while not self.base_dir == fname[:len(self.base_dir)]:
            print 'please set base_dir'
            self.set_base_dir()
                    
        self.prog_bar.show()
        self.diag.setEnabled(False)

        path_,fname_ = os.path.split(fname[(len(self.base_dir)+1):])
        new_cine_fname = infra.FilePath(self.base_dir,path_,fname_)
        self.fname_text.setText('/'.join(new_cine_fname[1:])
        self.clear_mbe()
        default_params = dict((d['name'],d['default']) for d in LFGui.spinner_lst)
        self.open_file_sig.emit(new_cine_fname,default_params)
        

        
    def create_diag(self):
        # make top level stuff
        self.diag = QtGui.QDockWidget('controls',parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,self.diag)
        diag_widget = QtGui.QWidget(self.diag)
        self.diag.setWidget(diag_widget)
        diag_layout = QtGui.QVBoxLayout()
        diag_widget.setLayout(diag_layout)
        

        # frame number lives on top
        self.frame_spinner = QtGui.QSpinBox()
        self.frame_spinner.setRange(0,len(self.worker)-1)
        self.frame_spinner.valueChanged.connect(self.set_cur_frame)
        fs_form = QtGui.QFormLayout()
        fs_form.addRow(QtGui.QLabel('frame #'),self.frame_spinner)


        diag_layout.addLayout(fs_form)
        
        # tool box for all the controls
        diag_tool_box = QtGui.QToolBox()
        diag_layout.addWidget(diag_tool_box)        

        
        # section for dealing with fringe finding 
        

        fringe_cntrls_w = QtGui.QWidget()   # the widget to shove into the toolbox
        fc_vboxes = QtGui.QVBoxLayout()     # vbox layout for this panel
        fringe_cntrls_w.setLayout(fc_vboxes)   # set the widget layout
        diag_tool_box.addItem(fringe_cntrls_w,"Fringe Finding Settings") # add to the tool box

        
        fringe_cntrls_spins = QtGui.QFormLayout()   # from layout to hold the spinners

        fc_vboxes.addLayout(fringe_cntrls_spins)   # add spinner layout

        # fill the spinners
        for spin_prams in LFGui.spinner_lst:

            s_type = np.dtype(spin_prams['type']).kind

            if s_type == 'i':
                spin_box = QtGui.QSpinBox(parent=self)
            elif s_type== 'f':
                spin_box = QtGui.QDoubleSpinBox(parent=self)
                spin_box.setDecimals(spin_prams['prec'])
            else:
                print s_type
                continue
            
            spin_box.setRange(spin_prams['min'],spin_prams['max'])
            spin_box.setSingleStep(spin_prams['step'])
            spin_box.setValue(spin_prams['default'])
            name = spin_prams['name']


            spin_box.valueChanged.connect(self._gen_update_closure(name))
            fringe_cntrls_spins.addRow(QtGui.QLabel(spin_prams['name']),spin_box)

        # button to grab initial spline
        grab_button = QtGui.QPushButton('Grab Spline')
        grab_button.clicked.connect(self.grab_sf_curve)
        fc_vboxes.addWidget(grab_button)
        # button to process this frame
        ptf_button = QtGui.QPushButton('Process This Frame')
        ptf_button.clicked.connect(self._proc_this_frame)
        ptf_button.setEnabled(False)
        fc_vboxes.addWidget(ptf_button)
        self.ptf_button = ptf_button
        # button to process next frame
        pnf_button = QtGui.QPushButton('Process Next Frame')
        pnf_button.clicked.connect(self._proc_next_frame)
        pnf_button.setEnabled(False)
        fc_vboxes.addWidget(pnf_button)
        self.pnf_button = pnf_button
        # nuke tracking data
        
        clear_mbe_button = QtGui.QPushButton('Clear fringes')
        clear_mbe_button.clicked.connect(self.clear_mbe)
        fc_vboxes.addWidget(clear_mbe_button)


        self.draw_fringes_ck = QtGui.QCheckBox('Draw Fringes')
        self.draw_fringes_ck.stateChanged.connect(self.set_fringes_visible)
        self.draw_fringes_ck.setChecked(self.draw_fringes)
        fc_vboxes.addWidget(self.draw_fringes_ck)


        
        iterate_button = QtGui.QPushButton('Iterate fringes')
        iterate_button.clicked.connect(self.iterate_frame)
        iterate_button.setEnabled(False)
        fc_vboxes.addWidget(iterate_button)
        self.iterate_button = iterate_button
        
        
        save_param_bttn = QtGui.QPushButton('Save Configuration')
        save_param_bttn.clicked.connect(self.save_param_acc.trigger)

        save_param_bttn.setEnabled(self.save_param_acc.isEnabled())
        self.save_param_acc.changed.connect(lambda : save_param_bttn.setEnabled(self.save_param_acc.isEnabled()))
        

        
        fc_vboxes.addWidget(save_param_bttn)

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
        
        clear_spline_button = QtGui.QPushButton('Clear Supline')
        clear_spline_button.clicked.connect(self.clear_spline)
        spline_cntrls.addWidget(clear_spline_button)


        
        diag_tool_box.addItem(spline_cntrls_w,"Manual Spline Fitting")
        
    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        # create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #

        self.fig = Figure((24,24))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        tmp = self.worker.get_frame(self.cur_frame)

        self.im = self.axes.imshow(tmp,cmap='cubehelix',interpolation = 'nearest')
        self.axes.set_aspect('equal')
        self.im.set_clim([.5,1.5])

        self.sf = infra.spline_fitter(self.axes)
        self.sf.disconnect_sf()

        self.axes.set_xlim(left=0)
        self.axes.set_ylim(top=0)         # this is because images are plotted upside down
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

    def _gen_update_closure(self,name):
        return lambda x :self.update_param(name,x)
    
    def create_status_bar(self):
        self.status_text = QtGui.QLabel(str(self.cur_frame))
        self.fname_text =  QtGui.QLabel('/'.join(self.cine_fname[1:]))
        self.statusBar().addWidget(self.status_text)
        self.prog_bar = QtGui.QProgressBar()
        self.prog_bar.setRange(0,0)
        self.prog_bar.hide()
        self.statusBar().addWidget(self.prog_bar, 1)
        self.statusBar().addPermanentWidget(fname_text)

    def closeEvent(self,ce):
        self.kill_thread.emit()
        #        QtGui.qApp.quit()
        #        self.thread.quit()
        self.diag.close()
        QtGui.QMainWindow.closeEvent(self,ce)

    def create_actions(self):
        self.show_cntrl_acc = QtGui.QAction(u'show controls',self)
        self.show_cntrl_acc.triggered.connect(self.show_cntrls)

        self.save_param_acc = QtGui.QAction(u'Save Parameters',self)
        self.save_param_acc.setEnabled(False)
        self.save_param_acc.triggered.connect(self.save_config)

        self.set_base_dir_acc =  QtGui.QAction(u'Select base dir',self)
        self.set_base_dir_acc.triggered.connect(self.set_base_dir)

        
        self.open_file_acc =  QtGui.QAction(u'Open &File',self)
        self.open_file_acc.triggered.connect(self.open_file)

    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.show_cntrl_acc)
        fileMenu.addAction(self.save_param_acc)
        fileMenu.addAction(self.set_base_dir_acc)
        fileMenu.addAction(self.open_file_acc)
