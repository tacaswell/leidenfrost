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

import fractions

import h5py
import cine
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import leidenfrost.db as db
import leidenfrost.infra as infra


class AnimateHdf(object): 
    '''A class to wrap up all the infrastructure
    needed to make nice movies out of processed hdf files '''
    def __init__(self,hdf_backend,max_frame=None,min_frame=0,step=1):

        # backend
        self.backend = hdf_backend
        self._update_funs = []
        
        # animation limits 
        if max_frame is None or max_frame > len(self.backend) or max_frame < min_frame:
            max_frame = len(self.backend)

        self.max_frame = max_frame
        self.min_frame = min_frame
        self.step = step
        
        # figure setup
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)

        img_ax = fig.add_subplot(gs[:, 0])
        mi = MarkedupImage(img_ax,
                           self._prep_img_limits(),
                           draw_fringes=False)
        

        zprof_ax = fig.add_subplot(gs[0, 1])
        zp = ZProfile(zprof_ax)

        rp_ax = fig.add_subplot(gs[1, 1])
        rp = RProfile(rp_ax)
        
        self._update_funs.append(mi.update)
        self._update_funs.append(zp.update)
        self._update_funs.append(rp.update)
        
        fig.text(0, 1, '/'.join(self.backend.cine_fname[1:]).replace('_','\_'), va='top')
        self.frame_label = fig.text(0,0,'')
                
        gs.tight_layout(fig)

        fig.set_size_inches(9.6,5.4,forward=True)
        # do the animation 
        self.ani =  animation.FuncAnimation(fig,
                                            self._update_figure,
                                            max_frame,
                                            interval=10)

    def save(self,fpath):
        
        writer = animation.writers['ffmpeg'](fps=20,bitrate=30000)
        self.ani.save(fpath,writer=writer,dpi=200)

                 
    def _prep_img_limits(self): 
        '''Figures out what the extent of the
        image should be so the drop does not walk out of image '''
        frame_step = 100
        max_x = -np.inf
        max_y = -np.inf
        min_x = np.inf
        min_y = np.inf
        
        for j in range(self.min_frame,self.max_frame,frame_step):
            x,y = self.backend.get_frame(j, False, False).curve.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, 1000))
            

            if np.max(x) > max_x:
                max_x = np.max(x)
            if np.max(y) > max_y:
                max_y = np.max(y)

            if np.min(x) < min_x:
                min_x = np.min(x)
            if np.min(y) < min_y:
                min_y = np.min(y)

        return [min_x * .9, max_x * 1.1, min_y * .9, max_y * 1.1]
        
    def _update_figure(self,j):
        '''Updates the figure, wraps up all of the axes '''
        res = ()
        for f in self._update_funs:
            res = res + f(self.backend, j)
        self.frame_label.set_text('frame %d' % j)
        return res + (self.frame_label, )

def format_frac(fr):
    sp = str(fr).split('/')
    if len(sp) == 1:
        return sp[0]
    else:
        return r'$\frac{%s}{%s}$' % tuple(sp)
            
class ZProfile(object):
    def __init__(self, ax):
        self.ax = ax
        self.prof_ln, = self.ax.plot([], [],'.k')
        self.ax.set_xlim([0,2 * np.pi])
        self.ax.set_ylim([-10,10])
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('$\Delta h$ [um]')
        frac_size = 4
        step = fractions.Fraction(1, frac_size)
        ax.set_xticks([np.pi * j * step for j in range(2 * frac_size + 1)])
        ax.set_xticklabels([format_frac(j * step) + '$\pi$' for j in range(2 * frac_size + 1)])


    
    def update(self, backend, j, delta_h = 0):
        mbe = backend.get_frame(j, False, False)

        th = np.hstack([mbe.res[0][1], mbe.res[1][1]]) 
        ch = np.hstack([mbe.res[0][0], mbe.res[1][0]]) 
        dh,th_new = infra.construct_corrected_profile((th,ch)) 
        dh -= np.mean(dh)
        dh += delta_h
        dh *= (.543 / 4)

        self.prof_ln.set_xdata(th_new)
        self.prof_ln.set_ydata(dh)
        return (self.prof_ln, )

class RProfile(object):
    def __init__(self,ax):
        self.ax = ax
        ax.cla()
        ax.set_xlim([0,2 * np.pi])
        ax.set_ylim(-50,50)
        self.th = np.linspace(0, 2 * np.pi,1000)
        th = self.th
        self.line, = ax.plot(th, np.zeros(th.shape), 'b-')

        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$R - \langle R \rangle$ [pix]')
        step = fractions.Fraction(1,4)
        ax.set_xticks([np.pi * j * step for j in range(9)])
        ax.set_xticklabels([format_frac(j * step) + '$\pi$' for j in range(9)])

    def update(self, backend, j):
        mbe = backend.get_frame(j + 1, False, False)
        q,phi = mbe.res[0][2], mbe.res[0][1]
        new_curve = infra.SplineCurve.from_pts(np.vstack(mbe.curve.q_phi_to_xy(q, phi)),
                                               **mbe.params)
        new_curve.fft_filter(mbe.params['fft_filter'])
        XY  = np.vstack(new_curve.q_phi_to_xy(0,self.th))

        center = np.mean(XY, 1).reshape(2,1)

        XY -= center

        R = np.hypot(*XY)

        R -= np.mean(R)

        self.line.set_ydata(R)

        return (self.line, )
        
        
        
    
class MarkedupImage(object):
    '''Class to encapsulate drawing the image the shape + (maybe) fringes on it'''
    def __init__(self, ax, limits, draw_fringes=False, draw_curve=True, clims=(.5, 1.5)):
        '''
        :param ax: `matplotlib.Axes` object to draw into
        :param limits: [x_min, x_max, y_min, y_max]
        :param draw_finges: if the raw fringes should be drawn
        :param draw_curve: if the center line should be drawn
        :param clims: len=2, limits for color axis

        '''
        self.ax = ax
        ax.cla()               #just to be safe
        # # set up for the image + fringe lines
        self.img = self.ax.imshow(np.zeros((limits[1] - limits[0],
                                            limits[3] - limits[2])),
                                            extent=limits, cmap='gray',
                                            origin='lower')
        self.img.set_clim(clims)
        self.ax.set_xlim(limits[:2])
        self.ax.set_ylim(limits[2:])
        self.ax.autoscale(False)
        self.ax.set_aspect('equal')
        self.fringe_lines = []

        self.draw_finges = draw_fringes
        self.draw_curve = draw_curve
        self.limits = limits

        self.fringe_lines = []

        self.center_line = None
        if self.draw_curve:
            self.center_line, = ax.plot([],[])
    
    def update(self,backend,j):
        '''Updates the axes with the image'''

        # deal with image
        mbe = backend.get_frame(j, raw=self.draw_finges, get_img=True)
        xmn, xmx, ymn, ymx = self.limits
        self.img.set_data(mbe.img[ymn:ymx, xmn:xmx])

        
        # deal with fringes
        
        for l in self.fringe_lines:
            l.remove()
        self.fringe_lines = []
        if self.draw_finges:
            self.fringe_lines.extend(
                mbe.ax_plot_tracks(self.img_ax,
                                   min_len=0,
                                   all_tracks=False)
                                   )

        center_line_lst = []
        if self.draw_curve:
            q = np.hstack([mbe.res[0][2], mbe.res[1][2]]) 
            phi = np.hstack([mbe.res[0][1], mbe.res[1][1]]) 
            q,phi = mbe.res[0][2], mbe.res[0][1]
            new_curve = infra.SplineCurve.from_pts(np.vstack(mbe.curve.q_phi_to_xy(q, phi)),
                                                   **mbe.params)
            new_curve.fft_filter(mbe.params['fft_filter'])            
            x,y = new_curve.q_phi_to_xy(0,np.linspace(0, 2 * np.pi, 1000))
            if self.center_line:
                self.center_line.set_xdata(x + .5)
                self.center_line.set_ydata(y + .5)
            else:
                self.center_line = ax.plot(x,y)

            center_line_lst.append(self.center_line)
        
        return tuple(self.fringe_lines) + (self.img, ) + tuple(center_line_lst)

        
def animate_profile(data_iter):
    def ln_helper(data, line, yscale=1, xscale=1):
        ch, th = data

        th = np.asarray(th)
        th = np.mod(th, 2 * np.pi)
        ch = np.asarray(ch)
        indx = th.argsort()

        # re-order to be monotonic
        th = th[indx]
        ch = ch[indx]

        ch = np.cumsum(ch)
        miss_cnt = ch[-1]
        corr_ln = th * (miss_cnt / (2 * np.pi))
        ch -= corr_ln
        #        ch = concatenate([ch, ch[:50]])
        ch -= np.mean(ch)
        #        th = concatenate([th, asarray(th[:50]) + 2*pi])

        line.set_xdata(th * xscale)
        line.set_ydata(ch * yscale)

        return miss_cnt

    def update_lines(mbe, lines, txt, miss_txt):
        txt.set_text('%0.3e s' % (mbe.frame_number * 1 / 2900))

        circ = mbe.curve.circumference() / (2 * np.pi) * 11

        min_t, max_t = mbe.res
        miss_min = ln_helper(min_t[:2], lines[0], yscale=.543 / 2, xscale=circ)

        miss_max = ln_helper(max_t[:2], lines[1], yscale=.543 / 2, xscale=circ)

        miss_bth = ln_helper([tuple(t) + tuple(tt) for
                                t, tt in zip(max_t[:2], min_t[:2])],
                              lines[2],
                              yscale=.543 / 4,
                              xscale=circ)

        miss_txt.set_text("min miss: %(i)d max miss: %(a)d" % {'i': miss_min,
                                                               'a': miss_max})
        return (txt, miss_txt) + lines

    fig = plt.figure()
    ax = fig.add_subplot(111)
    tmp_ch = np.cumsum(data_iter[0].res[0][0])
    tmp_ch -= np.mean(tmp_ch)

    lim = np.max(np.abs(tmp_ch)) * (.543 / 4) * 2
    circ = data_iter[0].curve.circumference() * 11
    line1, = ax.plot([0, circ], [-lim, lim], 'o-r', label='mins')
    line2, = ax.plot([0, circ], [-lim, lim], 'o-b', label='maxes')
    line3, = ax.plot([0, circ], [-lim, lim], 'o-g', label='maxes')
    ax.set_ylabel(r' height [$\mu$ m]')
    ax.set_xlabel(r' position [$\mu$ m]')
    fr_num = ax.text(.05, -lim * .95, '')
    miss_txt = ax.text(.05, lim * .95, '')

    # legend(loc = 0)
    prof_ani = animation.FuncAnimation(fig,
                                       update_lines,
                                       data_iter,
                                       fargs=((line2, line1, line3),
                                              fr_num, miss_txt),
                                       interval=100)
    return prof_ani



