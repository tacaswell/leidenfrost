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


import time
from collections import namedtuple
import numpy as np
import cPickle

import numpy.fft as fft
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates
import scipy.interpolate as sint
import scipy.interpolate as si

import os

import cine
from trackpy.tracking import Point
from trackpy.tracking import Track
import find_peaks.peakdetect as pd
import trackpy.tracking as pt

import weakref

from . import FilePath
import leidenfrost.db as ldb


class TooFewPointsException(Exception):
    pass


class hash_line_angular(object):
    def __init__(self, dims, bin_width):
        '''
        1D hash table with linked ends for doing the ridge linking
        around a rim


        Parameters
        ----------
        dims : float
            the maximum value of the parameritazation parameter
        bin_width : float
            the width of each bin in units of `dims`

        '''
        full_width = dims
        self.boxes = [[] for j
                      in range(0, int(np.ceil(full_width / bin_width)))]
        self.bin_width = bin_width
        self.bin_count = len(self.boxes)

    def add_point(self, point):
        '''
        :param point: the point object to add, assumed to be a :py:class:`~Point1D_circ`

        Adds a point on the hash line

        Assumes that the point have been properly rationalized 0<`point.phi`< max
        '''
        self.boxes[int(np.floor(point.phi / self.bin_width))].append(point)

    def get_region(self, point, bbuffer):
        '''
        :param point: point to get the region of
        :param bbuffer: the buffer around `point` in the local units

        Gets the region around the point

        Assumes that the point have been properly rationalized 0<`point.phi`< max
        '''
        bbuffer = int(np.ceil(bbuffer / self.bin_width))

        box_indx = int(np.floor(point.phi / self.bin_width))
        tmp_box = []
        for j in range(box_indx - bbuffer, box_indx + bbuffer + 1):
            j = j + self.bin_count if j < 0 else j
            j = j - self.bin_count if j >= self.bin_count else j
            tmp_box.extend(self.boxes[j])
        return tmp_box


class Point1D_circ(Point):
    '''
    Version of :py:class:`Point` for finding fringes

    :ivar q: the parameter for the curve where the
       point is (maps to time in standard tracking)
    :ivar phi: the angle of the point along the
       parametric curve
    :ivar v: any extra values that the point should carry


    '''
    #: the value at which :py:attr:`~Point1D_circ.phi` winds back on it's self
    WINDING = 2 * np.pi

    def __init__(self, q, phi, v=0):
        Point.__init__(self)                  # initialize base class
        self.q = q                            # parametric variable
        self.phi = np.mod(phi, self.WINDING)     # longitudinal value
        # the value at the extrema (can probably drop this)
        self.v = v      # any extra values that the point should carry

    def distance(self, point):
        '''
        :param point: point to give distance to
        :type point: :py:class:`~Point1D_circ`

        Returns the absolute value of the angular distance between
        two points mod :py:attr:`~Point1D_circ.WINDING`'''
        d = np.abs(self.phi - point.phi)
        if d > self.WINDING / 2:
            d = np.abs(self.WINDING - d)
        return d

    def __unicode__(self):
        return 'q: %0.2f, phi: %0.2f' % (self.q, self.phi)

    def __str__(self):
        return unicode(self).encode('utf-8')

    __repr__ = __unicode__


class lf_Track(Track):
    '''
    :param point: The first feature in the track if not  `None`.
    :type point: :py:class:`~trackpy.tracking.Point`

    Derived class from :py:class:`~trackpy.tracking.Track` for working with
    the chevrons that show up in Leidenfrost images.

    :ivar charge: if the fringe is an 'up' or a 'down', set by direction of chevron
    :ivar q: the `q` of the track.  See :py:attr:`Point1D_circ.q`
    :ivar phi:  the `phi` of the track. See :py:attr:`Point1D_circ.phi`
    '''

    def __init__(self, point=None):
        Track.__init__(self, point)
        self.charge = None
        self.q = None
        self.phi = None

    def sort(self):
        '''
        Order the points in the track by :py:attr:`~Point1D_circ.q`
        '''
        self.points.sort(key=lambda x: x.q)

    def plot_trk(self, ax, **kwargs):
        '''
        :param ax: the :py:class:`~matplotlib.axes.Axes` object to plot the track onto
        :type ax: :py:class:`~matplotlib.axes.Axes`

        Plots the track, in q-phi coordinates, onto `ax`.

        `**kwargs` are passed on to `ax.plot`, color will be over-ridden
        by this function.
        '''
        if self.charge is None:
            kwargs['color'] = 'm'
        elif self.charge == 1:
            kwargs['color'] = 'r'
        elif self.charge == -1:
            kwargs['color'] = 'b'
        else:
            kwargs['color'] = 'c'

        ax.plot(*zip(*[(p.q, p.phi) for p in self.points]), **kwargs)

    def plot_trk_img(self, curve, ax, **kwargs):
        '''
        :param curve: the curve used to convert (q,phi) -> (x,y)
        :type curve: :py:class:`~SplineCurve`
        :param ax: the :py:class:`~matplotlib.axes.Axes` object to plot the track onto
        :type ax: :py:class:`~matplotlib.axes.Axes`

        Plots the track, in x-y  coordinates, onto `ax`
        '''
        q, phi = zip(*[(p.q, p.phi) for p in self.points])
        x, y = curve.q_phi_to_xy(q, phi) + 0.5
        mark_charge = False
        if mark_charge:
            if self.charge is None:
                kwargs['marker'] = 's'
            elif self.charge == 1:
                kwargs['marker'] = '^'
            elif self.charge == -1:
                kwargs['marker'] = 'v'
            else:
                kwargs['marker'] = 'o'
            if 'markevery' not in kwargs:
                kwargs['markevery'] = 10
            if 'markersize' not in kwargs:
                kwargs['markersize'] = 7.5
        else:
            if 'marker' not in kwargs:
                kwargs['marker'] = ''

        if bool(self.charge):
            kwargs['lw'] = 2
        else:
            kwargs['lw'] = 1

        if self.charge == 0:
            kwargs['linestyle'] = ':'
            kwargs['lw'] = 1

        if 'picker' not in kwargs:
            kwargs['picker'] = 5
        ln, = ax.plot(x, y, **kwargs)
        ln.payload = weakref.ref(self)
        return ln

    def classify2(self, min_len=5, min_extent=None, straddle=True, **kwargs):
        '''
        :param min_len: the minimum length a track must be to considered
        :param min_extent: the minimum extent in :py:attr:`q` the of the track
           for the track to be considered

        Classification function which sorts out the charge of the track.

        '''
        # this is here because if the length is too small, it will
        # blow up the quad fit
        if min_len < 5:
            min_len = 5
        phi, q = zip(*[(p.phi, p.q) for p in self.points])
        q = np.asarray(q)
        # if the track is less than 25, don't try to classify

        if min_len is not None and len(phi) < min_len:
            self.charge = None
            self.q = None
            self.phi = None
            return

        p_shift = 0
        if np.min(phi) < 0.1 * np.pi or np.max(phi) > 2 * np.pi * .9:
            p_shift = np.pi
            phi = np.mod(np.asarray(phi) + p_shift, 2 * np.pi)

        if min_extent is not None and np.max(phi) - np.min(phi) < min_extent:
            self.charge = None
            self.q = None
            self.phi = None
            return

        # if the track does not straddle the seed curve, probably junk
        if straddle:
            if np.min(q) > 0 or np.max(q) < 0:
                self.charge = None
                self.q = None
                self.phi = None
                return

        ret_beta, R2 = _fit_quad_to_peak(q, phi)
        self.R2 = R2
        self.beta = ret_beta
        r2_thresh = .9
        if R2 < r2_thresh:
            # the quadratic is a bad fit, call this a charge 0 fringe
            prop_c = 0
            prop_q = np.mean(q)
            prop_phi = np.mean(phi)
        else:
            # convert the fit values -> usable values
            prop_c = -int(np.sign(ret_beta[0]))
            prop_q = ret_beta[1]
            prop_phi = ret_beta[2]

        if prop_q < np.min(q) or prop_q > np.max(q):
            # the 'center' in outside of the data we have
            # -> screwy track don't classify
            self.charge = None
            self.q = None
            self.phi = None
            return

        self.charge = prop_c
        self.q = prop_q
        self.phi = np.mod(prop_phi - p_shift, 2 * np.pi)

    def merge_track(self, to_merge_track):
        '''
        :param to_merge_track: track to merge with this one
        :type to_merge_track: :py:class:`~lf_Track`

        Merges `to_merge_track` into this one and re-classifies if needed.
        '''
        pt.Track.merge_track(self, to_merge_track)

        if self.charge is not None:
            self.classify()

    def get_xy(self, curve):
        """
        :param curve: curve to use for (q,phi) -> (x,y) conversion
        :type curve: :py:class:`SplineCurve`

        Returns the (x, y) coordinates of the points on the track
        based on the conversion using curve"""

        return curve.q_phi_to_xy(*zip(*[(p.q, p.phi) for p in self.points]))


class spline_fitter(object):
    def __init__(self, ax, pix_err=1):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.plot([], [], marker='o',
                               linestyle='none', zorder=5)[0]
        self.sp_plot = ax.plot([], [], lw=3, color='r')[0]
        self.pix_err = pix_err
        self.connect_sf()

    def set_visible(self, visible):
        '''sets if the curves are visible '''
        self.pt_plot.set_visible(visible)
        self.sp_plot.set_visible(visible)

    def clear(self):
        '''Clears the points'''
        self.pt_lst = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event',
                                               self.click_event)

    def disconnect_sf(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def remove_pt(self, loc):
        if len(self.pt_lst) > 0:
            self.pt_lst.pop(np.argmin(map(lambda x:
                                          np.sqrt((x[0] - loc[0]) ** 2 +
                                                  (x[1] - loc[1]) ** 2),
                                          self.pt_lst)))

    def redraw(self):
        if len(self.pt_lst) > 5:
            SC = SplineCurve.from_pts(self.pt_lst, pix_err=self.pix_err)
            new_pts = SC.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, 1000))
            center = SC.cntr
            self.sp_plot.set_xdata(new_pts[0])
            self.sp_plot.set_ydata(new_pts[1])
            self.pt_lst.sort(key=lambda x:
                             np.arctan2(x[1] - center[1], x[0] - center[0]))
        else:
            self.sp_plot.set_xdata([])
            self.sp_plot.set_ydata([])
        if len(self.pt_lst) > 0:
            x, y = zip(*self.pt_lst)
        else:
            x, y = [], []
        self.pt_plot.set_xdata(x)
        self.pt_plot.set_ydata(y)

        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points in the format the rest of the
        code expects'''
        return np.vstack(self.pt_lst).T

    def return_SplineCurve(self):
        curve = SplineCurve.from_pts(self.pt_lst, pix_err=self.pix_err)
        print curve.circ
        return curve


def gen_bck_img(fname):
    '''Computes the background image'''
    c_test = cine.Cine(fname)
    bck_img = reduce(lambda x, y: x + y, c_test,
                     np.zeros(c_test.get_frame(0).shape))
    print c_test.len()
    bck_img /= c_test.len()
    # hack to deal with
    bck_img[bck_img == 0] = .001
    return bck_img


WINDOW_DICT = {'flat': np.ones,
               'hanning': np.hanning,
               'hamming': np.hamming,
               'bartlett': np.bartlett,
               'blackman': np.blackman}


def l_smooth(values, window_len=2, window='flat'):
    window_len = window_len * 2 + 1
    s = np.r_[values[-(window_len - 1):], values, values[0:(window_len - 1)]]
    w = WINDOW_DICT[window](window_len)
    values = np.convolve(w / w.sum(), s, mode='valid')

    return values[(window_len // 2):-(window_len // 2)]


class SplineCurve(object):
    mode_param = namedtuple('mode_param', ['n', 'x', 'y'])
    abs_angle = namedtuple('abs_angle', ['abs', 'angle'])

    '''
    A class that wraps the scipy.interpolation objects
    '''
    @classmethod
    def _get_spline(cls, points, pix_err=2, need_sort=True, **kwargs):
        '''
        Returns a closed spline for the points handed in.
        Input is assumed to be a (2xN) array

        =====
        input
        =====

        :param points: the points to fit the spline to
        :type points: a 2xN ndarray or a list of len =2 tuples

        :param pix_err: the error is finding the spline in pixels
        :param need_sort: if the points need to be sorted
            or should be processed as-is

        =====
        output
        =====
        tck
           The return data from the spline fitting
        '''

        if type(points) is np.ndarray:
            # make into a list
            pt_lst = zip(*points)
            # get center
            center = np.mean(points, axis=1).reshape(2, 1)
        else:
            # make a copy of the list
            pt_lst = list(points)
            # compute center
            tmp_fun = lambda x, y: (x[0] + y[0], x[1] + y[1])
            center = np.array(reduce(tmp_fun, pt_lst)).reshape(2, 1)
            center /= len(pt_lst)

        if len(pt_lst) < 5:
            raise TooFewPointsException("not enough points")

        if need_sort:
            # sort the list by angle around center
            pt_lst.sort(key=lambda x: np.arctan2(x[1] - center[1],
                                                 x[0] - center[0]))

        # add first point to end because it is periodic (makes the
        # interpolation code happy)
        pt_lst.append(pt_lst[0])

        # make array for handing in to spline fitting
        pt_array = np.vstack(pt_lst).T
        # do spline fitting

        tck, u = si.splprep(pt_array, s=len(pt_lst) * (pix_err ** 2), per=True)

        return tck

    @classmethod
    def from_pts(cls, new_pts, **kwargs):
        tck = cls._get_spline(new_pts, **kwargs)
        this = cls(tck)
        this.raw_pts = new_pts
        return this

    @classmethod
    def from_hdf(cls, parent_group):
        #        center = parent_group.attrs['center']
        tck = [parent_group.attrs['tck0'],
               parent_group.attrs['tck1'],
               parent_group.attrs['tck2']]
        return cls(tck)

    @classmethod
    def from_pickle_dict(cls, pickle_dict):
        tck = [cPickle.loads(str(pickle_dict[_tk]))
               for _tk in ['tck0', 'tck1', 'tck2']]
        return cls(tck)

    def __init__(self, tck):
        '''A really hacky way of doing different
        '''
        self.tck = tck
        self._cntr = None
        self._circ = None
        self._th_offset = None

    def write_to_hdf(self, parent_group, name=None):
        '''
        Writes out the essential data (spline of central curve) to hdf file.
        '''
        if name is not None:
            curve_group = parent_group.create_group(name)
        else:
            curve_group = parent_group
        curve_group.attrs['tck0'] = self.tck[0]
        curve_group.attrs['tck1'] = np.vstack(self.tck[1])
        curve_group.attrs['tck2'] = self.tck[2]

    @property
    def circ(self):
        '''returns a rough estimate of the circumference'''
        if self._circ is None:
            new_pts = si.splev(np.linspace(0, 1, 1000), self.tck, ext=2)
            self._circ = np.sum(np.sqrt(np.sum(np.diff(new_pts, axis=1) ** 2,
                                               axis=0)))
        return self._circ

    @property
    def cntr(self):
        '''returns a rough estimate of the circumference'''
        if self._cntr is None:
            new_pts = si.splev(np.linspace(0, 1, 1000), self.tck, ext=2)
            self._cntr = np.mean(new_pts, 1)
        return self._cntr

    @property
    def th_offset(self):
        """
        The angle from the y-axis for (x, y) at `phi=0`
        """
        if self._th_offset is None:
            x, y = self.q_phi_to_xy(0, 0) - self.cntr.reshape(2, 1)
            self._th_offset = np.arctan2(y, x)
        return self._th_offset

    @property
    def tck0(self):
        return self.tck[0]

    @property
    def tck1(self):
        return self.tck[1]

    @property
    def tck2(self):
        return self.tck[2]

    @property
    def to_pickle_dict(self):
        return dict((lab, cPickle.dumps(getattr(self, lab)))
                    for lab in ['tck0', 'tck1', 'tck2'])

    def q_phi_to_xy(self, q, phi, cross=None):
        '''Converts q, phi pairs -> x, y pairs.  All other code that
        does this should move to using this so that there is minimal
        breakage when we change over to using additive q instead of
        multiplicative'''
        # make sure data is arrays
        q = np.asarray(q)
        # convert real units -> interpolation units
        phi = np.mod(np.asarray(phi), 2 * np.pi) / (2 * np.pi)
        # get the shapes
        q_shape, phi_shape = [_.shape if (_.shape != () and
                                          len(_) > 1) else None for
                              _ in (q, phi)]

        # flatten everything
        q = q.ravel()
        phi = phi.ravel()
        # sanity checks on shapes
        if cross is False:
            if phi_shape != q_shape:
                raise ValueError("q and phi must have same" +
                                 " dimensions to broadcast")
        if cross is None:
            if ((phi_shape is not None) and (q_shape is not None)
                  and (phi_shape == q_shape)):
                cross = False
            elif q_shape is None:
                cross = False
                q = q[0]
            else:
                cross = True

        x, y = si.splev(phi, self.tck, ext=2)
        dx, dy = si.splev(phi, self.tck, der=1, ext=2)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        nx, ny = dy / norm, -dx / norm

        # if cross, then
        if cross:
            data_out = zip(
                *map(lambda q_: ((x + q_ * nx).reshape(phi_shape),
                                 (y + q_ * ny).reshape(phi_shape)),
                q)
            )
        else:

            data_out = np.vstack([(x + q * nx).reshape(phi_shape),
                                  (y + q * ny).reshape(phi_shape)])

        return data_out

    def fft_filter(self, mode):
        if mode == 0:
            return
        sample_pow = 12
        tmp_pts = si.splev(np.linspace(0, 1, 2 ** sample_pow), self.tck)

        mask = np.zeros(2 ** sample_pow)
        mask[0] = 1
        mask[1:(mode + 1)] = 1
        mask[-mode:] = 1

        new_pts = []
        for w in tmp_pts:
            wfft = fft.fft(w)
            new_pts.append(np.real(fft.ifft(wfft * mask)))

        new_pts = np.vstack(new_pts)

        tck = self._get_spline(new_pts, pix_err=0.05, need_sort=False)

        self.tck = tck

    def draw_to_axes(self, ax, N=1024, **kwargs):
        return ax.plot(*(self.q_phi_to_xy(0, np.linspace(0, 2*np.pi, N))+0.5),
                       **kwargs)

    def curve_shape_fft(self, N=3):
        '''
        Returns the amplitude and phase of the components of the rim curve

        Parameters
        ----------
        self : SplineCurve
            The curve to extract the data from

        n : int
            The maximum mode to extract data for

        Returns
        -------
        ret : list
            [mode_param(n=n, x=abs_angle(x_amp, x_phase),
                        y=abs_angle(y_amp, y_phase)), ...]
        '''
        curve_data = self.q_phi_to_xy(1, np.linspace(0, 2*np.pi, 1000))
        curve_fft = [np.fft.fft(_d) / len(_d) for _d in curve_data]
        return [self.mode_param(n,
                                *[self.abs_angle(2 * np.abs(_cfft[n]),
                                                 np.angle(_cfft[n]))
                               for _cfft in curve_fft])
                for n in range(1, N + 1)]

    def cum_length(self, N=1024):
        '''Returns the cumulative length at N evenly
        sampled points in parameter space

        Parameters
        ----------
        N : int
            Number of points to sample

        Returns
        -------
        ret : ndarray
            An ndarray of length N which is the cumulative distance
            around the rim
        '''
        # turns out you _can_ write un-readable python
        return np.concatenate(([0],
                               np.cumsum(
                                   np.sqrt(
                                       np.sum(
                                           np.diff(
                                               si.splev(
                                                   np.linspace(0, 1, N),
                                                    self.tck,
                                                    ext=2),
                                                axis=1) ** 2,
                                            axis=0)))))


def find_rim_fringes(curve, lfimg, s_width, s_num,
                     smooth_rng=2, oversample=4, *args, **kwargs):
    """
    Does the actual work of finding the fringes on the image
    """

    # a really rough estimate of the circumference
    C = curve.circ

    # sample points at ~ 2/pix
    sample_count = int(np.ceil(C * int(oversample)))

    # get center of curve
    x0, y0 = curve.cntr

    q_vec = np.linspace(-s_width, s_width, s_num).reshape(-1, 1)   # q sampling
    phi_vec = np.linspace(0, 2 * np.pi, sample_count)   # angular sampling

    X, Y = [np.hstack(_) for _ in curve.q_phi_to_xy(q_vec, phi_vec)]

    # compute the region of interest in the image
    R = (np.sqrt(((np.max(X) - np.min(X))) ** 2 +
                 ((np.max(Y) - np.min(Y))) ** 2) * 1.1) / 2

    x_shift = int(x0 - R)
    if x_shift < 0:
        x_shift = 0
    x_lim = int(x0 + R)

    y_shift = int(y0 - R)
    if y_shift < 0:
        y_shift = 0
    y_lim = int(y0 + R)

    # chop down the image
    dlfimg = lfimg[y_shift:y_lim, x_shift:x_lim]

    # shove into
    zp_all = (np.vstack((Y.ravel(), X.ravel())) -
              np.array((y_shift, x_shift)).reshape(2, 1))

    # extract the values at those locations from the image.  The
    # extra flipud is to take a transpose of the points to deal
    # with the fact that the definition of the first direction
    # between plotting and the image libraries is inconsistent.
    zv_all = map_coordinates(dlfimg, zp_all, order=2)
    min_vec = []
    max_vec = []
    theta = np.linspace(0, 2 * np.pi, sample_count)
    for j, q in enumerate(q_vec.reshape(-1)):

        # select out the right region
        zv = zv_all[j * sample_count:(j + 1) * sample_count]
        # smooth the curve
        zv = l_smooth(zv, smooth_rng, 'blackman')

        # find the peaks
        peaks = pd.peakdetect_parabole(zv - np.mean(zv), theta, is_ring=True)
        # extract the maximums
        max_pk = np.vstack(peaks[0])
        # extract the minimums
        min_pk = np.vstack(peaks[1])

        # append to the export vectors
        min_vec.append((q, min_pk))
        max_vec.append((q, max_pk))

    return min_vec, max_vec


def proc_frame(curve, img, s_width, s_num, search_range, min_tlen=5, **kwargs):
    '''new version with different returns'''

    _t0 = time.time()

    miv, mav = find_rim_fringes(curve,
                                img,
                                s_width=s_width,
                                s_num=s_num,
                                **kwargs)

    tim = link_ridges(miv, search_range, **kwargs)
    tam = link_ridges(mav, search_range, **kwargs)

    tim = [t for t in tim if len(t) > min_tlen]
    tam = [t for t in tam if len(t) > min_tlen]

    trk_res = (zip(*[(t.charge, t.phi) for t in tim if t.charge is not None]),
               zip(*[(t.charge, t.phi) for t in tam if t.charge is not None]))

    _t1 = time.time()

    return (_t1 - _t0), trk_res, tim, tam, miv, mav


def link_ridges(vec, search_range, memory=0, **kwargs):
    # generate point levels from the previous steps

    levels = [[Point1D_circ(q, phi, v) for phi, v in pks] for q, pks in vec]

    trks = pt.link_full(levels,
                        2 * np.pi,
                        search_range,
                        hash_cls=hash_line_angular,
                        memory=memory,
                        track_cls=lf_Track)
    for t in trks:
        t.classify2(**kwargs)

    trks.sort(key=lambda x: x.phi)
    return trks


def resample_track(data, pt_num=250, interp_type='linear'):
    '''re-samples the curve on uniform points and averages out tilt
    due to fringe ID error'''

    # get data out
    ch, th = data

    # make sure that the full range is covered
    if th[0] != 0:
        ch = np.concatenate((ch[:1], ch))
        th = np.concatenate(([0], th))
    if th[-1] < 2 * np.pi:
        ch = np.concatenate((ch, ch[:1]))
        th = np.concatenate((th, [2 * np.pi]))

    # set up interpolation
    f = sint.interp1d(th, ch, kind=interp_type)
    # set up new points
    th_new = np.linspace(0, 2 * np.pi, pt_num)
    # get new interpolated values
    ch_new = f(th_new)
    # subtract off mean
    ch_new -= np.mean(ch_new)
    return ch_new, th_new


def construct_corrected_profile(data, th_offset=0):
    '''Takes in [ch, th] and return [delta_h, th].  Flattens '''
    th, ch = data
    th = np.array(th)
    ch = np.array(ch)

    # make negative points positive
    th = np.mod(th + th_offset, 2 * np.pi)
    indx = th.argsort()
    # reverse index for putting everything back in the order it came in
    rindx = indx.argsort()
    # re-order to be monotonic
    th = th[indx]
    ch = ch[indx]
    # sum the charges
    delta_h = np.cumsum(ch - np.hstack((ch[0] - ch[-1], np.diff(ch))) / 2)

    # figure out the miss/match
    miss_cnt = delta_h[-1]
    # the first and last fringes should be off by one, choose up or
    # down based on which side it missed on

    if ch[0] == ch[-1]:
        miss_cnt -= ch[0]

    corr_ln = th * (miss_cnt / (2 * np.pi))
    # add a linear line to make it come back to 0
    delta_h -= corr_ln
    delta_h -= np.min(delta_h)
    return delta_h[rindx], th[rindx]


def setup_spline_fitter(fname, bck_img=None):
    ''' gets the initial path '''
    clims = [.5, 1.5]
    # open the first frame and find the initial circle
    c_test = cine.Cine(fname)
    lfimg = c_test.get_frame(0)
    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
        clims = None
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    im = ax.imshow(lfimg / bck_img, cmap='cubehelix')
    if clims is not None:
        im.set_clim(clims)
    ef = spline_fitter(ax)
    plt.draw()

    return ef


def _fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1]) + b[2]

    x -- locations
    y -- values

    returns (b, R2)

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception('insufficient points handed in ')
    # set up fitting array
    X = np.vstack((x ** 2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y)

    SSerr = np.sum(np.power(np.polyval(beta, x) - y, 2))
    SStot = np.sum(np.power(y - np.mean(y), 2))
    # re-map the returned value to match the form we want
    ret_beta = (beta[0],
                -beta[1] / (2 * beta[0]),
                beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot


def update_average_cache(base_path):
    cine_fnames = []
    for dirpath, dirnames, fnames in os.walk(base_path + '/' + 'leidenfrost'):
        cine_fnames.extend([FilePath(base_path, dirpath[len(base_path)+1:], f)
                            for f in fnames if 'cine' in f])

    db = ldb.LFmongodb()

    for cn in cine_fnames:
        if 'cine' not in cn[-1]:
            continue
        cine_hash = cine.Cine(cn.format).hash
        bck_img = db.get_background_img(cine_hash)
        if bck_img is None:
            bck_img = gen_bck_img(cn.format)
            db.store_background_img(cine_hash, bck_img)
