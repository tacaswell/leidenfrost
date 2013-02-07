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
import collections

import os
import itertools

import numpy as np
import numpy.linalg as nl
import numpy.fft as fft
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates
import scipy.interpolate as sint
import scipy.interpolate as si

import h5py
import cine
from trackpy.tracking import Point
from trackpy.tracking import Track
import find_peaks.peakdetect as pd
import trackpy.tracking as pt

import shutil
import copy
import weakref


import leidenfrost.db as db

FilePath = collections.namedtuple('FilePath', ['base_path', 'path', 'fname'])
HdfBEPram = collections.namedtuple('HdfBEPram', ['raw', 'get_img'])


class hash_line_angular(object):
    '''1D hash table with linked ends for doing the ridge linking
    around a rim'''
    def __init__(self, dims, bin_width):
        '''
        :param dims: the maximum value of the parameritazation parameter
        :param bin_width: the width of each bin in units of `dims`
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
            j = j - self.bin_count if j > self.bin_count else j
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
        x, y = curve.q_phi_to_xy(q, phi)
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
            kwargs['lw'] = 3
        else:
            kwargs['lw'] = 1

        if self.charge == 0:
            kwargs['linestyle'] = ':'
            kwargs['lw'] = 2

        if 'picker' not in kwargs:
            kwargs['picker'] = 5
        ln, = ax.plot(x, y, **kwargs)
        ln.payload = weakref.ref(self)
        return ln

    def classify2(self, min_len=None, min_extent=None, straddle=True, **kwargs):
        '''
        :param min_len: the minimum length a track must be to considered
        :param min_extent: the minimum extent in :py:attr:`q` the of the track
           for the track to be considered

        Classification function which sorts out the charge of the track.

        '''
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
        self.phi = prop_phi - p_shift

    def mean_phi(self):
        '''
        Sets :py:attr:`phi` to be the average :py:attr:`phi` of the track

        :deprecated: this is a stupid way of doing this.
        '''
        self.phi = np.mean([p.phi for p in self.points])
        raise PendingDeprecationWarning()

    def mean_q(self):
        '''
        Sets :py:attr:`q` to be the average :py:attr:`q` of the track

        :deprecated: this is a stupid way of doing this.
        '''
        self.q = np.mean([p.q for p in self.points])
        raise PendingDeprecationWarning()

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
        self.pt_plot = ax.plot([], [], marker='x', linestyle='-')[0]
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
            new_pts = SC.get_xy_samples(1000)
            center = np.mean(new_pts, 1).ravel()
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
        print curve.circumference()
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


def _write_frame_tracks_to_file(parent_group,
                                t_min_lst,
                                t_max_lst,
                                curve,
                                md_args={}):
    '''
    Takes in an hdf object and creates the following data sets in
    `parent_group`


    raw_data_{min, max}
        a 2xN array with columns {ma, phi} which
        is all of the tracked points for this frame

    raw_track_md_{min, max}
        a 2x(track_count) array with columns
        {track_len, start_index} Start index refers to
        raw_data_{}

    trk_res_{min, max}
        a 2x(track_count) array with columns {charge, phi}

    everything in md_args is shoved into the group level attributes

    ======
    INPUT
    ======
    `parent_group`
        h5py group object.  Should not contain existing data sets with
        the same names

    `t_min_lst`
        an iterable of the tracks for the minimums in the frame.

    `t_max_lst`
        an iterable of the tracks for the minimums in the frame

    `md_args`
        a dictionary of meta-data to be attached to the group
    '''

    # names
    raw_data_name = 'raw_data_'
    raw_track_md_name = 'raw_track_md_'
    trk_res_name = 'trk_res_'
    name_mod = ('min', 'max')
    write_raw_data = True
    write_res = True
    curve.write_to_hdf(parent_group)
    for key, val in md_args.items():
        try:
            parent_group.attrs[key] = val
        except TypeError:
            print 'key: ' + key + ' can not be gracefully shoved into '
            print 'an hdf object, please reconsider your life choices'
    for t_lst, n_mod in zip((t_min_lst, t_max_lst), name_mod):
        if write_raw_data:
            # get total number of points
            pt_count = np.sum([len(t) for t in t_lst])
            # arrays to accumulate data into
            tmp_raw_data = np.zeros((pt_count, 2))
            tmp_raw_track_data = np.zeros((len(t_lst), 2))
            tmp_indx = 0
            #            print pt_count
            for i, t in enumerate(t_lst):
                t_len = len(t)
                d_slice = slice(tmp_indx, (tmp_indx + t_len))
                # shove in raw data
                tmp_raw_data[d_slice, 0] = np.array([p.q for p in t])
                tmp_raw_data[d_slice, 1] = np.array([p.phi for p in t])
                # shove in raw track data
                tmp_raw_track_data[i, :] = (t_len, tmp_indx)
                # increment index
                tmp_indx += t_len

            # create dataset and shove in data
            parent_group.create_dataset(raw_data_name + n_mod,
                                        tmp_raw_data.shape,
                                        np.float,
                                        compression='szip')
            parent_group[raw_data_name + n_mod][:] = tmp_raw_data

            parent_group.create_dataset(raw_track_md_name + n_mod,
                                        tmp_raw_track_data.shape,
                                        np.float,
                                        compression='szip')
            parent_group[raw_track_md_name + n_mod][:] = tmp_raw_track_data

        if write_res:
            good_t_lst = [t for t in t_lst if
                          t.charge is not None and t.charge != 0]
            tmp_track_res = np.zeros((len(good_t_lst), 3))

            # shove in results data
            for i, t in enumerate(good_t_lst):
                tmp_track_res[i, :] = (t.charge, t.phi, t.q)

            parent_group.create_dataset(trk_res_name + n_mod,
                                        tmp_track_res.shape,
                                        np.float,
                                        compression='szip')
            parent_group[trk_res_name + n_mod][:] = tmp_track_res


def _read_frame_tracks_from_file_raw(parent_group):
    '''
    inverse operation to `_write_frame_tracks_to_file`

    Reads out all of the raw data

    '''

    # names
    raw_data_name = 'raw_data_'
    raw_track_md_name = 'raw_track_md_'
    name_mod = ('min', 'max')
    trk_lsts_tmp = []
    for n_mod in name_mod:
        tmp_raw_data = parent_group[raw_data_name + n_mod][:]
        tmp_track_data = parent_group[raw_track_md_name + n_mod][:]
        t_lst = []
        for t_len, strt_indx in tmp_track_data:
            tmp_trk = lf_Track()
            for ma, phi in tmp_raw_data[strt_indx:(strt_indx + t_len), :]:
                tmp_trk.add_point(Point1D_circ(ma, phi))
            tmp_trk.classify2()
            t_lst.append(tmp_trk)
        trk_lsts_tmp.append(t_lst)

    return trk_lsts_tmp


def _read_frame_tracks_from_file_res(parent_group):
    '''
    Only reads out the charge and location of the tracks, not all of
    their points '''

    # names
    trk_res_name = 'trk_res_'
    name_mod = ('min', 'max')
    res_lst = []
    for n_mod in name_mod:
        tmp_trk_res = parent_group[trk_res_name + n_mod][:]
        tmp_charge = tmp_trk_res[:, 0]
        tmp_phi = tmp_trk_res[:, 1]
        tmp_q = tmp_trk_res[:, 2]
        res_lst.append((tmp_charge, tmp_phi, tmp_q))

    return res_lst


class ProcessBackend(object):
    req_args_lst = ['search_range', 's_width', 's_num', 'pix_err']

    def __len__(self):
        if self.cine_ is not None:
            return len(self.cine_)
        else:
            return 0

    def __init__(self):
        self.params = {}        # the parameters to feed to proc_frame

        self.cine_fname = None               # file name
        self.cine_ = None                    # the cine object

        self.bck_img = None       # back ground image for normalization
        try:
            self.db = db.LFmongodb()  # hard code the mongodb
        except:
            # this eats _ALL_ exceptions
            self.db = None

    @classmethod
    def _verify_params(cls, param, extra_req=None):
        if extra_req is None:
            extra_req = []
        for s in cls.req_args_lst + extra_req:
            if s not in param:
                raise Exception("missing required argument %s" % s)

    @classmethod
    def from_hdf_file(cls, cine_base_path, h5_fname):
        ''' Sets up object to process data based on MD in an hdf file.
        '''
        self = cls()
        tmp_file = h5py.File('/'.join(h5_fname), 'r')
        keys_lst = tmp_file.attrs.keys()
        lc_req_args = ['tck0', 'tck1', 'tck2']
        h5_req_args = ['cine_path', 'cine_fname']
        cls._verify_params(keys_lst, extra_req=(lc_req_args + h5_req_args))

        self.params = dict(tmp_file.attrs)

        for k in lc_req_args:
            del self.params[k]

        self.cine_fname = FilePath(cine_base_path,
                                   self.params.pop('cine_path'),
                                   self.params.pop('cine_fname'))
        self.cine_ = cine.Cine('/'.join(self.cine_fname))

        if 'bck_img' in tmp_file.keys():
            self.bck_img = tmp_file['bck_img'][:]
        else:
            self.bck_img = gen_bck_img('/'.join(self.cine_fname))

        seed_curve = SplineCurve.from_hdf(tmp_file)

        tmp_file.close()

        return self, seed_curve

    @classmethod
    def from_args(cls, cine_fname, h5_fname=None, *args, **kwargs):
        self = cls()
        '''Sets up the object based on arguments
        '''

        cls._verify_params(kwargs)

        self.params = kwargs
        try:
            self.bck_img = self.params.pop('bck_img')
        except KeyError:
            self.bck_img = None

        self.cine_fname = cine_fname

        self.cine_ = cine.Cine('/'.join(self.cine_fname))

        if self.bck_img is None:
            # not passed in, try the data base
            if self.db is not None:
                self.bck_img = self.db.get_background_img(self.cine_.hash)
            # if that fails too, run it
            if self.bck_img is None:
                self.bck_img = gen_bck_img('/'.join(self.cine_fname))
                # if we have a data base, shove in the data
                if self.db is not None:
                    self.db.store_background_img(self.cine_.hash, self.bck_img)

        return self

    def process_frame(self, frame_number, curve):
        # get the raw data, and convert to float
        tmp_img = self.get_frame(frame_number)

        tm, trk_res, tim, tam, miv, mav = proc_frame(curve,
                                                     tmp_img,
                                                     **self.params)

        mbe = MemBackendFrame(curve,
                              frame_number,
                              res=trk_res,
                              trk_lst=[tim, tam],
                              img=tmp_img)
        mbe.tm = tm
        next_curve = mbe.get_next_spline(**self.params)
        if 'fft_filter' in self.params:
            next_curve = copy.copy(next_curve)  # to not screw up the original
            next_curve.fft_filter(self.params['fft_filter'])
        return mbe, next_curve

    def get_frame(self, frame_number):
        '''Simply return the (possibly normalized) image for the given frame'''

        # get the raw data, and convert to float
        tmp_img = np.array(self.cine_[frame_number], dtype='float')
        # if
        if self.bck_img is not None:
            tmp_img /= self.bck_img

        return tmp_img

    def update_param(self, key, val):
        '''Updates the parameters'''
        self.params[key] = val

    def update_all_params(self, params):
        self._verify_params(params)
        self.params = params

    def gen_stub_h5(self, h5_fname, seed_curve):
        '''Generates a h5 file that can be read back in for later
        processing.  This assumes that the location of the h5 file is
        valid'''

        file_out = h5py.File(h5_fname, 'w-')   # open file
        file_out.attrs['ver'] = '0.1.4'       # set meta-data
        for key, val in self.params.items():
            try:
                file_out.attrs[key] = val
            except TypeError:
                print 'key: ' + key + ' can not be gracefully shoved into'
                print ' an hdf object, please reconsider your life choices'
            except Exception as e:
                print "FAILURE WITH HDF: " + e.__str__()

        file_out.attrs['cine_path'] = str(self.cine_fname.path)
        file_out.attrs['cine_fname'] = str(self.cine_fname.fname)

        file_out.attrs['cine_hash'] = self.cine_.hash

        if seed_curve is not None:
            seed_curve.write_to_hdf(file_out)
        if self.bck_img is not None:
            file_out.create_dataset('bck_img',
                                    self.bck_img.shape,
                                    np.float,
                                    compression='szip')
            file_out['bck_img'][:] = self.bck_img

        file_out.close()


class MemBackendFrame(object):
    """A class for keeping all of the relevant results about a frame in memory

    This class will get smarter over time.

     - add logic to generate res from raw
     - add visualization code to this object
    """
    def __init__(self,
                 curve,
                 frame_number,
                 res,
                 trk_lst=None,
                 img=None,
                 params=None,
                 *args,
                 **kwarg):
        self.curve = copy.copy(curve)
        self.res = res
        self.trk_lst = trk_lst
        self.frame_number = frame_number
        self.params = params
        self.next_curve = None
        self.img = img
        self.mix_in_count = None
        self.pix_err = None

        new_res = []
        for t_ in self.res:
            if len(t_) == 0:
                print t_
                continue
            tmp = ~np.isnan(t_[0])
            tmp_lst = [np.array(r)[tmp] for r in t_]
            new_res.append(tuple(tmp_lst))
        self.res = new_res

        pass

    def get_extent(self):
        if self.img is not None:
            return [0, self.img.shape[1], 0, self.img.shape[0]]
        else:
            x, y = self.curve.q_phi_to_xy(1, np.linspace(0, 2 * np.pi, 100))
            return [.9 * np.min(y), 1.1 * np.max(y),
                    .9 * np.min(x), 1.1 * np.max(x)]

    def get_next_spline(self, mix_in_count=0, pix_err=0, **kwargs):
        if self.next_curve is not None and self.mix_in_count == mix_in_count:
            return self.next_curve

        tim, tam = self.trk_lst

        # this is a parameter to forcibly mix in some number of points
        # from the last curve

        t_q = np.array([t.q for t in tim + tam if
                        t.q is not None
                        and t.phi is not None
                        and bool(t.charge)] +
                       [0] * int(mix_in_count))

        t_phi = np.array([t.phi for t in tim + tam if
                          t.q is not None
                          and t.phi is not None
                          and bool(t.charge)] +
                         list(np.linspace(0, 2 * np.pi, mix_in_count, endpoint=False)))

        indx = t_phi.argsort()
        t_q = t_q[indx]
        t_phi = t_phi[indx]
        # generate the new curve
        x, y = self.curve.q_phi_to_xy(t_q, t_phi, cross=False)

        new_curve = SplineCurve.from_pts(np.vstack((x, y)),
                                         pix_err=pix_err,
                                         **kwargs)

        self.next_curve = new_curve
        self.mix_in_count = mix_in_count

        return new_curve

    def plot_tracks(self, min_len=0, all_tracks=True):
        fig = plt.figure()
        ax = fig.gca()
        self.ax_draw_img(ax)
        self.ax_plot_tracks(ax, min_len, all_tracks)
        self.ax_draw_center_curves(ax)
        plt.draw()
        return ax

    def ax_plot_tracks(self, ax, min_len=0, all_tracks=True):
        color_cycle = ['r', 'b']
        lines = []
        for tk_l, c in zip(self.trk_lst, color_cycle):
            lines.extend([t.plot_trk_img(self.curve,
                                         ax,
                                         color=c,
                                         linestyle='-')
                          for t in tk_l
                          if len(t) > min_len and
                          (all_tracks or bool(t.charge))])
        return lines

    def ax_draw_center_curves(self, ax):
        lo = ax.plot(*self.curve.get_xy_samples(1000), color='g', lw=2)
        if self.next_curve is None:
            self.get_next_spline()

        new_curve = self.next_curve
        ln = ax.plot(*new_curve.get_xy_samples(1000), color='m',
                     lw=2, linestyle='--')

        return lo + ln

    def ax_draw_img(self, ax):
        if self.img is not None:
            c_img = ax.imshow(self.img,
                              cmap=plt.get_cmap('cubehelix'),
                              interpolation='nearest')
            c_img.set_clim([.5, 1.5])
            return c_img
        return None

    def write_to_hdf(self, parent_group):
        print 'frame_%05d' % self.frame_number
        group = parent_group.create_group('frame_%05d' % self.frame_number)
        _write_frame_tracks_to_file(group,
                                    self.trk_lst[0],
                                    self.trk_lst[1],
                                    self.curve)
        del group

    def get_profile(self):
        ch, th = zip(*sorted([(t.charge, t.phi) for
                              t in itertools.chain(*self.trk_lst) if
                              t.charge and len(t) > 15],
                            key=lambda x: x[-1]))

        dh, th_new = construct_corrected_profile((th, ch))

        return th_new, dh

    def get_theta_offset(self):
        r, th = self.curve.get_rt_samples(2)
        th = th.ravel()
        return th[0]

    def get_xyz_points(self, min_track_length=15):

        X, Y, ch, th = zip(
            *sorted([tuple(t.get_xy(self.curve)) + (t.charge, t.phi) for
                     t in itertools.chain(*self.trk_lst) if
                     t.charge and len(t) > 15],
                    key=lambda x: x[-1]))

        dh, th_new = construct_corrected_profile((th, ch))
        Z = [z * np.ones(x.shape) for z, x in zip(dh, X)]

        return X, Y, Z

    def get_xyz_curve(self, min_track_length=15):
        q, ch, th = zip(*sorted([(t.q, t.charge, t.phi) for
                                 t in itertools.chain(*self.trk_lst) if
                                 t.charge and len(t) > min_track_length],
                                key=lambda x: x[-1]))

        # scale the radius
        x, y = self.curve.q_phi_to_xy(q, th)

        z, th_new = construct_corrected_profile((th, ch))

        return x, y, z


def change_base_path(fpath, new_base_path):
    '''Returns a new FilePath object with a different base_path entry '''
    return FilePath(new_base_path, fpath.path, fpath.fname)


def copy_to_buffer_disk(fname, buffer_base_path):
    '''fname is a FilePath (or a 3 tuple with the layout
    (base_path, path, fname) '''
    if os.path.abspath(fname.base_path) == os.path.abspath(buffer_base_path):
        raise Exception("can not buffer to self!!")
    new_fname = change_base_path(fname, buffer_base_path)
    buff_path = '/'.join(new_fname[:2])
    ensure_path_exists(buff_path)
    src_fname = '/'.join(fname)
    buf_fname = '/'.join(new_fname)
    if not os.path.exists(buf_fname):
        shutil.copy2(src_fname, buf_fname)
    return new_fname


def ensure_path_exists(path):
    '''ensures that a given path exists, throws error if
    path points to a file'''
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isfile(path):
        raise Exception("there is a file where you think there is a path!")


class HdfBackend(object):
    """A class that wraps around an HDF results file"""
    def __init__(self,
                 fname,
                 cine_base_path=None,
                 h5_buffer_base_path=None,
                 cine_buffer_base_path=None,
                 *args,
                 **kwargs):
        print fname
        self._iter_cur_item = -1
        self.buffers = []
        self.file = None
        if h5_buffer_base_path is not None:
            fname = copy_to_buffer_disk(fname, h5_buffer_base_path)
            self.buffers.append(fname)
        self.file = h5py.File('/'.join(fname), 'r')
        self.num_frames = len([k for k in self.file.keys() if 'frame' in k])
        self.prams = HdfBEPram(True, True)
        self.proc_prams = dict(self.file.attrs)
        if 'bck_img' in self.file.keys():
            try:
                self.bck_img = self.file['bck_img'][:]
            except:
                self.bck_img = None
        else:
            self.bck_img = None
        if cine_base_path is not None:
            self.cine_fname = FilePath(cine_base_path,
                                       self.file.attrs['cine_path'],
                                       self.file.attrs['cine_fname'])
            if cine_buffer_base_path is not None:
                self.cine_fname = copy_to_buffer_disk(self.cine_fname,
                                                      cine_buffer_base_path)
                self.buffers.append(self.cine_fname)
            self.cine = cine.Cine('/'.join(self.cine_fname))
        else:
            self.cine_fname = None
            self.cine = None
        try:
            self.db = db.LFmongodb()  # hard code the mongodb
        except:
            # this eats _ALL_ exceptions
            self.db = None

        if self.bck_img is None:
            # not passed in, try the data base
            if self.db is not None:
                self.bck_img = self.db.get_background_img(self.cine_.hash)
            # if that fails too, run it
            if self.bck_img is None and self.cine is not None:
                self.gen_back_img()
                # if we have a data base, shove in the data
                if self.db is not None and self.bck_img is not None:
                    self.db.store_background_img(self.cine.hash, self.bck_img)

        pass

    def __len__(self):
        return self.num_frames

    def __del__(self):
        if self.file:
            self.file.close()
        for f in self.buffers:
            print 'removing ' + '/'.join(f)
            os.remove('/'.join(f))

    def get_frame(self, frame_num, raw=None, get_img=None, *args, **kwargs):
        trk_lst = None
        img = None
        g = self.file['frame_%05d' % frame_num]
        if raw is None:
            raw = self.prams.raw
        if raw:
            trk_lst = _read_frame_tracks_from_file_raw(g)
        if get_img is None:
            get_img = self.prams.get_img
        if get_img:
            if self.cine is not None:
                img = np.array(self.cine.get_frame(frame_num), dtype='float')
                if self.bck_img is not None:
                    img /= self.bck_img

        res = _read_frame_tracks_from_file_res(g)
        curve = SplineCurve.from_hdf(g)
        return MemBackendFrame(curve,
                               frame_num,
                               res,
                               trk_lst=trk_lst,
                               img=img,
                               params=self.proc_prams)

    def gen_back_img(self):
        if self.cine_fname is not None:
            self.bck_img = gen_bck_img(self.cine_fname)

    def __iter__(self):
        self._iter_cur_item = -1
        return self

    def next(self):
        self._iter_cur_item += 1
        if self._iter_cur_item >= self.num_frames:
            raise StopIteration
        else:
            return self.get_frame(self._iter_cur_item)

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.get_frame, range(self.num_frames)[key])

        return self.get_frame(key)


class SplineCurve(object):
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
        :param need_sort: if the points need to be sorted or should be processed as-is

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
            raise Exception("not enough points")

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

    def __init__(self, tck):
        '''A really hacky way of doing different
        '''
        self.tck = tck

    def get_xy_samples(self, sample_count):
        '''
        Returns the x-y coordinates of uniformly sampled points on the
        spline.

        STOP USING THIS
        '''
        return self.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, sample_count))

    def write_to_hdf(self, parent_group):
        '''
        Writes out the essential data (spline of central curve) to hdf file.
        '''
        parent_group.attrs['tck0'] = self.tck[0]
        parent_group.attrs['tck1'] = np.vstack(self.tck[1])
        parent_group.attrs['tck2'] = self.tck[2]

    def circumference(self):
        '''returns a rough estimate of the circumference'''
        new_pts = si.splev(np.linspace(0, 1, 1000), self.tck, ext=2)
        return np.sum(np.sqrt(np.sum(np.diff(new_pts, axis=1) ** 2, axis=0)))

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
            if ((phi_shape is not None) and (q_shape is not None) and (phi_shape == q_shape)):
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

            data_out = [(x + q * nx).reshape(phi_shape),
                        (y + q * ny).reshape(phi_shape)]

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
        return ax.plot(*(self.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, N))), **kwargs)


def find_rim_fringes(curve, lfimg, s_width, s_num,
                     smooth_rng=2, oversample=4, *args, **kwargs):
    """
    Does the actual work of finding the fringes on the image
    """

    # a really rough estimate of the circumference
    C = curve.circumference()

    # sample points at ~ 2/pix
    sample_count = int(np.ceil(C * int(oversample)))

    # get center of curve
    new_pts = curve.get_xy_samples(1000)
    x0, y0 = np.mean(new_pts, 1).ravel()

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


def _link_run(best_accum, best_order, cur_accum, cur_order, source, dest, max_search_range):
    '''

    A function to link 'runs' together.  A run is an ordered 1D
    sequence.  This will link together two subsequent runs, allowing
    there to be gaps and overhang at both ends, but does not allow
    crossing.  That is, if a1 < b1 -> a2 < b2.

    The algorithm is a modified version of the Crocker-Grier
    algorithm.  Given two lists, there are three possible options, the
    first item in each list are linked [0], the first item in the
    source list is not linked to anything (skipped) [1] or the first
    item in the second list is not linked to anything (skipped) [-1].
    The remainder of the lists are then dealt with recursively.  If
    one list is exhausted first, the remainder of the other list is
    marked as skipped.

    This function runs recursively, with a maximum depth the length of
    the longest input.

    This is an internal function and probably shouldn't be directly
    used

    :param best_accum: the current minimum penalty
    :param best_order: the current best linkange
    :param cur_accum: the accumulated penalty of the current candidate
    linkage
    :param cur_order: the current proposed linkage
    :param source: a list of objects to link from.  Objects must
    implement `distance`
    :param dest: a list of objects to link to.  Objects must implement
    `dist`
    :param max_search_range: the maximum distance away to consider a link.
    Also sets penalty for skipping a link
    '''

    # base cases
    if len(source) == 0:
        tmp_accum = cur_accum + len(dest) * max_search_range
        if tmp_accum < best_accum:
            # we have a winner
            cur_order = cur_order[:]      # get a copy
            cur_order.extend([1] * len(dest))
            return cur_order, tmp_accum
        else:
            # old way is still best
            return best_order, best_accum

    if len(dest) == 0:
        tmp_accum = cur_accum + len(source) * max_search_range
        if tmp_accum < best_accum:
            cur_order = cur_order[:]      # get a copy
            cur_order.extend([-1] * len(dest))
            return cur_order, tmp_accum
        else:
            # old way is still best
            return best_order, best_accum

    # try by linking the first two entries of the lists together, recurse on the rest
    source_head = source.pop(0)
    dest_head = dest.pop(0)
    dist = source_head.distance(dest_head)
    if dist < max_search_range:  # if the distance is less than the maximum
        tmp_accum = cur_accum + dist      # get new trial accum
        if tmp_accum < best_accum:
            cur_order.append(0)
            best_order, best_accum = _link_run(best_accum, best_order,
                                               tmp_accum, cur_order,
                                               source, dest,
                                               max_search_range)

            cur_order.pop()

    # only need to do this once for both of the next two checks
    tmp_accum = cur_accum + max_search_range
    # try dropping just the first entry in source
    dest.insert(0, dest_head)

    if tmp_accum < best_accum:
            cur_order.append(-1)
            best_order, best_accum = _link_run(best_accum, best_order,
                                               tmp_accum, cur_order,
                                               source, dest,
                                               max_search_range)

            cur_order.pop()

    # try dropping the first entry of the dest
    source.insert(0, source_head)
    dest_head = dest.pop(0)

    if tmp_accum < best_accum:
            cur_order.append(1)
            best_order, best_accum = _link_run(best_accum, best_order,
                                               tmp_accum, cur_order,
                                               source, dest,
                                               max_search_range)

            cur_order.pop()

    dest.insert(0, dest_head)             # make sure list is unchanged by pass through function

    return best_order, best_accum


# class FringeRing(object):
#     '''
#     A class to carry around and work with the location of fringes for
#     the purpose of tracking a fixed height faring from frame to frame.
#     '''
#     def __init__(self, theta, charge, th_offset=0, ringID=None):

#         rel_height, th_new = construct_corrected_profile((theta, charge),
#                                                          th_offset)

#         self.fringes = []

#         for th, h in zip(th_new, rel_height):

#             self.fringes.append(Point1D_circ(ringID, th, h))

#     def __iter__(self):
#         return self.fringes.__iter__()

#     def plot_fringe_flat(self, ax, **kwargs):
#         q, phi = zip(*sorted([(f.q, f.phi) for f in self.fringes],
#                              key=lambda x: x[1]))
#         ax.plot(phi, q, **kwargs)
#         plt.draw()


class FringeRun(object):
    '''
    A class to carry around 'runs' of fringes, that is fringes of the same charge in a row.
    '''
    def __init__(self, thetas, run_charge):
        self.theta = np.mean(thetas)
        self.thetas = thetas
        self.charge = run_charge


class Fringe(Point1D_circ):
    '''
    Version of :py:class:`Point1D_circ` for representing fringes

    '''

    def __init__(self, q, phi):
        Point1D_circ.__init__(self, q=q, phi=phi)                  # initialize first base class
        # linked list for time
        self.next_T = None   #: next fringe in time
        self.prev_T = None   #: prev fringe in time
        # linked list for space
        self.next_P = None   #: next fringe in angle
        self.prev_P = None   #: prev fringe in angle
        # properties of fringe shape
        self.charge = None   #: up or down [-1, 0, 1]
        self.R2 = None       # how well shape in described by quadratic
        # self.q and self, phi are set by Point1D_circ __init__
        self.color = None    #: if this is a light or dark fringe [-1, 1]
        # properties of fringe height

        self.f_dh = None     #: dh figured going forward
        self.r_dh = None     #: dh figured going backwards
        self.f_cumh = None   #: the cumulative shift counting forward
        self.r_cumh = None   #: the cumulative shift counting forward

        self.height = None   #: the height of this fringe as given by tracking in time

        self.slope_f = None  #: the slope going forward
        self.slope_r = None  #: the slope going backward
        self.slope = None    #: the 'average' slope at this point

    def add_to_track(self, track):
        if self.prev_T is not None:
            raise Exception("This fringe already has a preceding fringe")
        prev_F = track.last_point()
        if prev_F.next_T is not None:
            raise Exception("The last fringe in this track already has a next, something is very wrong")
        self.prev_T = prev_F
        prev_F.next_T = self
        # pass it off to the super class
        Point1D_circ.add_to_track(self, track)

    def remove_from_track(self, track):
        # re-link the linked list... not sure if we ever will _want_ to do this
        if self.prev_T is not None:
            self.prev_T.next_T = self.next_T
        if self.next_T is not None:
            self.next_T.prev_T = self.prev_T
        Point1D_circ.remove_from_track(self, track)

    def insert_ahead(self, other):
        '''
        Inserts `other` ahead of this Fringe in the spatial linked-list
        '''
        if self.next_P is not None:
            self.next_P.prev_P = other
            other.next_P = self.next_P

        self.next_P = other
        other.prev_P = self

    def insert_behind(self, other):
        '''
        Inserts other behind this Fringe in the spatial linked-list
        '''
        if self.prev_P is not None:
            self.prev_P.next_P = other
            other.prev_P = self.prev_P

        self.prev_P = other
        other.next_P = self

    def remove_R(self):
        '''
        Removes this Fringe from the spatial linked-list
        '''
        if self.prev_P is not None:
            self.prev_P.next_P = self.next_P
        if self.next_P is not None:
            self.next_P.prev_P = self.prev_P

    def determine_forward_dh(self):
        '''
        Looks at the trailing fringe and figures out what the delta
        from that fringe to this one should be.

        Raises an exception of there is an invalid sequence
        '''

        prev = self.prev_P
        cur = self
        if prev is None:
            raise RuntimeError("should not have called this function with out a valid angular linkage")
        if prev.color is None or prev.charge is None:
            self.f_dh = None
            raise RuntimeError("The previous fringe in invalid")
        if cur.charge is None or cur.color is None:
            self.f_dh = None
            raise RuntimeError("The current fringe in invalid")
        if prev.color == cur.color:
            # the fringes are the same color (light-light or dark-dark)
            # then they must have opposite charge.
            if prev.charge == -cur.charge:
                # they have opposite charge
                # no height change between these two fringes (inner fringes on extrema
                self.f_dh = 0
            else:
                self.f_dh = None
                # they do not have opposite charge, this is a problem
                if cur.charge == 0:
                    raise InvalidSequence([(-prev.color, prev.charge)], 'same color, next 0')
                elif prev.charge == 0:
                    raise InvalidSequence([(-cur.color, cur.charge)], 'same color, prev 0')
                else:
                    raise InvalidSequence([(-prev.color, prev.charge)], 'same color, same charge')
        else:
            # they have different color
            if prev.charge == 0 and prev.charge == 0:
                self.f_dh = -np.sign(prev.prev_P.charge)
            elif prev.charge == 0:
                self.f_dh = np.sign(cur.charge)
            elif cur.charge == 0:
                self.f_dh = np.sign(prev.charge)
            elif prev.charge == cur.charge:
                self.f_dh = np.sign(cur.charge)
            else:
                # they have different charges, both are non-zero, and
                # different colors, this is a problem
                self.f_dh = None
                raise InvalidSequence([(prev.color, -prev.charge),
                                       (cur.color, -cur.charge)
                                       ], 'different color, different charge')

    def is_valid_order(self):
        '''
        Looks at the region around a fringe an determines if it is valid or not.

        If it looks good, it will return (True, None)

        If it looks like the current fringe is miss-classified, fix it and return (True, None)

        If there is a missing fringe, return (False, [(color,charge),..])

        '''
        prev_p = self.prev_P
        cur_p = self

        if prev_p is None:
            raise RuntimeError("should not have called this function with out a valid angular linkage")
        if prev_p.color is None or prev_p.charge is None:
            return True, None  # can't tell
        if cur_p.charge is None or cur_p.color is None:
            return True, None  # can't tell

        if cur_p.charge == 0:
            # need the next fringe to reason about charge 0 fringes
            next_p = self.next_P
            if next_p is None:
                raise RuntimeError("should not have called this function with out a valid angular linkage")
            # bracket by same color
            if prev_p.color == next_p.color:
                if prev_p.charge == -next_p.charge:
                    if prev_p.color != cur_p.color:
                        # this zero is in an extreama happy day (or a run of 3 zeros)
                        return True, None
                    elif prev_p.color == cur_p.color:
                        # we have three things in the same color and charge in a row
                        # because we always insert _behind_ the current fringe when
                        # we add extra fringes, only worry about fixing the trailing gap
                        return False, [(-prev_p.color, prev_p.charge)]
                    else:
                        raise RuntimeError("should never hit this")
                elif prev_p.charge == next_p.charge:
                    if cur_p.color == prev_p.color:
                        # deal only with the trailing correction
                        return False, [(-cur_p.color, prev_p.charge)]
                    elif cur_p.color != prev_p.color:
                        # this is the case of a zero in the middle of a run, probably a miss classification
                        cur_p.charge = prev_p.charge
                        return True, None
                    else:
                        raise RuntimeError("should never hit this")
                elif prev_p.charge == 0 or next_p.charge == 0:
                    if prev_p.color == cur_p.color:
                        return False, [(-prev_p.color, prev_p.charge)]
                    elif prev_p.color != cur_p.color:
                        return True, None
                    else:
                        raise RuntimeError("should never hit this")

                else:
                    raise RuntimeError("should never hit this")
            # bracketed by different colors
            elif prev_p.color != next_p.color:
                if prev_p.charge == -next_p.charge:
                    # this is likely a fringe incorrectly labeled as zero charge
                    # and is really the other side of an exmtrema pair
                    if cur_p.color == prev_p.color:
                        cur_p.charge = -prev_p.charge
                        return True, None
                    elif cur_p.color == next_p.color:
                        cur_p.charge = -next_p.charge
                        return True, None
                    else:
                        raise RuntimeError("should never hit this")
                elif prev_p.charge == next_p.charge:
                    # this is another configuration that needs more
                    # than one total fringe insterted, only deal with
                    # trailing case
                    if cur_p.color == prev_p.color:
                        return False, [(-cur_p.color, prev_p.charge)]
                    elif cur_p.color == next_p.color:
                        return True, None
                    else:
                        raise RuntimeError("should never hit this")
                elif prev_p.charge == 0 or next_p.charge == 0:
                    if prev_p.color == cur_p.color:
                        return False, [(-prev_p.color, prev_p.charge)]
                    elif prev_p.color != cur_p.color:
                        return True, None
                    else:
                        raise RuntimeError("should never hit this")
                else:
                    raise RuntimeError("should never hit this")
            else:
                raise RuntimeError("should never hit this")

        elif cur_p.charge != 0:
            if prev_p.color == cur_p.color:
                if prev_p.charge == -cur_p.charge:
                    # same color, opposite charge -> extrema we are happy
                    return True, None
                elif prev_p.charge == cur_p.charge:
                    # same color, same charge, missing fringe in run
                    return False, [(-cur_p.color, cur_p.charge)]
                elif prev_p.charge == 0:
                    # only one possible insertion
                    return False, [(-cur_p.color, cur_p.charge)]
                else:
                    raise RuntimeError("should never hit this")

            elif prev_p.color != cur_p.color:
                if prev_p.charge == -cur_p.charge:
                    # this is a blown extrma, could be one of two fringes
                    return False, [(prev_p.color, -prev_p.charge),
                                   (cur_p.color, -cur_p.charge)]
                elif prev_p.charge == cur_p.charge:
                    # on a run
                    return True, None
                elif prev_p.charge == 0:
                    return True, None
                else:
                    raise RuntimeError("should never hit this")

            else:
                raise RuntimeError("should never hit this")

        else:
            raise RuntimeError("should never hit this")
        print cur_p.charge, cur_p.color
        raise RuntimeError("should never hit this")

    def determine_reverse_dh(self):
        '''
        Looks at the trailing fringe and figures out what the delta
        from that fringe to this one should be

        Raises an exception of there is an invalid sequence
        '''
        pass

    def set_color_charge(self, color, charge):
        self.color = color
        self.charge = charge

    def clear_color_charge(self):
        self.color = None
        self.charge = None

    def __unicode__(self):
        return 'q: %0.2f, phi: %0.2f, charge: %s, color: %s' % (self.q, self.phi, str(self.charge), str(self.color))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return 'Fringe(%f, %f, %s, %s)' % (self.q, self.phi, str(self.charge), str(self.color))


class InvalidSequence(Exception):
    def __init__(self, valid_configs, msg='', *args, **kwargs):
        Exception.__init__(self, msg, *args, **kwargs)
        self.valid_configs = valid_configs


class FringeRing(object):
    '''
    A class to carry around Fringe data
    '''
    def __init__(self, tau):
        self.fringes = []
        self.tau = tau
        self.invalid_fringes = []

    def gen_fringe_runs(self):
        pass

    def link_to_next_ring(self, other):
        pass

    def set_forward_deltas(self):
        for f in self.fringes:
            f.determine_forward_dh()
        return np.cumsum([f.f_dh for f in self.fringes])

    def find_invalid_sequence(self):
        for f in self.fringes:
            valid, configs = f.is_valid_order()
            if not valid:
                pre_f = f.prev_P
                invalid_fringe = Fringe(0, pre_f.phi + pre_f.distance(f) / 2)
                f.insert_behind(invalid_fringe)
                self.invalid_fringes.append((invalid_fringe, configs))

        self.fringes.extend([f for f, _ in self.invalid_fringes])
        self.fringes.sort(key=lambda x: x.phi)

    def set_reverse_deltas(self):
        for f in self.fringes:
            try:
                f.determine_reverse_dh()
            except InvalidSequence as e:
                pre_f = f.next_P
                invalid_fringe = Fringe(0, pre_f.phi + pre_f.distance(f) / 2)
                f.insert_ahead(invalid_fringe)
                self.invalid_fringes.append((invalid_fringe, e.valid_configs))

    def compute_cumulative_forward(self):
        pass

    def compute_cumulative_reverse(self):
        pass

    def sort_out_invalid(self):

        pass


def get_rf(hf, j):
    mbe = hf[j]
    th_offset = mbe.get_theta_offset()
    rf = FringeRing(mbe.res[0][1],
                    mbe.res[0][0],
                    th_offset=th_offset,
                    ringID=j)
    return rf


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
