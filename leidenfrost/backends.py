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
import itertools
import cPickle
import os

import numpy as np
import matplotlib.pyplot as plt


import h5py
import cine


import copy

import shutil
import collections

import leidenfrost.db as db
import leidenfrost.infra as infra
import leidenfrost.ellipse as ellipse
from leidenfrost import FilePath

HdfBEPram = collections.namedtuple('HdfBEPram', ['raw', 'get_img'])


class HdfBackend(object):
    """A class that wraps around an HDF results file"""
    def __init__(self,
                 fname,
                 cine_base_path=None,
                 h5_buffer_base_path=None,
                 cine_buffer_base_path=None,
                 mode='r',
                 *args,
                 **kwargs):
        """
        Parameters
        ----------
        fname: `Leidenfrost.FilePath`
            Fully qualified path to the hdf file to open
        cine_base_path: str or `None`
            If not `None`, base path to find the raw cine files
        h5_buffer_bas_path: str or `None`
            If not `None`, base path for buffering the h5 file
        cine_buffer_base_path: str or `None`
            If not `None`, base path for buffering the cine file
        """

        print fname
        self._iter_cur_item = -1
        self.buffers = []
        self.file = None
        if h5_buffer_base_path is not None:
            fname = copy_to_buffer_disk(fname, h5_buffer_base_path)
            self.buffers.append(fname)
        if mode == 'rw':
            self.file = h5py.File(fname.format, 'r+')
            self.writeable = True
        else:
            self.file = h5py.File(fname.format, 'r')
            self.writeable = False

        self.num_frames = len([k for k in self.file.keys() if 'frame' in k])
        self.prams = HdfBEPram(True, True)
        self.proc_prams = dict(self.file.attrs)

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
            print 'gave up and the DB'
            # this eats _ALL_ exceptions
            self.db = None

        self.bck_img = None
        if self.db is not None:
            self.bck_img = self.db.get_background_img(self.cine.hash)
        # if that fails too, run it
        if self.bck_img is None and self.cine is not None:
            self.gen_back_img()
            # if we have a data base, shove in the data
            if self.db is not None and self.bck_img is not None:
                self.db.store_background_img(self.cine.hash, self.bck_img)

        if self.file.attrs['ver'] < '0.1.5':
            self._frame_str = 'frame_{:05}'
        else:
            self._frame_str = 'frame_{:07}'

        pass

    @property
    def ver(self):
        return self.file.attrs['ver']

    def _del_frame(self, j):
        """Deletes frame j.

        *THIS BLOODY DELETES DATA!!!*

        Parameters
        ----------
        j : int
            the frame to delete

        """
        if self.contains_frame(j):
            del self.file[self._frame_str.format(j)]
        self.num_frames = len([k for k in self.file.keys() if 'frame' in k])

    def contains_frame(self, j):
        '''Returns if frame `j` is saved in the hdf file

        Parameters
        ----------
        j : int
            The frame to check
        '''
        return self._frame_str.format(j) in self.file

    def __len__(self):
        return self.num_frames

    @property
    def cine_len(self):
        return len(self.cine)

    def __del__(self):
        if self.file:
            self.file.close()
        for f in self.buffers:
            print 'removing ' + '/'.join(f)
            os.remove('/'.join(f))

    def get_frame(self, frame_num, raw=None, get_img=None, *args, **kwargs):
        trk_lst = None
        img = None
        g = self.file[self._frame_str.format(frame_num)]
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
        next_curve = None
        if self.file.attrs['ver'] < '1.1.5':
            seed_curve = infra.SplineCurve.from_hdf(g)
        else:
            seed_curve = infra.SplineCurve.from_hdf(g['seed_curve'])
            if 'next_curve' in g:
                next_curve = infra.SplineCurve.from_hdf(g['next_curve'])
        return MemBackendFrame(seed_curve,
                               frame_num,
                               res,
                               trk_lst=trk_lst,
                               img=img,
                               params=self.proc_prams,
                               next_curve=next_curve)

    def gen_back_img(self):
        if self.cine_fname is not None:
            self.bck_img = infra.gen_bck_img(self.cine_fname)

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


class ProcessBackend(object):
    req_args_lst = ['search_range', 's_width', 's_num', 'pix_err']

    def __len__(self):
        if self.cine_ is not None:
            return len(self.cine_)
        else:
            return 0

    def __init__(self, ver='1.1.5'):
        self.params = {}        # the parameters to feed to proc_frame

        self.cine_fname = None               # file name
        self.cine_ = None                    # the cine object

        self.bck_img = None       # back ground image for normalization
        try:
            self.db = db.LFmongodb()  # hard code the mongodb
        except:
            print 'gave up and DB'
            # this eats _ALL_ exceptions
            self.db = None
        self.ver = ver

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
            self.bck_img = infra.gen_bck_img('/'.join(self.cine_fname))

        seed_curve = infra.SplineCurve.from_hdf(tmp_file)

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
                self.bck_img = infra.gen_bck_img('/'.join(self.cine_fname))
                # if we have a data base, shove in the data
                if self.db is not None:
                    self.db.store_background_img(self.cine_.hash, self.bck_img)

        return self

    def process_frame(self, frame_number, curve):
        # get the raw data, and convert to float
        tmp_img = self.get_image(frame_number)

        tm, trk_res, tim, tam, miv, mav = infra.proc_frame(curve,
                                                           tmp_img,
                                                           **self.params)

        mbe = MemBackendFrame(curve,
                              frame_number,
                              res=trk_res,
                              trk_lst=[tim, tam],
                              img=tmp_img,
                              ver=self.ver,
                              params=self.params)
        mbe.tm = tm
        next_curve = mbe.get_next_spline(**self.params)
        if 'fft_filter' in self.params:
            next_curve = copy.copy(next_curve)  # to not screw up the original
            next_curve.fft_filter(self.params['fft_filter'])
        return mbe, next_curve

    def get_frame(self, ind, raw, img):
        return NotImplementedError('need to write this')

    def get_image(self, frame_number):
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

    def write_config(self, seed_curve, cur_frame=None):
        if self.db is None:
            raise RuntimeError("need a valid db object to do this")
        tmpcurve_dict = dict((lab, cPickle.dumps(_tck))
                             for lab, _tck in zip(['tck0', 'tck1', 'tck2'],
                                                  seed_curve.tck))

        tmp_params = copy.copy(self.params)
        if cur_frame is not None:
            tmp_params['start_frame'] = cur_frame

        conf_id = self.db.store_config(self.cine_.hash, tmp_params,
                                       {str(0): tmpcurve_dict})
        return conf_id

    def gen_stub_h5(self, h5_fname, seed_curve):
        '''Generates a h5 file that can be read back in for later
        processing.  This assumes that the location of the h5 file is
        valid'''

        file_out = h5py.File(h5_fname, 'w-')   # open file
        file_out.attrs['ver'] = '0.1.5'       # set meta-data
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

        file_out.close()


class MemBackendFrame(object):
    """A class for keeping all of the relevant results about a frame in memory

    This class will get smarter over time.

     - add logic to generate res from raw
     - add visualization code to this object
    """
    def __init__(self,
                 seed_curve,
                 frame_number,
                 res=None,
                 trk_lst=None,
                 img=None,
                 params=None,
                 next_curve=None,
                 *args,
                 **kwarg):
        self.curve = copy.copy(seed_curve)
        self.res = res
        self.trk_lst = trk_lst
        self.frame_number = frame_number
        if params is not None:
            self.params = params
        else:
            self.params = {}

        self.next_curve = copy.copy(next_curve)
        self._params_cache = None
        self.img = img
        self.mix_in_count = None
        self.pix_err = None

        if self.res is not None:
            new_res = []
            for t_ in self.res:
                if len(t_) == 0:
                    print t_
                    continue
                tmp = ~np.isnan(t_[0])
                tmp_lst = [np.array(r)[tmp] for r in t_]
                new_res.append(tuple(tmp_lst))
            self.res = new_res

        self._frame_str = 'frame_{:07}'
        if 'ver' in kwarg:
            ver = kwarg.pop('ver')
            if ver < '1.1.5':
                self._frame_str = 'frame_{:05}'
        pass

    def get_extent(self, curve_extent=True):
        if self.img is not None and not curve_extent:
            return [0, self.img.shape[1], 0, self.img.shape[0]]
        else:
            x, y = self.curve.q_phi_to_xy(1, np.linspace(0, 2 * np.pi, 100))
            return [.9 * np.min(x), 1.1 * np.max(x),
                    .9 * np.min(y), 1.1 * np.max(y),
                    ]

    def get_next_spline(self, mix_in_count=0, pix_err=0, max_gap=None, **kwargs):
        _params_cache = (mix_in_count, pix_err, max_gap)
        if _params_cache == self._params_cache and self.next_curve is not None:
            return self.next_curve
        else:
            self._params_cache = _params_cache

        tim, tam = self.trk_lst

        # this is a parameter to forcibly mix in some number of points
        # from the last curve

        t_q = np.array([t.q for t in tim + tam if
                        t.q is not None
                        and t.phi is not None
                        and bool(t.charge)] +
                       [0] * int(mix_in_count))

        # this mod can come out later if the track code is fixed
        t_phi = np.array([np.mod(t.phi, 2 * np.pi) for t in tim + tam if
                         t.q is not None
                         and t.phi is not None
                         and bool(t.charge)] +
                         list(np.linspace(0, 2 * np.pi, mix_in_count, endpoint=False)))

        indx = t_phi.argsort()
        t_q = t_q[indx]
        t_phi = t_phi[indx]
        # get x,y points
        x, y = self.curve.q_phi_to_xy(t_q, t_phi, cross=False)

        if max_gap is not None:
            # check for gaps
            t_phi_diff = np.diff(t_phi)
            R = self.curve.circ / (np.pi*2)
            cntr = self.curve.cntr
            N = 12                             # how much buffer to take

            filler_data = []
            for gap in np.where(t_phi_diff > max_gap)[0]:
                #        print 'MIND THE GAP', gap, len(t_phi)
                gap += 1

                if gap < N:
                    # deal with wrap-around
                    _x_l = np.hstack((x[len(t_phi) - (N - gap):], x[:gap]))
                    _x_r = x[gap:gap+N+1]
                    _y_l = np.hstack((y[len(t_phi) - (N - gap):], y[:gap]))
                    _y_r = y[gap:gap+N+1]
                elif gap >= len(t_phi) - N:
                    # deal with wrap-around
                    _x_l = x[gap-N:gap]
                    _x_r = np.hstack((x[gap:], x[:(N - (len(t_phi) - gap)) + 1]))
                    _y_l = y[gap-N:gap]
                    _y_r = np.hstack((y[gap:], y[:(N - (len(t_phi) - gap)) + 1]))
                else:
                    _x_l = x[gap-N:gap]
                    _x_r = x[gap:gap+N+1]
                    _y_l = y[gap-N:gap]
                    _y_r = y[gap:gap+N+1]

                try:
                    filler_data.append((gap,
                                        ellipse.gap_filler((_x_l, _x_r), (_y_l, _y_r), R, cntr)))
                except ellipse.EllipseException:  # as e:
                    #                    print e
                    pass

            # deal with gap between last and first points
            if np.mod(t_phi[0] - t_phi[-1], 2 * np.pi) > max_gap:
                gap = len(t_phi)
                _x_l = x[-N:]
                _x_r = x[:N+1]
                _y_l = y[-N:]
                _y_r = y[:N+1]

                try:
                    filler_data.append((gap,
                                        ellipse.gap_filler((_x_l, _x_r), (_y_l, _y_r), R, cntr)))
                except ellipse.EllipseException:  # as e:
                    #                    print e
                    pass

            start_indx = 0
            accum_lst = []
            for gap, i_data in filler_data:
                accum_lst.append(np.vstack((x[start_indx:gap], y[start_indx:gap])))
                accum_lst.append(i_data)
                start_indx = gap
            accum_lst.append(np.vstack((x[start_indx:], y[start_indx:])))

            pts = np.hstack(accum_lst)
        else:
            # don't look for a gap
            pts = np.vstack((x, y))

        # generate the new curve
        new_curve = infra.SplineCurve.from_pts(pts,
                                               pix_err=pix_err,
                                               need_sort=False,
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

    def ax_draw_center_curves(self, ax, prev_c=True, next_c=True):
        if prev_c:
            lo = self.curve.draw_to_axes(ax, color='g',
                                         lw=2, linestyle='--')
        else:
            lo = []
        if next_c:
            new_curve = self.get_next_spline(**self.params)
            ln = new_curve.draw_to_axes(ax, color='m',
                                        lw=1, linestyle='--')
        else:
            ln = []

        return lo + ln

    def ax_draw_img(self, ax, cmap='cubehelix'):
        if self.img is not None:
            c_img = ax.imshow(self.img,
                              cmap=plt.get_cmap(cmap),
                              interpolation='nearest')
            c_img.set_clim([.5, 1.5])
            return c_img
        return None

    def write_to_hdf(self, parent_group):
        #        print 'frame_%05d' % self.frame_number

        group = parent_group.create_group(self._frame_str.format(self.frame_number))
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

        dh, th_new = infra.construct_corrected_profile((th, ch))

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

        dh, th_new = infra.construct_corrected_profile((th, ch))
        Z = [z * np.ones(x.shape) for z, x in zip(dh, X)]

        return X, Y, Z

    def get_xyz_curve(self, min_track_length=15):
        q, ch, th = zip(*sorted([(t.q, t.charge, t.phi) for
                                 t in itertools.chain(*self.trk_lst) if
                                 t.charge and len(t) > min_track_length],
                                key=lambda x: x[-1]))

        # scale the radius
        x, y = self.curve.q_phi_to_xy(q, th)

        z, th_new = infra.construct_corrected_profile((th, ch))

        return x, y, z


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


def change_base_path(fpath, new_base_path):
    '''Returns a new FilePath object with a different base_path entry '''
    return FilePath(new_base_path, fpath.path, fpath.fname)


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
            tmp_trk = infra.lf_Track()
            for ma, phi in tmp_raw_data[strt_indx:(strt_indx + t_len), :]:
                tmp_trk.add_point(infra.Point1D_circ(ma, phi))
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
        try:
            tmp_trk_res = parent_group[trk_res_name + n_mod][:]
            tmp_charge = tmp_trk_res[:, 0]
            tmp_phi = tmp_trk_res[:, 1]
            tmp_q = tmp_trk_res[:, 2]
            res_lst.append((tmp_charge, tmp_phi, tmp_q))
        except Exception as E:
            print E
            print n_mod

    if len(res_lst) != 2:
        res_lst = None

    return res_lst


def _write_frame_tracks_to_file(parent_group,
                                t_min_lst,
                                t_max_lst,
                                orig_curve,
                                next_curve=None,
                                md_args=None):
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

    `orig_curve`
        the curve for this frame that all the positions are based on

    'next_curve`
       the curve found from the centers of the fringes in this
        frame

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
    # this is here to not have to convert mixed files
    orig_curve.write_to_hdf(parent_group)
    orig_curve.write_to_hdf(parent_group, 'seed_curve')
    if next_curve is not None:
        next_curve.write_to_hdf(parent_group, 'next_curve')

    if md_args is None:
        md_args = {}

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
                                        np.float)

            if tmp_raw_data.shape[0] > 0:
                parent_group[raw_data_name + n_mod][:] = tmp_raw_data

            parent_group.create_dataset(raw_track_md_name + n_mod,
                                        tmp_raw_track_data.shape,
                                        np.float)

            if tmp_raw_track_data.shape[0] > 0:
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
                                        np.float)
            if tmp_track_res.shape[0] > 0:
                parent_group[trk_res_name + n_mod][:] = tmp_track_res
