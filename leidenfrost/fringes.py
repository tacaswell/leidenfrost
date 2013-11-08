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
import __builtin__
from itertools import tee, izip, cycle
from collections import namedtuple, defaultdict, deque

from contextlib import closing

from matplotlib import cm
import matplotlib
import fractions
import scipy.ndimage as ndi
import scipy.signal
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import numpy as np
import h5py
from bisect import bisect
import scipy
import types
import heapq
from mpl_toolkits.axes_grid1 import Grid
import networkx

fringe_cls = namedtuple('fringe_cls', ['color', 'charge', 'hint'])
fringe_loc = namedtuple('fringe_loc', ['q', 'phi'])


class Region_Edges(namedtuple('Region_Edges', ['starts', 'labels', 'ends'])):
    __slots__ = ()

    def __eq__(self, other):
        return all(np.all(_s == _o) for _s, _o in izip(self, other))

# set up all the look-up dictionaries
# list of the needed combinations
fringe_type_list = [fringe_cls(1, 0, 1),
                    fringe_cls(1, 0, -1),
                    fringe_cls(1, 1, 0),
                    fringe_cls(1, -1, 0),
                    fringe_cls(-1, 0, 1),
                    fringe_cls(-1, 0, -1),
                    fringe_cls(-1, 1, 0),
                    fringe_cls(-1, -1, 0)]

fringe_type_dict = dict(enumerate(fringe_type_list))
fringe_type_dict.update(reversed(i) for i in fringe_type_dict.items())

valid_follow_dict = defaultdict(list)
valid_follow_dict[fringe_type_dict[4]] = [fringe_type_dict[j]
                                          for j in [3, 6, 4, 1]]
valid_follow_dict[fringe_type_dict[5]] = [fringe_type_dict[j]
                                          for j in [2, 7, 5, 0]]
valid_follow_dict[fringe_type_dict[7]] = [fringe_type_dict[j]
                                          for j in [3, 6, 4, 1]]
valid_follow_dict[fringe_type_dict[6]] = [fringe_type_dict[j]
                                          for j in [2, 7, 5, 0]]

valid_follow_dict[fringe_type_dict[0]] = [fringe_type_dict[j]
                                          for j in [7, 2, 0, 5]]
valid_follow_dict[fringe_type_dict[1]] = [fringe_type_dict[j]
                                          for j in [6, 3, 1, 4]]
valid_follow_dict[fringe_type_dict[3]] = [fringe_type_dict[j]
                                          for j in [7, 2, 0, 5]]
valid_follow_dict[fringe_type_dict[2]] = [fringe_type_dict[j]
                                          for j in [6, 3, 1, 4]]

valid_precede_dict = defaultdict(list)
valid_precede_dict[fringe_type_dict[4]] = [fringe_type_dict[j]
                                           for j in [1, 4, 7, 2]]
valid_precede_dict[fringe_type_dict[5]] = [fringe_type_dict[j]
                                           for j in [0, 5, 6, 3]]
valid_precede_dict[fringe_type_dict[7]] = [fringe_type_dict[j]
                                           for j in [0, 5, 6, 3]]
valid_precede_dict[fringe_type_dict[6]] = [fringe_type_dict[j]
                                           for j in [1, 4, 7, 2]]

valid_precede_dict[fringe_type_dict[0]] = [fringe_type_dict[j]
                                           for j in [5, 0, 3, 6]]
valid_precede_dict[fringe_type_dict[1]] = [fringe_type_dict[j]
                                           for j in [4, 1, 2, 7]]
valid_precede_dict[fringe_type_dict[3]] = [fringe_type_dict[j]
                                           for j in [4, 1, 2, 7]]
valid_precede_dict[fringe_type_dict[2]] = [fringe_type_dict[j]
                                           for j in [5, 0, 3, 6]]
forward_dh_dict = dict()
forward_dh_dict[(fringe_type_dict[0], fringe_type_dict[7])] = 1
forward_dh_dict[(fringe_type_dict[0], fringe_type_dict[2])] = 0
forward_dh_dict[(fringe_type_dict[0], fringe_type_dict[0])] = 0
forward_dh_dict[(fringe_type_dict[0], fringe_type_dict[5])] = 1

forward_dh_dict[(fringe_type_dict[1], fringe_type_dict[6])] = -1
forward_dh_dict[(fringe_type_dict[1], fringe_type_dict[3])] = 0
forward_dh_dict[(fringe_type_dict[1], fringe_type_dict[1])] = 0
forward_dh_dict[(fringe_type_dict[1], fringe_type_dict[4])] = -1

forward_dh_dict[(fringe_type_dict[2], fringe_type_dict[6])] = -1
forward_dh_dict[(fringe_type_dict[2], fringe_type_dict[3])] = 0
forward_dh_dict[(fringe_type_dict[2], fringe_type_dict[1])] = 0
forward_dh_dict[(fringe_type_dict[2], fringe_type_dict[4])] = -1


forward_dh_dict[(fringe_type_dict[3], fringe_type_dict[7])] = 1
forward_dh_dict[(fringe_type_dict[3], fringe_type_dict[2])] = 0
forward_dh_dict[(fringe_type_dict[3], fringe_type_dict[0])] = 0
forward_dh_dict[(fringe_type_dict[3], fringe_type_dict[5])] = 1


forward_dh_dict[(fringe_type_dict[4], fringe_type_dict[3])] = 1
forward_dh_dict[(fringe_type_dict[4], fringe_type_dict[6])] = 0
forward_dh_dict[(fringe_type_dict[4], fringe_type_dict[4])] = 0
forward_dh_dict[(fringe_type_dict[4], fringe_type_dict[1])] = 1


forward_dh_dict[(fringe_type_dict[5], fringe_type_dict[2])] = -1
forward_dh_dict[(fringe_type_dict[5], fringe_type_dict[7])] = 0
forward_dh_dict[(fringe_type_dict[5], fringe_type_dict[5])] = 0
forward_dh_dict[(fringe_type_dict[5], fringe_type_dict[0])] = -1


forward_dh_dict[(fringe_type_dict[6], fringe_type_dict[2])] = -1
forward_dh_dict[(fringe_type_dict[6], fringe_type_dict[7])] = 0
forward_dh_dict[(fringe_type_dict[6], fringe_type_dict[5])] = 0
forward_dh_dict[(fringe_type_dict[6], fringe_type_dict[0])] = -1


forward_dh_dict[(fringe_type_dict[7], fringe_type_dict[3])] = 1
forward_dh_dict[(fringe_type_dict[7], fringe_type_dict[6])] = 0
forward_dh_dict[(fringe_type_dict[7], fringe_type_dict[4])] = 0
forward_dh_dict[(fringe_type_dict[7], fringe_type_dict[1])] = 1


def format_frac(fr):
    sp = str(fr).split('/')
    if len(sp) == 1:
        return sp[0]
    else:
        return r'$\frac{%s}{%s}$' % tuple(sp)


def latex_print_pairs():
    '''
    Prints out the information in the look-up dictionaries in latex
    format for checking that they match what I think they should be
    '''

    color_format_dict = {1: 'Green4', -1: 'black'}
    charge_format_dict = {1: '\subset', -1: '\supset', 0: '|'}
    hint_format_dict = {1: '+', -1: '-', 0: ''}
    format_dicts = [color_format_dict, charge_format_dict, hint_format_dict]
    fringe_smy_fmt_str = r'{{\color{{{0}}}{1}_{{{2}}}}}^{{{{\color{{red}}{3}}}}}'
    format_fringe = lambda x: fringe_smy_fmt_str.format(*([_f[_v] for _f, _v in
            zip(format_dicts, x)] + [fringe_type_dict[x]]))

    key_lst = []
    res_lst = []
    for j, ft in enumerate(fringe_type_list):
        prece_lst = valid_precede_dict[ft]
        follow_lst = valid_follow_dict[ft]
        fmt_str = (r'{{\color{{DodgerBlue2}}\left[}}' +
                   r'{0}{{\color{{DodgerBlue2}},}} ' +
                   r'{1}{{\color{{DodgerBlue2}},}} {2}{{\color{{DodgerBlue2}},}} ' +
                   r'{3}{{\color{{DodgerBlue2}}' +
                   r'\right]}} ')
        res1 = fmt_str.format(*[format_fringe(_f) for _f in prece_lst])
        res2 = r'&\Leftarrow{0}\Rightarrow&'.format((format_fringe(ft)))
        res3 = fmt_str.format(*[format_fringe(_f) for _f in follow_lst])

        res_lst.append(res1 + res2 + res3)
        ft_fmt = '({0}, \quad{1},\quad {2})'.format(*ft)
        print ft_fmt
        key = r'{{\color{{red}}{0}}}&{{\color{{DodgerBlue2}}{1} }}: &{2}'.format(
            j, ft_fmt, format_fringe(ft))
        key_lst.append(key)

    return (('\\begin{eqnarray*}\n' +
             '\\\\\n'.join(key_lst) +
             '\n\end{eqnarray*}'),
            ('\\begin{eqnarray*}\n' +
             '\\\\\n'.join(res_lst) +
             '\n\end{eqnarray*}'))


### iterator toys
def pairwise_periodic(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..., (sn,s0)

    modified from example in documentation
    """
    a, b = tee(iterable)
    b = cycle(b)
    next(b, None)
    return izip(a, b)


class Fringe(object):
    '''
    Version of :py:class:`Point1D_circ` for representing fringes

    '''

    def __init__(self, f_class, f_loc, frame_number):
        self.frame_number = frame_number
        self.f_class = f_class            #: fringe class
        self.f_loc = f_loc                #: fringe location

        # linked list for space
        self.next_P = None   #: next fringe in angle
        self.prev_P = None   #: prev fringe in angle

        self._fdh = None
        self._rdh = None

        self.region = 0     #: region of the khymograph
        self.abs_height = None   #: the height of this fringe as given
                                 #by tracking in time

    def __eq__(self, other):
        # test the object like things
        for k in ['f_class',
                  'f_loc']:
            if not getattr(self, k) == getattr(other, k):
                return False
        # test links
        for k in ['next_P',
                  'prev_P']:
            _s = getattr(self, k)
            _o = getattr(other, k)
            if _s is None and _o is None:
                continue
            try:
                if not ((_s.f_class == _o.f_class) and
                        (_s.f_loc == _o.f_loc)):
                    return False
            except:
                return False
        # test the numpy-like things
        for k in ['forward_dh', 'reverse_dh', 'abs_height', 'frame_number',
                  'region']:
            try:
                np.testing.assert_equal(getattr(self, k), getattr(other, k))
            except AssertionError:
                print k
                return False
        return True

    @property
    def q(self):
        return self.f_loc.q

    @property
    def phi(self):
        return self.f_loc.phi

    def insert_ahead(self, other):
        '''
        Inserts `other` ahead of this Fringe in the spatial linked-list
        '''
        if self.next_P is not None:
            self.next_P.prev_P = other
            other.next_P = self.next_P

        self.next_P = other
        other.prev_P = self

        self._fdh = None
        other._rdh = None

    def insert_behind(self, other):
        '''
        Inserts other behind this Fringe in the spatial linked-list
        '''
        if self.prev_P is not None:
            self.prev_P.next_P = other
            other.prev_P = self.prev_P

        self.prev_P = other
        other.next_P = self

        other._fdh = None
        self._rdh = None

    def remove_R(self):
        '''
        Removes this Fringe from the spatial linked-list
        '''
        if self.prev_P is not None:
            self.prev_P.next_P = self.next_P
        if self.next_P is not None:
            self.next_P.prev_P = self.prev_P

        self.prev_P._fdh = None
        self.next_P._rdh = None

    def valid_follower(self, other):
        return other.f_class in valid_follow_dict[self.f_class]

    @property
    def forward_dh(self):
        if self._fdh is None:
            other = self.next_P
            if other is None:
                self._fdh = np.nan
            elif self.valid_follower(other):
                self._fdh = forward_dh_dict[(self.f_class, other.f_class)]
            else:
                self._fdh = np.nan

        return self._fdh

    @property
    def reverse_dh(self):
        if self._rdh is None:
            other = self.prev_P
            if other is None:
                self._rdh = np.nan
            elif other.valid_follower(self):
                # need the negative, as we want the step from this one
                # _to_ that one
                self._rdh = -forward_dh_dict[(other.f_class, self.f_class)]
            else:
                self._rdh = np.nan
        return self._rdh


def format_fringe_txt(f):
    return ''.join([_d[_f] for _d, _f in izip([{1: 'B', -1: 'D'},
                                               {1: 'L', 0: '_', -1: 'R'},
                                               {1: 'P', 0: '_', -1: 'S'}], f)])


class FringeRing(object):
    '''
    A class to carry around Fringe data
    '''

    @classmethod
    def from_mbe(cls, mbe, reclassify=False):
        '''Extracts the data from the mbe and passes the relevant
        data structures to the `__init__` function.
        '''
        frame_number = mbe.frame_number
        f_classes, f_locs = _get_fc_lists(mbe, reclassify)

        return cls(frame_number, f_classes, f_locs)

    def __init__(self, frame_number, f_classes, f_locs):
        self.frame_number = frame_number
        self.fringes = [Fringe(fcls, floc, frame_number)
                        for fcls, floc in izip(f_classes, f_locs)]
        self.fringes.sort(key=lambda x: x.phi)

    def link_fringes(self, region_starts, region_labels, region_ends,
                    fringe_starts, fringe_labels, fringe_ends,
                    N_samples, length=2*np.pi):
        '''
        Given the region edges, properly link the fringes which are adjacent.

        Parameters
        ----------
        region_starts : list-like
            the sample where the regions start

        region_labels : list-like
            The labels on the regions

        region_ends : list_like
            The sample where the regions end

        fringe_starts : list-like
            the sample where the fringe starts in the image

        fringe_labels : list-like
            The labels on the image fringe

        fringe_ends : list_like
            The sample where the image fringe ends

        N_samples : int
            The number of samples

        length : float
            The length of the region sampled in the 'natural' units

        '''
        # figure out which bin each fringe goes into just once
        scaled_fringe_loc = [int((np.mod(_fr.phi,
                                         length)/(length)) * N_samples)
                             for _fr in self.fringes]
        bins = [_bin_region(_n, region_starts, region_ends) for
                _n in scaled_fringe_loc]

        f_bins = [_bin_region(_n, fringe_starts, fringe_ends) for
                  _n in scaled_fringe_loc]

        for ((fringe_back, bin_back, fbin_back),
             (fringe_front, bin_front, fbin_front)) in\
                pairwise_periodic(izip(self.fringes, bins, f_bins)):

            fringe_back.abs_height = np.nan

            # handle the other mapping
            if bin_back is None:
                label = 0
            else:
                label = region_labels[bin_back]
            fringe_back.region = label
            fringe_back.abs_height = np.nan

            # handle the linking
            if fbin_back is None or fbin_front is None:
                continue

            # make sure the link is even valid
            if not fringe_back.valid_follower(fringe_front):
                continue

            # make sure we are in adjacent fringes
            fl_max = len(fringe_labels)
            if (fbin_back + 1 == fbin_front) or (fbin_front == 0 and
                                    fbin_back == fl_max - 1):
                fringe_back.insert_ahead(fringe_front)
            # add special case for jumping over extreama one fringe
            # between them and they are the same color and opposite
            # change
            elif (((fbin_back + 2 == fbin_front) or (fbin_front == 0 and
                    fbin_back == fl_max - 2)) and
                    ((fringe_front.f_class.color ==
                         fringe_back.f_class.color) and
                    (fringe_front.f_class.charge ==
                     -fringe_back.f_class.charge))):
                fringe_back.insert_ahead(fringe_front)

    def __iter__(self):
        return self.fringes.__iter__()

    def __len__(self):
        return len(self.fringes)

    def __getitem__(self, key):
        return self.fringes[key]

    def __eq__(self, other):
        if self.frame_number != other.frame_number:
            return False
        return all(_f1 == _f2 for _f1, _f2
                   in izip(self.fringes, other.fringes))


def _get_fc_lists(mbe, reclassify):
    """ Generate `f_class` and `f_loc` lists from a backend object

    Parameters
    ----------
    mbe: `MemoryBackend`
        data backend for the frame data
    reclassify: `Bool`
        if the location/class should be re-computed from the raw
        fringes.  If `False`, use the results chached in the hdf file
    """
    colors = [-1, 1]
    f_classes = deque()
    f_locs = deque()
    junk_fringes = []

    # get center
    center = mbe.curve.cntr.reshape(2, 1)
    if reclassify:
        for color, trk_lst in izip(colors, mbe.trk_lst):
            for t in trk_lst:
                t.classify2()
                if t.charge is not None:
                    xy = mbe.curve.q_phi_to_xy(0, t.phi) - center
                    th = np.mod(np.arctan2(xy[1], xy[0]),
                                2*np.pi)
                    f_locs.append(fringe_loc(t.q, th))
                    f_classes.append(fringe_cls(color, t.charge, 0))
                else:
                    junk_fringes.append(t)
    else:
        for res_lst, color in izip(mbe.res, colors):
            charge_lst, phi_lst, q_lst = res_lst
            XY = mbe.curve.q_phi_to_xy(q_lst, phi_lst) - center
            th = np.mod(np.arctan2(XY[1], XY[0]), 2*np.pi)
            for charge, theta, q in izip(charge_lst, th, q_lst):
                f_locs.append(fringe_loc(q, theta))
                f_classes.append(fringe_cls(color, charge, 0))

    f_classes, f_locs = zip(*sorted(zip(f_classes, f_locs),
                                    key=lambda x: x[1][1]))
    # TODO: deal with junk fringes in sensible way

    return list(f_classes), list(f_locs)


class Region_map(object):
    def __eq__(self, other):
        # test numpy like things
        try:
            np.testing.assert_equal(self.label_img, other.label_img)
            np.testing.assert_equal(self.height_map, other.height_map)
            np.testing.assert_equal(self.working_img, other.working_img)
            np.testing.assert_equal(self.frame_index, other.frame_index)
        except AssertionError:
            return False
        # test fringe rings
        if not all(_fr1 == _fr2 for _fr1, _fr2 in
                   izip(self.fringe_rings, other.fringe_rings)):
            return False
        # test region edges
        if not all(_r1 == _r2 for _r1, _r2 in
                   izip(self.region_edges, other.region_edges)):
            return False
        # don't bother to test the parameters
        return True

    @classmethod
    def from_backend(cls, backend, mask_fun,
                     f_slice=None,
                     reclassify=False,
                     N=2**12, status_output=False,
                     **kwargs):
        '''
        Constructor style class method

        extra kwrags are passed through to `__init__` and from_working_img

        Parameters
        ----------
        backend : HdfBackend
            Data source

        f_slice : None or slice
            slice object setting which frames from the backend should be used

        reclassify : bool
            If the fringes should be reclassified

        mask_fun : function
            Used to filter the light/dark masks.  Must take one argument
            of an `ndarray` and return an  `ndarray` of the same size.

            The function takes in the `working_img` and returns two masks
            corresponding to the light and dark regions.

        N : int
            Number of sample to take around rim

        Returns
        -------

        ret : Region_map
        '''

        if f_slice is None:
            f_slice = slice(None)
        img_bck_grnd_slices = []
        fringe_rings = []
        sample_theta = np.linspace(0, 2*np.pi, N)
        intep_func = scipy.interpolate.interp1d
        for j in xrange(*f_slice.indices(len(backend))):
            if status_output and (j % 1000 == 0):
                print j
            mbe = backend.get_frame(j, get_img=True, raw=reclassify)

            fringe_rings.append(FringeRing.from_mbe(mbe,
                                                    reclassify=reclassify))
            if reclassify:
                curve = mbe.get_next_spline()
            if hasattr(mbe, 'next_curve') and mbe.next_curve is not None:
                curve = mbe.next_curve
            else:
                curve = mbe.curve
            img = mbe.img
            # get center
            center = curve.cntr

            XY = np.vstack(curve.q_phi_to_xy(0, sample_theta))
            # slice the image
            sliced_data = map_coordinates(img,
                                          XY[::-1],
                                          order=2).astype(np.float16)
            # sample_theta != theta so re-sample _again_
            theta = np.arctan2(XY[1] - center[1], XY[0] - center[0])
            theta = np.mod(theta, 2*np.pi)
            indx = np.argsort(theta)

            theta = theta[indx]
            sliced_data = sliced_data[indx]
            # pad so that we have one point periodic
            theta = np.r_[theta[-1] - 2*np.pi,
                          theta,
                          theta[0] + 2 * np.pi]
            sliced_data = np.r_[sliced_data[-1],
                                sliced_data,
                                sliced_data[0]]
            # generate the interpolate object
            f = intep_func(theta, sliced_data)
            # get values and shove into accumulation list
            img_bck_grnd_slices.append(f(sample_theta))

        working_img = np.vstack(img_bck_grnd_slices).T
        del img_bck_grnd_slices
        return cls.from_working_img(
            working_img, fringe_rings, mask_fun,
            frame_indx=np.arange(*f_slice.indices(len(backend))), **kwargs)

    @classmethod
    def from_working_img(cls, working_img, fringe_rings, mask_fun, thresh,
                     size_cut=100, link_threshold=5,
                     conflict_threshold=2, frame_indx=None,
                     **kwargs):
        '''
        Generates a Region_map object from a kymograph
        and a set of fringe rings

        Parameters
        ----------
        working_img : ndarray
            Kymograph

        fringe_rings : list of FringeRing objects
            Fringe data

        mask_fun : function
            Used to filter the light/dark masks.  Must take one argument
            of an `ndarray` and return an  `ndarray` of the same size.

            The function takes in the `working_img` and returns two masks
            corresponding to the light and dark regions.

        thresh : float
            threshold for labeling dark/light regions.  1 \pm thresh

        size_cut : int
            The minimum size of a segmented region

        link_threshold : int
            How many valid fringes are needed between two regions to get
            a dh between them

        conflict_threshold : int
            The number of conflicts (number of times that looking forwards
            and looking backwards between a pair of fringes is
            inconstant) required to get a region blacklisted.

            Passed to _boot_strap


        Returns
        -------
        ret : Region_map

        '''
        if frame_indx is None:
            frame_indx = np.arange(working_img.shape[1])
        up_mask_dt, down_mask_dt = mask_fun(working_img, thresh)

        lab_bright_regions, nb_br = _label_regions(up_mask_dt,
                                                        size_cut)
        lab_dark_regions, nb_dr = _label_regions(down_mask_dt,
                                                      size_cut)

        lab_dark_regions[lab_dark_regions > 0] += nb_br

        assert np.sum(lab_dark_regions * lab_bright_regions) == 0

        label_regions = np.asarray(lab_dark_regions + lab_bright_regions,
                                        dtype=np.uint32)
        N = nb_br + nb_dr

        region_edges = [_segment_labels(region_list)
                             for region_list in label_regions.T]

        fringe_edges = [_segment_fringes(fringe_slice)
                             for fringe_slice in working_img.T]

        for (FR, (region_starts,
                 region_labels,
                 region_ends),
                (fringe_starts,
                 fringe_labels,
                 fringe_ends)) in izip(fringe_rings,
                                       region_edges,
                                       fringe_edges):
            FR.link_fringes(region_starts, region_labels, region_ends,
                            fringe_starts, fringe_labels, fringe_ends,
                            working_img.shape[0])

        # boot strap up the heights
        height_map, set_by, fails = _boot_strap(N,
                                        fringe_rings,
                                        link_threshold,
                                        conflict_threshold=conflict_threshold)
        RM = cls(fringe_rings, region_edges, working_img, height_map,
                   thresh=thresh, size_cut=size_cut, frame_indx=frame_indx,
                   **kwargs)
        # stash diagnostics about boot strapping
        RM._set_by = set_by
        RM._fails = fails
        return RM

    def __init__(self, fringe_rings, region_edges, working_img,
                 height_map, frame_indx,
                 **kwargs):
        # fringes group by a per-time basis
        self.fringe_rings = fringe_rings
        # edges of the regions on a per-time basis
        self.region_edges = region_edges
        # the raw image not sure why we are carrying this around)
        self.working_img = working_img
        # the mapping between regions and heights
        self.height_map = height_map
        # maps the columns of working_img to frames
        self.frame_indx = frame_indx

        # image of the heights
        self._height_img = None
        # image of the labeled regions
        self._label_img = None
        # re-sampled height image
        self._resampled_height = None
        # dict to hold parameters
        self.params = kwargs

        self._rs_func = self._resample_height2D_savgol
        self._rs_kwargs = dict()
        pass

    def __len__(self):
        '''
        Returns number of frames in this region map
        '''
        return self.working_img.shape[1]

    # only make this if we _need_ it
    @property
    def height_img(self):
        if self._height_img is None:
            self._height_img = self.reconstruct_height_img()
        return self._height_img

    # only make this if we _need_ it
    @property
    def label_img(self):
        if self._label_img is None:
            self._label_img = self.reconstruct_label_img()
        return self._label_img

    def display_height(self, ax=None, cmap='jet', bckgnd=True,
                       alpha=.65, t_scale=1, t_units=''):

        height_img = self.height_img

        if ax is None:
            # make this smarter
            ax = plt.gca()

        my_cmap = cm.get_cmap(cmap)
        my_cmap.set_bad('k', alpha=.5)

        frac_size = 4
        step = fractions.Fraction(1, frac_size)
        ax.set_yticks([np.pi * j * step for j in range(2 * frac_size + 1)])
        ax.set_yticklabels([format_frac(j * step) + '$\pi$'
                            for j in range(2 * frac_size + 1)])

        ax.set_xlabel(' '.join([r'$\tau$', t_units.strip()]))
        ax.set_ylabel(r'$\theta$')

        ax.imshow(-height_img,
                  interpolation='none',
                  cmap=my_cmap,
                  extent=[0, (height_img.shape[1] - 1) * t_scale,
                          0, 2 * np.pi],
                  aspect='auto',
                  origin='bottom',
                  )
        if bckgnd:
            ax.imshow(self.working_img,
                      interpolation='none',
                      cmap='gray',
                      extent=[0, (height_img.shape[1] - 1) * t_scale,
                              0, 2 * np.pi],
                      aspect='auto',
                      origin='bottom',
                      alpha=alpha)
        ax.format_coord = self.format_factory(xscale=t_scale,
            yscale=2*np.pi / self.working_img.shape[0])
        ax.figure.canvas.draw()

    def display_all_regions(self, ax=None, cmap='jet',
                            t_scale=1, t_units=''):
        if ax is None:
            # make this smarter
            ax = plt.gca()

        my_cmap = cm.get_cmap(cmap)
        my_cmap.set_under('w', alpha=0)

        frac_size = 4
        step = fractions.Fraction(1, frac_size)
        ax.set_yticks([np.pi * j * step for j in range(2 * frac_size + 1)])
        ax.set_yticklabels([format_frac(j * step) + '$\pi$'
                            for j in range(2 * frac_size + 1)])

        ax.set_xlabel(' '.join([r'$\tau$', t_units.strip()]))
        ax.set_ylabel(r'$\theta$')

        im = ax.imshow(self.label_img,
                  interpolation='none',
                  cmap=my_cmap,
                  extent=[0, (self.working_img.shape[1]) * t_scale,
                          0, 2 * np.pi],
                  aspect='auto',
                  origin='bottom',
                  )
        im.set_clim([1, len(self.height_map)])
        ax.format_coord = self.format_factory(xscale=t_scale,
            yscale=2*np.pi / self.working_img.shape[0])
        ax.figure.canvas.draw()

    def reconstruct_height_img(self):
        '''
        Reconstructs the height image

        Parameters
        ----------

        Returns
        -------
        tmp : ndarray
            ndarray of the same shape
        '''
        tmp = np.zeros(self.working_img.shape) * np.nan
        for j, edges in enumerate(self.region_edges):
            for (r_start, r_label, r_stop) in zip(*edges):
                tmp[r_start:r_stop, j] = self.height_map[r_label]

        return tmp

    def reconstruct_label_img(self):
        '''
        Reconstructs the label image

        Parameters
        ----------

        Returns
        -------
        tmp : ndarray
            ndarray of the same shape
        '''
        tmp = np.zeros(self.working_img.shape)
        for j, edges in enumerate(self.region_edges):
            for (r_start, r_label, r_stop) in zip(*edges):
                tmp[r_start:r_stop, j] = r_label

        # hack to deal with obob
        tmp[-1, :] = tmp[-2, :]
        return tmp

    def _frame_fringe_positions(self,
                           frame_num,
                            length=2*np.pi):
        """
        Returns the locations of where fringes folding in all available
        information
        """
        local_tuple = namedtuple('local_tuple', ['regions', 'fringes'])
        # get fringe ring
        FR = self.fringe_rings[frame_num]
        # re-set fringes
        for _fr in FR:
            _fr.abs_height = np.nan
        # get the region edges
        region_edges = self.region_edges[frame_num]
        # get image edges
        image_edges = _segment_fringes(self.working_img[:, frame_num])
        if len(image_edges.labels) < 5:
            # not enough to do anything with, bail
            return [], []
        # set up working data
        work_data = [local_tuple([], [])
                     for k in xrange(len(image_edges.labels))]
        work_height_map = np.ones(len(image_edges.labels),
                                  dtype=np.float) * np.nan
        fail_flags = np.zeros(len(image_edges.labels), dtype=np.bool)

        N_samples = self.working_img.shape[0]
        for _fr in FR:
            # figure out which image region the fringe falls into
            _bin = _bin_region(int((np.mod(_fr.phi,
                                           length)/(length)) * N_samples),
                               image_edges.starts,
                               image_edges.ends)
            # if it falls in a bin, add it to the working list
            if _bin is not None:
                work_data[_bin].fringes.append(_fr)

        for rs, rl, re in zip(*region_edges):
            # get the image region that the region start is in
            _bin_start = _bin_region(rs,
                                     image_edges.starts,
                                     image_edges.ends)
            # get the image region that the region start is in
            _bin_end = _bin_region(re,
                                   image_edges.starts,
                                   image_edges.ends)
            # if they are in the same bin yay !
            if _bin_start == _bin_end:
                # if both are None, continue
                if _bin_start is None:
                    continue
                # else, add to work list
                work_data[_bin_end].regions.append(rl)
            # if just the end is out side of a region
            elif _bin_end is None:
                # use the start
                work_data[_bin_start].regions.append(rl)
            # if just the start is outside of a region
            elif _bin_start is None:
                # use the end
                work_data[_bin_end].regions.append(rl)
            else:
                # this means that the region spans two image regions,
                # don't like this mark both image regions as bad
                fail_flags[_bin_end] = True
                fail_flags[_bin_start] = True

        for j, wd in enumerate(work_data):
            # skip regions we didn't like above
            if fail_flags[j]:
                continue
            trial_height = np.nan
            heights = []
            # if we have more than 0 region in this image region
            if len(wd.regions) > 0:
                # get the non-nan height
                heights = [self.height_map[r] for r in wd.regions
                           if ~np.isnan(self.height_map[r])]
                if len(heights) > 0:
                    trial_height = heights[0]
                    if not __builtin__.all(trial_height == h
                                           for h in heights[1:]):
                        # not all the same
                        trial_height = np.nan

            # if the trial height is not nan
            if not np.isnan(trial_height):
                # set the fringes height
                for _fr in wd.fringes:
                    _fr.abs_height = trial_height

            work_height_map[j] = trial_height

        # walk the fringes
        work_list = deque()
        # find fringes with labeled heights and with neightbors
        for _f in FR:
            # if the region has a height, and the fringe has unlabeled
            # neighbors
            if (~np.isnan(_f.abs_height) and
                ((_f.next_P is not None and np.isnan(_f.next_P.abs_height)) or
                    (_f.prev_P is not None and
                     np.isnan(_f.prev_P.abs_height)))):
                work_list.append(_f)

        while len(work_list) > 0:
            # grab a fringe to work on
            cur = work_list.pop()
            assert ~np.isnan(cur.abs_height), (cur.abs_height,
                                               cur.forward_dh,
                                               cur.reverse_dh)
            # has foward link, the forward fringe is un-heighted and
            # the forward step is valid
            if (cur.next_P is not None and np.isnan(cur.next_P.abs_height) and
                  ~np.isnan(cur.forward_dh)):
                cur.next_P.abs_height = cur.abs_height + cur.forward_dh
                work_list.append(cur.next_P)
            # same for reverse
            if (cur.prev_P is not None and np.isnan(cur.prev_P.abs_height) and
                  ~np.isnan(cur.reverse_dh)):
                cur.prev_P.abs_height = cur.abs_height + cur.reverse_dh
                work_list.append(cur.prev_P)

        for j, wd in enumerate(work_data):
            # skip regions we didn't like above
            if fail_flags[j]:
                continue
            # if this image region already has a height don't tempt fate
            if ~np.isnan(work_height_map[j]):
                continue
            # did any of the fringes picked up a height
            heights = [_fr.abs_height for _fr in wd.fringes
                       if ~np.isnan(_fr.abs_height)]
            if len(heights) > 0:
                trial_height = heights[0]
                if __builtin__.all((trial_height == h for h in heights[1:])):
                    work_height_map[j] = trial_height

        if np.all(np.isnan(work_height_map)):
            return [], []

        phi, h = [], []
        #deal with wrap-around
        ir_s_f, ir_l_f, ir_e_f = [p[0] for p in image_edges]
        ir_s_L, ir_l_L, ir_e_L = [p[-1] for p in image_edges]

        post_list = None
        # over lapping regions
        if ir_s_f == 0 and ir_e_L == N_samples and ir_l_f == ir_l_L:
            if ~np.isnan(work_height_map[0]) or ~np.isnan(work_height_map[-1]):
                if np.isnan(work_height_map[0]):
                    work_height_map[0] = work_height_map[-1]

                elif np.isnan(work_height_map[-1]):
                    work_height_map[-1] = work_height_map[0]

                cent = ir_s_L + ((N_samples - ir_s_L) + ir_e_f)/2
                if cent < N_samples:
                    post_list = [[2 * np.pi * cent / N_samples],
                                 [work_height_map[0]]]
                else:
                    cent = cent - N_samples
                    phi.append(2*np.pi * cent / N_samples)
                    h.append(work_height_map[0])
        # non-wrapped around region
        else:
            if not np.isnan(work_height_map[-1]):
                post_list = [[np.pi * (ir_s_L + ir_e_L) / N_samples],
                             [work_height_map[-1]]]
            if not np.isnan(work_height_map[0]):
                phi.append(np.pi * (ir_s_f + ir_e_f) / N_samples)
                h.append(work_height_map[0])
        # deal with middle region
        # note, 2s cancel
        _phi, _h = zip(*[(np.pi * (ir_s + ir_e) / (N_samples), _h)
                       for (ir_s, ir_l, ir_e), _h in zip(izip(*image_edges),
                                                         work_height_map)[1:-1]
                        if ~np.isnan(_h)])
        phi.extend(_phi)
        h.extend(_h)
        if post_list is not None:
            phi.extend(post_list[0])
            h.extend(post_list[1])

        return phi, h

    # make getting resampled height easy
    @property
    def resampled_height(self):
        if self._resampled_height is None:
            self._resampled_height = self._rs_func(**self._rs_kwargs)
        return self._resampled_height

    def _resample_height_1D(self,
                        N=1024,
                        intep_func=None,
                        min_range=0,
                        max_range=2*np.pi):
        '''
        Returns a re-sampled and interpolated version of the height map.

        Stashes the result in `self._resampled_height`

        Parameters
        ----------
             N : int
                The number of spatial sample points
            intep_func : function
                The function to use for interpolation. Must match specs of
                functions in `scipy.interpolate`
            min_range : float
                minimum value of the interpolation range
            max_range : float
                maximum value of the interpolation range

        Returns
        -------
        img : np.ndarray
            Re-sampled image
        '''
        if intep_func is None:
            #intep_func = scipy.interpolate.InterpolatedUnivariateSpline
            intep_func = scipy.interpolate.interp1d
        tmp = []
        for j in xrange(len(self)):
            phi, h = self._frame_fringe_positions(j)
            phi_new, h_new = _height_interpolate(
                phi, h, N, min_range, max_range, intep_func=intep_func)
            tmp.append(-h_new)

        ret = np.vstack(tmp).T

        return ret

    def _resample_height2D(self,
                          N=1024,
                          intep_func=None,
                          min_th_range=0,
                          max_th_range=2*np.pi,
                          scale=2*np.pi / 100):
        """
        Interpolates
        """
        if intep_func is None:
            intep_func = scipy.interpolate.LinearNDInterpolator

        # look into doing this rolling
        phi_accum = []
        h_accum = []
        tau_accum = []
        for j in xrange(len(self)):
            phi, h = self._frame_fringe_positions(j)
            phi_accum.extend(phi)
            h_accum.extend(h)
            tau_accum.extend(np.ones(len(h)) * j * scale)
            # make sure edges are included
            if len(phi) > 0:
                phi_accum.append(phi[-1] - max_th_range)
                h_accum.append(h[-1])
                phi_accum.append(phi[0] + max_th_range)
                h_accum.append(h[0])
                tau_accum.extend([j * scale] * 2)

        X, Y = np.meshgrid(np.linspace(0, max_th_range, N),
                           np.arange(len(self), dtype='float') * scale)
        intp_obj = intep_func(np.vstack((phi_accum, tau_accum)).T,
                              h_accum)
        ret = -intp_obj(
            np.vstack((X.ravel(), Y.ravel())).T).reshape(X.shape).T

        return ret

    def _resample_height2D_savgol(self, k_window=15, k_order=2,
                                t_window=5, t_order=2, **kwargs):
        """
        Does 2D interpolation on the theta, tau location of the fringe/regions

        Applies a Savitzky-Golay in both the k (along the rim at fixed
        time) and t (fixed location through time) to the result to smooth
        out junk.

        This seems to be the most robust method.  Relies on a local
        modification to scipy.signal.savgol_filter to allow wrap as
        a mode

        Parameters
        ----------
        k_window : odd int
            Size of the window used for along rim filtering

        k_order : positive int
            The order polynomial used for along rim filtering

        t_window : odd int
            Size of the window used for along rim filtering

        t_order : positive int
            The order polynomial used for along rim filtering
        """
        raw_int = self._resample_height2D(**kwargs)
        return scipy.signal.savgol_filter(
                    scipy.signal.savgol_filter(raw_int,
                                               k_window, k_order, axis=0,
                                               mode='wrap'),
                 t_window, t_order, axis=1)

    def _resample_height2D_fft_filter(self, k_band=20, scale=None,
                                      **kwargs):

        raw_int = self._resample_height2D(**kwargs)
        fft_tmp = np.fft.fft(raw_int, axis=1)
        fft_tmp[:, (k_band+1):-k_band] = 0
        return np.fft.ifft(fft_tmp, axis=1).T.real

    def display_height_resampled(self, ax=None, cmap='jet',
                                 bckgnd=True, alpha=.65,
                                 t_scale=1, t_units=''):
        height_img = self.resampled_height
        print np.min(height_img), np.max(height_img)
        if ax is None:
            # make this smarter
            ax = plt.gca()

        my_cmap = cm.get_cmap(cmap)
        my_cmap.set_bad('k', alpha=.5)

        frac_size = 4
        step = fractions.Fraction(1, frac_size)
        ax.set_yticks([np.pi * j * step for j in range(2 * frac_size + 1)])
        ax.set_yticklabels([format_frac(j * step) + '$\pi$'
                            for j in range(2 * frac_size + 1)])

        ax.set_xlabel(' '.join([r'$\tau$', t_units.strip()]))
        ax.set_ylabel(r'$\theta$')

        ax.imshow(height_img,
                  interpolation='none',
                  cmap=my_cmap,
                  extent=[0, (self.working_img.shape[1] - 1) * t_scale,
                          0, 2 * np.pi],
                  aspect='auto',
                  origin='bottom',
                  )
        if bckgnd:
            ax.imshow(self.working_img,
                      interpolation='none',
                      cmap='gray',
                      extent=[0, (self.working_img.shape[1] - 1) * t_scale,
                              0, 2 * np.pi],
                      aspect='auto',
                      origin='bottom',
                      alpha=alpha)
        ax.format_coord = self.format_factory(xscale=t_scale,
            yscale=2*np.pi / self.working_img.shape[0])
        ax.figure.canvas.draw()

    def display_region(self, n, ax=None):

        if ax is None:
            # make this smarter
            ax = plt.gca()

        data = self.label_img

        norm_br = matplotlib.colors.Normalize(vmin=.5,
                                              vmax=np.nanmax(data), clip=False)
        my_cmap = cm.get_cmap('jet')
        my_cmap.set_under(alpha=0)

        ax.imshow(n * (data == n),
                  cmap=my_cmap,
                  norm=norm_br,
                  aspect='auto',
                  interpolation='none'
                  )
        ax.figure.canvas.draw()

    def display_regions(self, ns, ax=None):

        if ax is None:
            # make this smarter
            ax = plt.gca()

        data = self.label_img

        norm_br = matplotlib.colors.Normalize(vmin=.5,
                                              vmax=np.max(data), clip=False)
        my_cmap = cm.get_cmap('jet')
        my_cmap.set_under(alpha=0)
        tmp = np.zeros(data.shape, dtype='uint32')

        for n in ns:
            tmp[data == n] = n

        ax.imshow(tmp,
                  cmap=my_cmap,
                  norm=norm_br,
                  aspect='auto',
                  interpolation='none'
                  )
        ax.figure.canvas.draw()

    def get_height(self, frame_num, theta):
        theta_indx = int((np.mod(theta, 2 * np.pi) / (2 * np.pi)) *
                         self.working_img.shape[0])
        label = self.label_img[theta_indx, frame_num]
        return self.height_map[label]

    def display_height_taper(self, h5_backend, ax,
                             f_slice=None, t_scale=1, c_scale=1, h_scale=1,
                             **kwargs):
        """

        Displays the height map with the vertical
        dimension as distance around the rim
        instead of angle.

        Extra kwargs are passed to `pcolormesh`

        Parameters
        ----------
        h5_backend : HdfBackend
            Data source to pull the rim distance from
        ax : Axes
           axes to plot to
        t_scale : float
           Scaling factor for time
        c_scale : float
           Scaling factor for distance
        h_scale : float
           Scaling factor for vertical distance

        """
        assert len(h5_backend) >= self.resampled_height.shape[1]
        rs_h_shape = self.resampled_height.shape
        if f_slice is None:
            f_slice = slice(None)

        # make sure we don't load stuff we don't need
        h5_backend.prams = (False, False)
        # get the cumulative lengths
        Y = np.vstack([mbe.curve.cum_length_theta(rs_h_shape[0])
                       for mbe in h5_backend[self.frame_indx[f_slice]]]).T
        # shift
        Y -= np.mean(Y, axis=0)
        # scale
        Y *= c_scale
        # generate X array via tricksy outer product
        X = (np.ones((1, rs_h_shape[0])) *
             (np.arange(*f_slice.indices(rs_h_shape[1])).reshape(-1, 1) *
              (t_scale / h5_backend.frame_rate))).T
        if 'rasterized' not in kwargs:
            kwargs['rasterized'] = True

        # create pcolormesh
        pm = ax.pcolormesh(
            X, Y,
            self.resampled_height[:, self.frame_indx[f_slice]] * h_scale,
            **kwargs)
        pm.set_clim(np.array([-np.nanmax(self.height_map),
                              -np.nanmin(self.height_map)]) * h_scale)
        # force re-draw
        ax.figure.canvas.draw()
        return pm

    def write_to_hdf(self, out_file, md_dict=None, mode='w-'):
        fmt_str = "region_edges_{frn:07}"
        # this will blow up if the file exists
        with closing(h5py.File(out_file.format, mode)) as h5file:

            h5file.attrs['ver'] = '0.2'
            # store all the md passed in

            if md_dict is not None:
                for key, val in md_dict.iteritems():
                    if val is None:
                        continue
                    if isinstance(val, types.FunctionType):
                        val = '.'.join((val.__module__, val.__name__))
                    try:
                        h5file.attrs[key] = val
                    except TypeError:
                        print 'key: ' + key + ' can not be gracefully'
                        print 'shoved into an hdf object, please reconsider'
                        print ' your life choices'

            # save the parametrs
            param_grp = h5file.create_group('params')
            for key, val in self.params.iteritems():
                if val is None:
                    continue
                try:
                    param_grp.attrs[key] = val
                except TypeError:
                    print 'key: ' + key + ' can not be gracefully shoved into'
                    print ' an hdf object, please reconsider your life choices'

            # the kymograph extracted from the image series
            h5file.create_dataset('working_img',
                                  self.working_img.shape,
                                  self.working_img.dtype,
                                  compression='szip')
            h5file['working_img'][:] = self.working_img
            # the frame index
            h5file.create_dataset('frame_indx', data=self.frame_indx)
            # the mapping of region number to bootstrapped height
            h5file.create_dataset('height_map',
                                  self.height_map.shape,
                                  self.height_map.dtype)
            h5file['height_map'][:] = self.height_map

            fr_c_grp = h5file.create_group('fringe_classes')
            fr_l_grp = h5file.create_group('fringe_locs')
            re_grp = h5file.create_group('region_edges')
            for FR, region_edges in izip(self.fringe_rings, self.region_edges):
                f_class = np.vstack([fr.f_class for fr in FR])
                f_loc = np.vstack([fr.f_loc for fr in FR])
                re_data = np.vstack(region_edges)
                frame_number = fr.frame_number
                dset_names = ["f_class_{frn:07}".format(frn=frame_number),
                              "f_loc_{frn:07}".format(frn=frame_number)]
                for n, v, t, _g in izip(dset_names,
                                    [f_class, f_loc],
                                    [np.int8, np.float],
                                    [fr_c_grp, fr_l_grp]):
                    fr_ds = _g.create_dataset(n,
                                          v.shape,
                                          t)
                    if v.shape[0] > 0:
                        fr_ds[:] = v

                re_ds = re_grp.create_dataset(fmt_str.format(frn=frame_number),
                                      re_data.shape,
                                      np.uint32)
                re_ds[:] = re_data

    @classmethod
    def from_hdf(cls, in_file):
        VER_DICT = {'0.2': cls._from_hdf_0_2,
                    }

        with closing(h5py.File(in_file.format, 'r')) as h5file:
            ver = h5file.attrs['ver']
            if ver not in VER_DICT:
                raise Exception("unsupported serialization version")
            return VER_DICT[ver](h5file)

    @classmethod
    def _from_hdf_0_2(cls, h5file):
        print 'starting read'
        # extract parameters
        #        md = dict(h5file.attrs)
        # get working image
        working_img = h5file['working_img'][:]
        # get the height map
        height_map = h5file['height_map'][:]
        if 'frame_indx' in h5file:
            frame_indx = h5file['frame_indx'][:]
        else:
            frame_indx = np.arange(working_img.shape[1])
        # pull out the edges of the regions
        re_grp = h5file['region_edges']
        print 'starting region edges'
        # this relies on the iterator in h5py returning things sorted
        # and the padding being sufficient to always sort correctly
        region_edges = [Region_Edges(*(re_grp[k][:])) for k in re_grp]
        # extract the fringe properties
        fr_c_grp = h5file['fringe_classes']
        fr_l_grp = h5file['fringe_locs']
        print 'starting fringe rings'
        # this relies on the iterator in h5py returning things sorted
        # and the padding being sufficient to always sort correctly
        fringe_rings = [FringeRing(int(_cn.split('_')[-1]),
                                  [fringe_cls(*_c) for _c in fr_c_grp[_cn][:]],
                                  [fringe_loc(*_l) for _l in fr_l_grp[_ln][:]])
                                  for _cn, _ln in izip(fr_c_grp, fr_l_grp)
                                  ]

        fringe_edges = [_segment_fringes(fringe_slice)
                             for fringe_slice in working_img.T]

        print 'starting linking'
        # link the fringes.
        # This is simpler than storing the linking information
        for (FR, (region_starts,
                 region_labels,
                 region_ends),
                (fringe_starts,
                 fringe_labels,
                 fringe_ends)) in izip(
                     fringe_rings, region_edges, fringe_edges):
            FR.link_fringes(region_starts, region_labels, region_ends,
                            fringe_starts, fringe_labels, fringe_ends,
                            working_img.shape[0])

        params = dict(h5file['params'].attrs)

        return cls(fringe_rings, region_edges, working_img,
                   height_map, frame_indx, **params)

    def get_frame_profile(self, j):
        '''
        return a list of tuples (center, dh) for identified regions in frame j

        units of center set by `Region_map._scale`

        Parameters
        ----------
        j : int
            frame number

        Returns
        -------
        ret : list
             list of (center, dh)
        '''
        return list(self.get_frame_profile_gen(j))

    def get_frame_profile_gen(self, j):
        '''
        return a generator of tuples (center, dh) for identified regions
        in frame j.

        units of center set by `Region_map._scale`

        Parameters
        ----------
        j : int frame number

        Returns
        -------
        ret : generator
            yields tuples of form of (center, dh)

        '''
        return (((start + end) / (2 * self._scale), self.height_map[label])
                  for start, label, end in izip(*self.region_edges[j])
                  if not np.isnan(self.height_map[label]))

    def error_check(self):
        error_count = 0
        correct_count = 0
        error_regions = defaultdict(int)
        for FR in self.fringe_rings:
            for fr in FR:
                fdh = fr.forward_dh

                if (not np.isnan(fdh)) and \
                        (not np.isnan(fr.abs_height)) and \
                        (not np.isnan(fr.next_P.abs_height)):
                    if int(fr.next_P.abs_height - fr.abs_height) != fdh:

                        error_count += 1
                        ra, rb = fr.region, fr.next_P.region

                        error_regions[(ra, rb)] += 1

                    else:
                        correct_count += 1

        return error_count, error_regions

    def format_factory(self, xscale=1, yscale=1):
        numrows, numcols = self.working_img.shape

        def format_coord(x, y):
            col = int(x / xscale)
            row = int(y / yscale)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                r_start, r_labels, r_end = self.region_edges[col]
                region_indx = _bin_region(row, r_start, r_end)
                if region_indx is not None:
                    region = r_labels[region_indx]
                else:
                    region = 0

                rs_h = ''
                if self._resampled_height is not None:
                    r = int(row *
                            (self._resampled_height.shape[0] /
                             self.working_img.shape[0]))
                    rs_h = self._resampled_height[r, col]
                return ("x:{x}({col}), " +
                        "y:{y}({row}), " +
                        "r:{reg}, " +
                        "h:{h}, " +
                        "h_rs:{rs_h}, " +
                        "I:{I}").format(x=col * xscale,
                                        y=row * yscale,
                                        reg=region,
                                        h=-self.height_map[region],
                                        rs_h=rs_h,
                                        I=self.working_img[row, col],
                                        row=row,
                                        col=col)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        return format_coord


def _segment_labels(region_list, zero_thresh=2):
    '''
    Segments the regions.  Returns list of where contiguous regions begin
    and end

    parameters
    ----------
    region_list : list
        a list indicating what region each position is in

    zero_thresh : int
        how many zeros are needed to trigger the end of a region
    '''
    region_starts = []
    region_labels = []
    region_ends = []
    cur_region = None
    zero_count = 0
    for j, lab in enumerate(region_list):
        # we are still in the same region, keep going
        if lab == cur_region:
            zero_count = 0
            continue
        # we are
        elif lab == 0:
            if cur_region is not None:
                zero_count += 1
                # if we hit enough zeros in a row, start looking for a
                # new region
                if zero_count > zero_thresh:
                    region_ends.append(j - zero_count)
                    # add end of previous label
                    cur_region = None
                    # reset current label
                    zero_count = 0  #
            continue
        elif lab != cur_region:
            # if the current region is not
            if cur_region is not None:
                region_ends.append(j - zero_count)
            zero_count = 0
            region_labels.append(lab)
            region_starts.append(j)
            cur_region = lab
    if cur_region is not None:
        region_ends.append(len(region_list))

    return Region_Edges(region_starts, region_labels, region_ends)


def _segment_fringes(image_slice, thresh=.1, filter_width=3):
    """
    segments a single column into light/dark fringes

    Parameters
    ----------
    image_slice : ndarray
        The slice to work on

    thresh : float
        defaults to .1, the difference from 1 to be 'light' or 'dark'

    filter_width : int
        width of maximum filter applied to thresholded lines
    """
    dark = scipy.ndimage.filters.maximum_filter1d(image_slice < 1 - thresh,
                                                  filter_width, mode='wrap')
    light = scipy.ndimage.filters.maximum_filter1d(image_slice > 1 + thresh,
                                                   filter_width, mode='wrap')
    # this implements the xor logic, if both are true -> sums to 0, if
    # neither are true stays 0, if only dark is true -> -1, if only
    # light is true -> 1
    return _segment_labels(-1 * dark + light)
    ## tmp_xor = np.logical_xor(dark, light)

    ## dark_dt = np.logical_and(dark, tmp_xor)
    ## light_dt = np.logical_and(light, tmp_xor)
    ## return _segment_labels(-1 * dark_dt + light_dt)

    # f_segs = np.zeros(image_slice.shape, dtype=np.int16)

    # r_indx = 0
    # flag = 0
    # for j, (_l, _d) in enumerate(izip(light_dt, dark_dt)):
    #     if _l:
    #         assert ~_d, 'should never both be true'
    #         if flag != 1:
    #             r_indx += 1
    #             flag = 1
    #     elif _d:
    #         assert ~_l, 'should never both be true'
    #         if flag != -1:
    #             r_indx += 1
    #             flag = -1
    #     else:
    #         flag = 0
    #     f_segs[j] = r_indx * flag * flag
    # return _segment_labels(f_segs)


def _bin_region(N, region_starts, region_ends):
    '''
    Returns what region an index is in given a list of the region edges

    Parameters
    ----------
    N : int
        The index of interest

    region_starts : list
        The index of the first position in the region

    region_ends : list
        The index of the last position in the region
    '''
    n = bisect(region_starts, N) - 1
    if n == -1:
        return None
    if region_ends[n] > N:
        return n
    else:
        return None


def _dict_to_dh(input_d, threshold=15):
    """Converts a (-1, 0, 1) dict -> a single dh.  Returns `None` if
    the conversion is ambiguous.  The conversion in ambiguous if a)
    more than one bin has counts b) the number of counts is less than
    `threshold`.


    Parameters
    ----------
    input_d : dict
        `input_d` contains the keys [-1, 0, 1] with values that are counts
    threshold: int
        the minimum number of counts to be valid

    Returns
    -------
    dh : int, (-1, 0, 1) or `None`
    """
    # check of there is more than one entry with non-zero counts
    if input_d[0] and input_d[-1] and input_d[1]:
        return None

    # if any
    for k in [-1, 0, 1]:
        if input_d[k] > threshold:
            return k
    return None


def _dict_to_dh_Nstep(input_d, threshold=15):
    """Converts a (-1, 0, 1) dict -> a single dh.  Returns `None` if
    the conversion is ambiguous.  The conversion in ambiguous if a)
    more than one bin has counts b) the number of counts is less than
    `threshold`.


    Parameters
    ----------
    input_d : dict
        `input_d` contains the keys [-1, 0, 1] with values that are counts
    threshold: int
        the minimum number of counts to be valid

    Returns
    -------
    dh : int, (-1, 0, 1) or `None`
    """
    # check of there is more than one entry with non-zero counts
    if len(input_d) > 1 or len(input_d) == 0:
        return None

    ((k, v), ) = input_d.items()
    if v > threshold:
        return k

    return None


def _height_interpolate(phi, h, steps, min_range, max_range, intep_func):
    assert len(phi) == len(h), 'mismatched input'
    # generate even sampling
    new_phi = np.linspace(min_range, max_range, steps)

    if len(phi) == 0:
        return new_phi, np.nan * np.ones(new_phi.shape)
    # make periodic
    phi = np.hstack((phi[-1] - max_range, phi, phi[0] + max_range))
    h = np.hstack((h[-1], h, h[0]))

    # set up interpolation
    try:
        f = intep_func(phi, h)
        new_h = f(new_phi)
    except:
        new_h = np.nan * np.ones(new_phi.shape)
    # return new values
    return new_phi, new_h


def _connection_network(N, fringe_rings, dirc='f'):
    """
    Sets up the network between the regions based on what fringes fall in
    them.

    Parameters
    ----------
    N : int
        number of regions

    fringe_rings : list of FringRing objects
        The data used to link the regions

    Returns
    -------
    connections : list of dicts of dicts
        The list is by starting region, the first level of dict is
        keyed by the region the connection is _to_.   The inner
        dict has keys {-1, 0, 1} and counts the number of times a
        pair of fringes has that dh between the two regions.

    """
    inner_dict = lambda: dict({-1: 0, 1: 0, 0: 0})

    link_dict = {'f': 'next_P', 'r': 'prev_P'}
    dh_dict = {'f': 'forward_dh', 'r': 'reverse_dh'}

    dh_str = dh_dict[dirc]
    ln_str = link_dict[dirc]

    # main data structure
    connections = [defaultdict(inner_dict)
                for j in range(N)]
    for FR in fringe_rings:
        for fr in FR:
            if fr is None:
                print 'WTF mate'
                continue
            if fr.region == 0:
                continue
            ln_fr = getattr(fr, ln_str)
            if ln_fr is None:
                continue
            ln_region = ln_fr.region
            if ln_region == 0:
                continue

            dh = getattr(fr, dh_str)
            if not np.isnan(dh):
                dh = int(dh)
                connections[fr.region][ln_region][dh] += 1

    return connections


def _connection_network_Nstep(N, fringe_rings, dirc='f'):
    """
    Sets up the network between the regions based on what fringes fall in
    them.

    Parameters
    ----------
    N : int
        number of regions

    fringe_rings : list of FringRing objects
        The data used to link the regions

    Returns
    -------
    connections : list of dicts of dicts
        The list is by starting region, the first level of dict is
        keyed by the region the connection is _to_.   The inner
        dict has keys {-1, 0, 1} and counts the number of times a
        pair of fringes has that dh between the two regions.

    """
    inner_dict = lambda: defaultdict(int)

    link_dict = {'f': 'next_P', 'r': 'prev_P'}
    dh_dict = {'f': 'forward_dh', 'r': 'reverse_dh'}

    dh_str = dh_dict[dirc]
    ln_str = link_dict[dirc]

    # main data structure
    connections = [defaultdict(inner_dict)
                for j in range(N)]
    for FR in fringe_rings:
        for fr in FR:
            fr_region = fr.region
            if fr is None:
                print 'WTF mate'
                continue
            if fr_region == 0:
                continue
            ln_fr = getattr(fr, ln_str)
            if ln_fr is None:
                continue
            # get the region of the linked to fringe
            ln_region = ln_fr.region
            # if it is zero, walk until we find a non-zero region or
            # run out of connectivity
            if ln_region == 0:
                # start at zero
                accum_dh = 0
                while ln_region == 0:
                    # get the next dh
                    dh = getattr(fr, dh_str)
                    if np.isnan(dh):
                        break
                    # add to the accumulated dh
                    accum_dh += dh
                    # asign the current link fringe to fr
                    fr = ln_fr
                    # get the next link fringe
                    ln_fr = getattr(fr, ln_str)
                    # check if next fringe is none
                    if ln_fr is None:
                        break
                    # get the region of the link fringe
                    ln_region = ln_fr.region
                # only do this if the while condition becomes false
                else:
                    # if we have walked back to our self, no connection
                    if ln_region == fr_region:
                        break
                    # get the last step
                    dh = getattr(fr, dh_str)
                    # if it isn't nan
                    if ~np.isnan(dh):
                        accum_dh += dh
                        connections[fr_region][ln_region][int(accum_dh)] += 1
            # else, we have a one step connection, just assign it
            else:
                if ln_region == fr_region:
                    break
                dh = getattr(fr, dh_str)
                if not np.isnan(dh):
                    dh = int(dh)
                    connections[fr.region][ln_region][dh] += 1

    return connections


def _boot_strap(N, FRs, connection_threshold, conflict_threshold, status_output=False):
    """
    An improved boot-strap operation

    Parameters
    ----------
    N : int
        The number of regions

    FRs : list of FringeRings
        The fringe data structurens

    connection_threshold : int
       The number of fringe connections between two regions to link them

    conflict_threshold : int
       The number of conflicts (number of times that looking forwards
        and looking backwards between a pair of fringes is
        inconstant) required to get a region blacklisted.
    """
    # pre-allocate the connection dicts
    valid_connections = [{} for j in range(N)]
    conflict_flags = np.zeros((N,), dtype=np.uint32)
    # zip together the forward and backwards linking
    for i, (forward_d, backward_d) in enumerate(
            izip(_connection_network_Nstep(N, FRs, 'f'),
                  _connection_network_Nstep(N, FRs, 'r'))):

        tmp_dict = valid_connections[i]

        for dest_region, link_dict in forward_d.items():
            dh = _dict_to_dh_Nstep(link_dict, threshold=connection_threshold)
            if dh is not None:
                tmp_dict[dest_region] = dh
        for dest_region, link_dict in backward_d.items():
            dh = _dict_to_dh_Nstep(link_dict, threshold=connection_threshold)
            if dh is not None:
                if dest_region in tmp_dict and tmp_dict[dest_region] != dh:
                    if status_output:
                        print ("conflict from: {} to: {} " +
                               "old_dh: {} new_dh: {}").format(
                                    i, dest_region, tmp_dict[dest_region], dh)
                    # if we have inconsistent linking (between forward
                    # and backwards), throw everything out this
                    # happens
                    del tmp_dict[dest_region]
                    conflict_flags[dest_region] += 1
                    conflict_flags[i] += 1
                    continue
                tmp_dict[dest_region] = dh

    black_list = np.flatnonzero(conflict_flags > conflict_threshold)
    if status_output:
        print "black list length: {}".format(len(black_list))
    for bl in black_list:
        # we don't want to try any connections with the black-listed regions
        # nuke it's outward connections
        valid_connections[bl] = {}
        # nuke it as an inward connection
        for d in valid_connections:
            if bl in d:
                del d[bl]

    # network computation interlude
    G = networkx.Graph()
    for n in xrange(N):
        G.add_node(n)

    for n, g in enumerate(valid_connections):
        for k, v in g.items():
            G.add_edge(n, k)

    G.remove_nodes_from(networkx.isolates(G))
    res = networkx.connected_component_subgraphs(G)

    # pick a node in the largest connected component
    start = res[0].node.keys()[0]

    # clear height map
    height_map = np.ones(N, dtype=np.float32) * np.nan
    #set first height
    height_map[start] = 0

    set_by = dict()
    fails = deque()
    work_list = []
    for e in ((abs(v), start, k) for k, v in valid_connections[start].items()):
        heapq.heappush(work_list, e)
    while len(work_list) > 0:
        _, a, b = heapq.heappop(work_list)
        assert a != b, "should never link a region to it's self"
        prop_height = height_map[a] + valid_connections[a][b]
        if np.isnan(height_map[b]):
            height_map[b] = prop_height
            for e in ((abs(v), b, k) for k, v in valid_connections[b].items()):
                heapq.heappush(work_list, e)

            set_by[b] = a
        else:
            if height_map[b] != prop_height:
                if status_output:
                    print ("from {}({}) to {}({}) proposed delta:" +
                           " {} current delta: {}").format(
                        a, height_map[a], b, height_map[b],
                        valid_connections[a][b],
                        height_map[b] - height_map[a])
                fails.append((a, b))

    return height_map, set_by, fails


def _label_regions(mask, size_cut):
    '''
    Labels the regions

    Parameters
    ----------
    mask : binary ndarray
        The array to identify regions in.  Assumes that
        the masks are pre-filtered

    size_cut : int
        Maximum number of pixels to be in a
    '''

    lab_regions, nb = ndi.label(mask)
    # loop to make periodic
    for j in range(lab_regions.shape[1]):
        top_lab = lab_regions[0, j]
        bot_lab = lab_regions[-1, j]
        if top_lab != bot_lab and top_lab != 0 and bot_lab != 0:
            lab_regions[lab_regions == bot_lab] = top_lab

    sizes = ndi.sum(mask, lab_regions, range(nb + 1))
    mask_sizes = sizes < size_cut
    #    print len(sizes), sum(mask_sizes)

    remove_pix = mask_sizes[lab_regions]
    lab_regions[remove_pix] = 0
    labels = np.unique(lab_regions)
    lab_regions = np.searchsorted(labels, lab_regions)
    return lab_regions, len(labels)


def filter_fun(working_img, thresh, struct=None):
    print 'threshold {}'.format(thresh)
    if struct is None:
        #        struct = ndi.morphology.generate_binary_structure(2, 1)
        struct = [[1, 1, 1]]
        #    struct = np.ones((3, 3))

    up_mask = ndi.binary_dilation(working_img > 1 + thresh,
                                  structure=struct,
                                  iterations=1)
    down_mask = ndi.binary_dilation(working_img < 1 - thresh,
                                    structure=struct,
                                    iterations=1)

    up_mask_dt = np.logical_and(up_mask, ~down_mask)
    down_mask_dt = np.logical_and(down_mask, ~up_mask)

    return up_mask_dt, down_mask_dt


def filter_fun_orig(working_img, thresh, struct=None):

    if struct is None:
        #        struct = ndi.morphology.generate_binary_structure(2, 1)
        struct = [[1, 1, 1]]
        #    struct = np.ones((3, 3))

    up_mask = ndi.binary_erosion(working_img > 1 + thresh,
                                 structure=struct,
                                 iterations=1,
                                 border_value=1)
    down_mask = ndi.binary_erosion(working_img < 1 - thresh,
                                   structure=struct,
                                   iterations=1,
                                   border_value=1)

    return up_mask, down_mask


def texture_std_power(RM, k_list, f_slice=None):
    """
    Returns a measure of the texture based on the
    variance

    Parameters
    ----------
    k_list : list
       The modes to extract data for

    f_slice : slice or None
       The frames to extract data for

    Returns
    -------
    tuple : a scalar measure of the texture
    """
    if f_slice is None:
        f_slice = slice(None, None, None)
    tmp_fft = np.fft.fft(RM.resampled_height[:, f_slice], axis=0)
    return np.var(np.abs(tmp_fft[k_list, :]), axis=1)


def texture_mean_power(RM, k_list, f_slice=None):
    """
    Returns a measure of the texture based on the
    mean

    Parameters
    ----------
    k_list : list
       The modes to extract data for

    f_slice : slice or None
       The frames to extract data for

    Returns
    -------
    tuple : a scalar measure of the texture
    """
    if f_slice is None:
        f_slice = slice(None, None, None)
    tmp_fft = np.fft.fft(RM.resampled_height[:, f_slice], axis=0)
    return np.mean(np.abs(tmp_fft[k_list, :]), axis=1)


def texture_std_angle(RM, k_list, f_slice=None):
    """
    Estimate the texture by looking at the phase of the
    fft change

    Parameters
    ----------
    k_list : list
       The modes to extract data for

    f_slice : slice or None
       The frames to extract data for

    """
    if f_slice is None:
        f_slice = slice(None, None, None)
    print f_slice
    tmp_fft = np.fft.fft(RM.resampled_height[:, f_slice], axis=0)
    print tmp_fft.shape
    angles = np.angle(tmp_fft[k_list, :])

    return np.std(np.unwrap(angles, axis=1), axis=1)


def make_kymo_mode_panels(_rm, ax_lst, k_list, hbe,
                          fix_scale=True, cmap='gray', extent=None):
    assert len(k_list) == len(ax_lst)

    m = _rm.resampled_height.shape[0]
    s = np.linspace(0, 1, m)

    tmp_fft = np.fft.fft(_rm.resampled_height, axis=0) / m

    if fix_scale:
        h_range = np.abs(np.nanmax(_rm.height_map) - np.nanmin(_rm.height_map))
        vmin = -h_range / 2
        vmax = h_range / 2
    else:
        vmin = vmax = None
    for ax, k in zip(ax_lst, k_list):
        k_kymo = 2*(tmp_fft[k, :].real.reshape(-1, 1) * np.cos(s*k * 2*np.pi) -
                    tmp_fft[k, :].imag.reshape(-1, 1) * np.sin(s*k * 2*np.pi))
        ax.imshow(k_kymo.T, cmap=cmap,
                  interpolation='none',
                  aspect='auto',
                  vmin=vmin, vmax=vmax, extent=extent, origin='bottom')
    #        ax.set_aspect('auto', adjustable='box')
    #       ax.colorbar(im)


def make_kymo_mode_figure(_rm, hbe, k_list=None, **kwargs):
    """
    Makes figure with mode kymos
    """
    if k_list is None:
        k_list = [1, 2, 3, 4]
    fig = plt.figure(figsize=(3.375*2 + .25, 3.375*2 + .25))

    full_grid = Grid(fig, nrows_ncols=(len(k_list) + 2, 1),
                     rect=[.1, .1, .85, .85],
                     share_all=True, axes_pad=.1)

    _rm.display_height(ax=full_grid[0], t_scale=1/hbe.frame_rate)
    _rm.display_height_resampled(ax=full_grid[1], t_scale=1/hbe.frame_rate)
    make_kymo_mode_panels(full_grid[2:], k_list, _rm, hbe,
                          extent=[0, len(_rm) / hbe.frame_rate, 0, 2*np.pi])


class IS_FS_recon(object):
    """
    A class to wrap up dealing with the irregularly sampled periodic
    data
    """
    @classmethod
    def reconstruct_iterative(cls, phi, h, target_error, start_N=3, bound_N=15, iters=25):
        """
        reconstruct a band-limited a curve from unevenly
        sampled points

        Parameters
        ----------
        h : array
            The value of the curve

        phi : array
            The sample points.  Assumed to be in range [0, 2*np.pi)

        target_error : float
            The average difference between the reconstruction and the input data.  Stops
            iterating when hit

        start_N : int, optional
            The initial bandwidth.  Defaults to 3

        bound_N : int, optional
            The maximum bandwidth.  Defaults to 15

        iters : int, optional
            The maximum number of refinement steps at each N
        Returns:
        reconstrution: IS_FS_recon
           A callable object

        """
        # internal details for interpolation
        min_range = 0
        max_range = 2*np.pi
        _N = 1024
        intep_func = scipy.interpolate.interp1d
        new_phi = np.linspace(min_range, max_range, _N)

        # make sure everything is ndarrays
        phi = np.asarray(phi)
        h = np.asarray(h)

        # make sure phi is in range and sorted
        phi = np.mod(phi, 2*np.pi)
        indx = np.argsort(phi)
        phi = phi[indx]
        h = h[indx]
        # add bounds to make the interpolation happy
        pad = 1
        _phi = np.hstack((phi[-pad::-1] - max_range, phi, phi[:pad:-1] + max_range))
        _h = np.hstack((h[-pad::-1], h, h[:pad:-1]))

        # do first FFT pass
        f = intep_func(_phi, _h)
        new_h = f(new_phi)
        # / _N is to deal with the normalization scheme used by numpy
        tmp_fft = np.fft.fft(new_h) / _N

        max_N = start_N

        # set this up once to save computation
        sin_list = np.vstack([np.sin(k * phi)
                              for k in xrange(1, max_N+1)])
        cos_list = np.vstack([np.cos(k * phi)
                              for k in xrange(1, max_N+1)])

        not_done_flag = True
        # truncate fft values
        A_n = tmp_fft[:max_N+1]

        while not_done_flag and max_N < bound_N:

            # loop a bunch more times
            j = 0
            keep_going_flag = True
            prev_err = None
            while keep_going_flag and j < iters:
                # pull out constant
                A0 = A_n[0].real
                # reshape the rest of the list for broadcasting tricks
                A_list = A_n[1:].reshape(-1, 1)
                # compute reconstruction via broadcasting + sum
                re_con = 2*np.sum(A_list.real * cos_list -
                                  A_list.imag * sin_list, axis=0) + A0
                # compute error
                error_h = h - re_con
                mean_err = np.abs(np.mean(error_h))
                # check if we are done for good
                if mean_err < target_error:
                    not_done_flag = False
                    break
                if prev_err is not None:
                    if (prev_err - mean_err) < .01 * target_error:
                        keep_going_flag = False
                prev_err = mean_err

                # add padding to error so that the interpolation is happy
                _h = np.hstack((error_h[-pad::-1], error_h, error_h[:pad:-1]))
                # compute FFT of error
                f = intep_func(_phi, _h, kind='nearest')
                new_h = f(new_phi)
                tmp_fft = np.fft.fft(new_h) / _N
                # add correction term to A_n list
                A_n += tmp_fft[:max_N+1]
            else:
                tmp_an = np.zeros(max_N + 2, dtype='complex')
                tmp_an[:max_N + 1] = A_n
                max_N += 1
                A_n = tmp_an

                # set this up once to save computation
                sin_list = np.vstack([np.sin(k * phi)
                                      for k in xrange(1, max_N+1)])
                cos_list = np.vstack([np.cos(k * phi)
                                      for k in xrange(1, max_N+1)])

        # create callable object with the reconstructed coefficients
        return cls(A_n)

    @classmethod
    def reconstruct(cls, phi, h, max_N=10, iters=25):
        """
        reconstruct a band-limited a curve from unevenly
        sampled points

        Parameters
        ----------
        h : array
            The value of the curve

        phi : array
            The sample points.  Assumed to be in range [0, 2*np.pi)

        max_N : int, optional
            The number of modes to use

        iters : int, optional
            The number of iterations

        Returns:
        reconstrution: IS_FS_recon
           A callable object

        """
        # internal details for interpolation
        min_range = 0
        max_range = 2*np.pi
        _N = 1024
        intep_func = scipy.interpolate.interp1d
        new_phi = np.linspace(min_range, max_range, _N)

        # make sure everything is ndarrays
        phi = np.asarray(phi)
        h = np.asarray(h)

        # make sure phi is in range and sorted
        phi = np.mod(phi, max_range)
        indx = np.argsort(phi)
        phi = phi[indx]
        h = h[indx]

        # set this up once to save computation
        sin_list = np.vstack([np.sin(k * phi)
                              for k in xrange(1, max_N+1)])
        cos_list = np.vstack([np.cos(k * phi)
                              for k in xrange(1, max_N+1)])

        # add bounds to make the interpolation happy
        pad = 2
        _phi = np.hstack((phi[-pad::-1] - max_range,
                          phi,
                          phi[:pad:-1] + max_range))
        _h = np.hstack((h[-pad::-1], h, h[:pad:-1]))

        # do first FFT pass
        f = intep_func(_phi, _h)
        new_h = f(new_phi)
        # / _N is to deal with the normalization scheme used by numpy
        tmp_fft = np.fft.fft(new_h) / _N

        # truncate fft values
        A_n = tmp_fft[:max_N+1]
        prev_err = None
        # loop a bunch more times
        for j in range(iters):
            # pull out constant
            A0 = A_n[0].real
            # reshape the rest of the list for broadcasting tricks
            A_list = A_n[1:].reshape(-1, 1)
            # compute reconstruction via broadcasting + sum
            re_con = 2*np.sum(A_list.real * cos_list -
                              A_list.imag * sin_list, axis=0) + A0
            # compute error
            error_h = h - re_con
            curr_err = np.abs(np.mean(error_h))
            if prev_err is not None:
                # if the error isn't really improving, bail
                if 1 - curr_err / prev_err < .001:
                    break
            prev_err = curr_err
            # add padding to error so that the interpolation is happy
            _h = np.hstack((error_h[-pad::-1], error_h, error_h[:pad:-1]))
            # compute FFT of error
            f = intep_func(_phi, _h, kind='nearest')
            new_h = f(new_phi)
            tmp_fft = np.fft.fft(new_h) / _N
            # add correction term to A_n list
            A_n += tmp_fft[:max_N+1]

        # create callable object with the reconstructed coefficients
        return cls(A_n)

    def __init__(self, A_n):
        self.A_n = np.asarray(A_n)
        self.N = len(A_n)

    @property
    def max_N(self):
        return len(self.A_n) - 1

    def __call__(self, th, deriv=0):
        """
        Returns the re-constructed curve at the points
        `th`.

        The order of the derivative is specified by deriv.
        """
        th = np.asarray(th)
        sin_list = np.vstack([k**deriv * np.sin(k * th)
                              for k in xrange(1, self.N)])
        cos_list = np.vstack([k**deriv * np.cos(k * th)
                              for k in xrange(1, self.N)])

        A0 = self.A_n[0].real if deriv == 0 else 0
        A_list = self.A_n[1:].reshape(-1, 1)

        # there has to be a better way to write this
        deriv = deriv % 4
        if deriv == 0:
            a, b = cos_list, sin_list
        elif deriv == 1:
            a, b = -sin_list, cos_list
        elif deriv == 2:
            a, b = -cos_list, -sin_list
        elif deriv == 3:
            a, b = sin_list, -cos_list

        return 2*np.sum(A_list.real * a -
                        A_list.imag * b, axis=0) + A0
