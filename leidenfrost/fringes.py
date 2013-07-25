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
from itertools import tee, izip, cycle
from collections import namedtuple, defaultdict, deque

from contextlib import closing

from matplotlib import cm
import matplotlib
import fractions
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import numpy as np
import h5py
from bisect import bisect
from scipy.interpolate import griddata


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
        key = r'{{\color{{red}}{0}}}&{{\color{{DodgerBlue2}}{1} }}: &{2}'.format(j, ft_fmt, format_fringe(ft))
        key_lst.append(key)

    return ('\\begin{eqnarray*}\n' + '\\\\\n'.join(key_lst) + '\n\end{eqnarray*}',
            '\\begin{eqnarray*}\n' + '\\\\\n'.join(res_lst) + '\n\end{eqnarray*}')


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

        self.region = np.nan     #: region of the khymograph
        self.abs_height = None   #: the height of this fringe as given by tracking in time

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
                # need the negative, as we want the step from this one _to_ that one
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
        self.fringes = [Fringe(fcls, floc, frame_number) for fcls, floc in izip(f_classes, f_locs)]
        self.fringes.sort(key=lambda x: x.phi)

    def link_fringes(self, region_starts, region_labels, region_ends,
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

        N_samples : int
            The number of samples

        length : float
            The length of the region sampled in the 'natural' units

        '''
        # figure out which bin each fringe goes into just once
        bins = [_bin_region(int((np.mod(_fr.phi, length)/(length)) * N_samples),
                                    region_starts,
                                    region_ends) for _fr in self.fringes]

        for (fr_b, b_b), (fr_f, b_f) in pairwise_periodic(izip(self.fringes, bins)):

            fr_b.abs_height = np.nan

            # handle the other mapping
            if b_b is None:
                label = 0
            else:
                label = region_labels[b_b]
            fr_b.region = label
            fr_b.abs_height = np.nan

            # handle the linking
            if b_b is None or b_f is None:
                continue
            if (b_b + 1 == b_f) or (b_f == 0 and
                                    b_b == len(region_labels) - 1):
                fr_b.insert_ahead(fr_f)

    def __iter__(self):
        return self.fringes.__iter__()

    def __len__(self):
        return len(self.fringes)

    def __getitem__(self, key):
        return self.fringes[key]

    def __eq__(self, other):
        if self.frame_number != other.frame_number:
            return False
        return all(_f1 == _f2 for _f1, _f2 in izip(self.fringes, other.fringes))


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

    # convert the curve to X,Y
    XY = np.vstack(mbe.curve.q_phi_to_xy(0,
                                         np.linspace(0, 2 * np.pi, 2 ** 10)))
    # get center
    center = np.mean(XY, 1)
    # get relative position of first point
    first_pt = center - XY[:, 0]
    # get the off set angle of the first
    th_offset = np.arctan2(first_pt[1], first_pt[0])
    if reclassify:
        for color, trk_lst in izip(colors, mbe.trk_lst):
            for t in trk_lst:
                t.classify2()
                if t.charge is not None:
                    f_locs.append(fringe_loc(t.q, np.mod(t.phi + th_offset, 2 * np.pi)))
                    f_classes.append(fringe_cls(color, t.charge, 0))
                else:
                    junk_fringes.append(t)
    else:
        for res_lst, color in izip(mbe.res, colors):
            for charge, phi, q in izip(*res_lst):
                f_locs.append(fringe_loc(q, np.mod(phi + th_offset, 2 * np.pi)))
                f_classes.append(fringe_cls(color, charge, 0))

    f_classes, f_locs = zip(*sorted(zip(f_classes, f_locs), key=lambda x: x[1][1]))
    # TODO: deal with junk fringes in sensible way

    return list(f_classes), list(f_locs)


class Region_map(object):
    def __eq__(self, other):
        # test numpy like things
        try:
            np.testing.assert_equal(self.label_img, other.label_img)
            np.testing.assert_equal(self.height_map, other.height_map)
            np.testing.assert_equal(self.working_img, other.working_img)
        except AssertionError:
            return False
        # test fringe rings
        if not all(_fr1 == _fr2 for _fr1, _fr2 in izip(self.fringe_rings, other.fringe_rings)):
            return False
        # test region edges
        if not all(_r1 == _r2 for _r1, _r2 in izip(self.region_edges, other.region_edges)):
            return False
        # don't bother to test the parameters
        return True

    @staticmethod
    def _label_regions(mask, size_cut, structure):
        '''
        Labels the regions

        Parameters
        ----------
        mask : binary ndarray
            The array to identify regions in

        size_cut : int
            Maximum number of pixels to be in a
        '''
        if structure is not None:
            mask = ndi.binary_erosion(mask, structure=structure, border_value=1)
        #    mask = ndi.binary_propagation(mask)

        lab_regions, nb = ndi.label(mask)
        # loop to make periodic
        for j in range(lab_regions.shape[1]):
            top_lab = lab_regions[0, j]
            bot_lab = lab_regions[-1, j]
            if top_lab != bot_lab and top_lab != 0 and bot_lab != 0:
                lab_regions[lab_regions == bot_lab] = top_lab

        sizes = ndi.sum(mask, lab_regions, range(nb + 1))
        mask_sizes = sizes < size_cut
        print len(sizes), sum(mask_sizes)

        remove_pix = mask_sizes[lab_regions]
        lab_regions[remove_pix] = 0
        labels = np.unique(lab_regions)
        lab_regions = np.searchsorted(labels, lab_regions)
        return lab_regions, len(labels)

    @classmethod
    def from_backend(cls, backend, n_frames=None, reclassify=False, **kwargs):
        if n_frames is None:
            n_frames = len(backend)
        img_bck_grnd_slices = []
        fringe_rings = []
        for j in range(n_frames):
            if j % 1000 == 0:
                print j
            mbe = backend.get_frame(j, img=True, raw=reclassify)

            fringe_rings.append(FringeRing.from_mbe(mbe, reclassify=reclassify))
            if reclassify:
                curve = mbe.get_next_spline()
            else:
                curve = mbe.curve
            img = mbe.img

            # convert the curve to X,Y
            XY = np.vstack(curve.q_phi_to_xy(0,
                                             np.linspace(0, 2 * np.pi, 2 ** 10)))
            # get center
            center = np.mean(XY, 1)
            # get relative position of first point
            first_pt = center - XY[:, 0]
            # get the off set angle of the first
            th_offset = np.arctan2(first_pt[1], first_pt[0])

            XY = np.vstack(curve.q_phi_to_xy(0,
                                             -th_offset + np.linspace(0, 2 * np.pi, 2 ** 12)))
            img_bck_grnd_slices.append(map_coordinates(img, XY[::-1], order=2))

        working_img = np.vstack(img_bck_grnd_slices).T

        return cls._from_raw_data(working_img, FRs=fringe_rings, **kwargs)

    @classmethod
    def _connection_network(cls, N, fringe_rings, dirc='f'):
        """
        Sets up the network between the regions based on what fringes fall in
        them.


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

    @classmethod
    def _from_raw_data(cls, working_img, FRs, thresh=0,
                 size_cut=100, structure=None, **kwargs):

        up_mask = working_img > 1 + thresh
        down_mask = working_img < 1 - thresh

        lab_bright_regions, nb_br = cls._label_regions(up_mask,
                                                        size_cut,
                                                        structure)
        lab_dark_regions, nb_dr = cls._label_regions(down_mask,
                                                      size_cut,
                                                      structure)
        lab_dark_regions[lab_dark_regions > 0] += nb_br

        fringe_rings = FRs
        label_regions = np.asarray(lab_dark_regions + lab_bright_regions,
                                        dtype=np.uint32)
        N = nb_br + nb_dr

        region_edges = [_segment_labels(region_list)
                             for region_list in label_regions.T]

        for FR, (region_starts,
                 region_labels,
                 region_ends) in izip(fringe_rings, region_edges):
            FR.link_fringes(region_starts, region_labels, region_ends,
                            working_img.shape[0])

        # boot strap up the heights
        height_map, set_by, fails = cls._boot_strap(N, fringe_rings)
        return cls(fringe_rings, region_edges, working_img, height_map,
                   thresh=thresh, size_cut=size_cut, structure=structure,
                   **kwargs)

    @classmethod
    def _boot_strap(cls, N, FRs):
        """An improved boot-strap operation
        """
        valid_connections = deque()
        for dd in izip(cls._connection_network(N, FRs, 'f'),
                       cls._connection_network(N, FRs, 'r')):
            tmp_dict = {}
            for _dd in dd:
                for k, v in _dd.items():
                    dh = _dict_to_dh(v, threshold=5)
                    if dh is not None:
                        if k in tmp_dict and tmp_dict[k] != dh:
                            print 'conflict'
                            # if we have inconsistent linking, throw
                            # everything out
                            del tmp_dict[k]
                            continue
                        tmp_dict[k] = dh
            valid_connections.append(tmp_dict)

        valid_connections = list(valid_connections)

        # pick the one with the most forward connections
        start = np.argmax([len(r) for r in valid_connections])
        # clear height map
        height_map = np.ones(N, dtype=np.float32) * np.nan
        #set first height
        height_map[start] = 0

        set_by = dict()
        fails = deque()
        work_list = deque()
        work_list.extend([(start, k) for k in valid_connections[start].keys()])
        while len(work_list) > 0:
            a, b = work_list.pop()
            prop_height = height_map[a] + valid_connections[a][b]
            if np.isnan(height_map[b]):
                height_map[b] = prop_height
                work_list.extend([(b, k) for k in valid_connections[b].keys()])
                set_by[b] = a
            else:
                if height_map[b] != prop_height:
                    print a, b, prop_height, \
                          height_map[b], valid_connections[a][b]
                    fails.append((a, b))

        return height_map, set_by, fails

    def __init__(self, fringe_rings, region_edges, working_img, height_map,
                 **kwargs):
        self.fringe_rings = fringe_rings      # fringes group by a per-time basis
        self.region_edges = region_edges      # edges of the regions on a per-time basis
        self.working_img = working_img        # the raw image not sure why we are carrying this around)
        self.height_map = height_map          # the mapping between regions and heights

        self._height_img = None           # image of the heights
        self._label_img = None            # image of the labeled regions
        self.params = kwargs              # dict to hold parameters
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

        ax.imshow(height_img,
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
        tmp = np.zeros(self.working_img.shape) * np.nan
        for j, edges in enumerate(self.region_edges):
            for (r_start, r_label, r_stop) in zip(*edges):
                tmp[r_start:r_stop, j] = r_label

        return tmp

    def resample_height_img(self, th_step=1000, tau_step=5000, method='cubic'):
        assert method in ['linear', 'cubic', 'nearest']

        scale = 2 * np.pi / self.working_img.shape[0]
        tmp_pts = []

        print 'started'
        for j, (region_start,
                region_label,
                region_ends) in enumerate(self.region_edges):
            tmp_pts.extend(((j, scale * (re + rs) / 2), self.height_map[rl])
                           for rs, re, rl in izip(region_start,
                                                  region_ends, region_label)
                           if not np.isnan(self.height_map[rl]))

        print 'mapped'
        print len(tmp_pts)
        points, vals = zip(*tmp_pts)
        points = np.vstack(points)
        grid_y, grid_x = np.mgrid[0:2 * np.pi:th_step*1j,
                                  0:self.working_img.shape[1]:tau_step*1j]

        print 'gridding'
        grid_z2 = griddata(points, vals, (grid_x, grid_y), method=method)
        print 'gridded'
        return grid_z2

    def display_height_resampled(self, ax=None, cmap='jet',
                                 bckgnd=True, alpha=.65,
                                 t_scale=1, t_units=''):
        height_img = self.resample_height_img()
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

        ax.figure.canvas.draw()

    def display_region(self, n, ax=None):

        if ax is None:
            # make this smarter
            ax = plt.gca()

        data = self.label_img

        norm_br = matplotlib.colors.Normalize(vmin=.5,
                                              vmax=np.max(data), clip=False)
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

    def write_to_hdf(self, out_file, md_dict=None, mode='w-'):

        # this will blow up if the file exists
        with closing(h5py.File(out_file.format, mode)) as h5file:

            h5file.attrs['ver'] = '0.2'
            # store all the md passed in

            if md_dict is not None:
                for key, val in md_dict.iteritems():
                    if val is None:
                        continue
                    try:
                        h5file.attrs[key] = val
                    except TypeError:
                        print 'key: ' + key + ' can not be gracefully shoved into'
                        print ' an hdf object, please reconsider your life choices'

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
                re_ds = re_grp.create_dataset("region_edges_{frn:07}".format(frn=frame_number),
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
        # pull out the edges of the regions
        re_grp = h5file['region_edges']
        print 'starting region edges'
        # this relies on the iterator in h5py returning things sorted
        # and the padding being sufficient to always sort correctly
        region_edges = [Region_Edges(*re_grp[k][:]) for k in re_grp]
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
        print 'starting linking'
        # link the fringes.  This is simpler than storing the linking information
        for FR, (region_starts,
                 region_labels,
                 region_ends) in izip(fringe_rings, region_edges):
            FR.link_fringes(region_starts, region_labels, region_ends,
                            working_img.shape[0])

        params = dict(h5file['params'].attrs)

        return cls(fringe_rings, region_edges, working_img, height_map, **params)

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
                r_start, r_labels, r_end = self.region_edges[row]
                region = r_labels[_bin_region(col, r_start, r_end)]
                return "x:{x}, y:{y}, r:{reg}, h:{h}".format(x=col * xscale,
                                                             y=row * yscale,
                                                             reg=region,
                                                    h=self.height_map[region])
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        return format_coord


def _segment_labels(region_list, zero_thresh=2):
    '''
    Segments the regions.  Returns list of where contiguous regions begin and end

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
        region_ends.append(len(region_list) - 1)

    return Region_Edges(region_starts, region_labels, region_ends)


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
    input_d: dict
        `input_d` contains the keys [-1, 0, 1] with values that are counts
    threshold: int
        the minimum number of counts to be valid

    Returns
    -------
    dh: int, (-1, 0, 1) or `None`
    """
    # check of there is more than one entry with non-zero counts
    if input_d[0] and input_d[-1] and input_d[1]:
        return None

    # if any
    for k in [-1, 0, 1]:
        if input_d[k] > threshold:
            return k
    return None
