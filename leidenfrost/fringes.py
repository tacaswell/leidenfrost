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

fringe_cls = namedtuple('fringe_cls', ['color', 'charge', 'hint'])
fringe_loc = namedtuple('fringe_loc', ['q', 'phi'])

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
            if self.valid_follower(other):
                self._fdh = forward_dh_dict[(self.f_class, other.f_class)]
            else:
                self._fdh = np.nan

        return self._fdh

    @property
    def reverse_dh(self):
        if self._rdh is None:
            other = self.prev_P
            if other.valid_follower(self):
                self._rdh = forward_dh_dict[(other.f_class, self.f_class)]
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
        for a, b in pairwise_periodic(self.fringes):
            a.insert_ahead(b)

    def __iter__(self):
        return self.fringes.__iter__()

    def __len__(self):
        return len(self.fringes)

    def __getitem__(self, key):
        return self.fringes[key]


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
    @staticmethod
    def _label_regions(mask, size_cut, structure):

        if structure is not None:
            mask = ndi.binary_erosion(mask, structure=structure, border_value=1)
        #    mask = ndi.binary_propagation(mask)

        lab_regions, nb = ndi.label(mask)
        sizes = ndi.sum(mask, lab_regions, range(nb + 1))
        print len(sizes)
        mask_sizes = sizes < size_cut
        print sum(mask_sizes)
        remove_pix = mask_sizes[lab_regions]
        lab_regions[remove_pix] = 0
        # loop to make periodic
        for j in range(lab_regions.shape[1]):
            top_lab = lab_regions[0, j]
            bot_lab = lab_regions[-1, j]
            if top_lab != bot_lab and top_lab != 0 and bot_lab != 0:
                lab_regions[lab_regions == bot_lab] = top_lab

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

        return cls(working_img, FRs=fringe_rings, **kwargs)

    @classmethod
    def from_RM(cls, RM):
        return cls(RM.working_img, RM.fring_rings, RM.thresh, RM.size_cut, RM.structure)

    def __init__(self, working_img, FRs, thresh=0, size_cut=100, structure=None):
        up_mask = working_img > 1 + thresh
        down_mask = working_img < 1 - thresh

        lab_bright_regions, nb_br = self._label_regions(up_mask, size_cut, structure)
        lab_dark_regions, nb_dr = self._label_regions(down_mask, size_cut, structure)
        lab_dark_regions[lab_dark_regions > 0] += nb_br

        self.fring_rings = FRs
        self.label_regions = np.asarray(lab_dark_regions + lab_bright_regions, dtype=np.uint32)
        self.height_map = np.ones(nb_br + nb_dr, dtype=np.float32) * np.nan
        self.region_fringes = [[] for j in range(nb_br + nb_dr)]
        self.working_img = working_img

        self.thresh = thresh
        self.structure = structure
        self.size_cut = size_cut

        for FR in self.fring_rings:
            for fr in FR:
                theta_indx = int((np.mod(fr.phi, 2 * np.pi) / (2 * np.pi)) * self.label_regions.shape[0])
                label = self.label_regions[theta_indx, FR.frame_number]
                fr.region = label
                fr.abs_height = np.nan
                self.region_fringes[label].append(fr)

    def display_height(self, ax=None, cmap='jet', bckgnd=True, alpha=.65, t_scale=1, t_units=''):
        height_img = np.ones(self.label_regions.shape, dtype=np.float32) * np.nan
        for j in range(1, len(self.height_map)):
                height_img[self.label_regions == j] = self.height_map[j]

        if ax is None:
            # make this smarter
            ax = plt.gca()

        my_cmap = cm.get_cmap(cmap)
        my_cmap.set_bad(alpha=0)

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
                  extent=[0, (height_img.shape[1] - 1) * t_scale, 0, 2 * np.pi],
                  aspect='auto',
                  origin='bottom',
                  )
        if bckgnd:
            ax.imshow(self.working_img,
                      interpolation='none',
                      cmap='gray',
                      extent=[0, (height_img.shape[1] - 1) * t_scale, 0, 2 * np.pi],
                      aspect='auto',
                      origin='bottom',
                      alpha=alpha)

        ax.figure.canvas.draw()

    def display_region(self, n, ax=None):

        if ax is None:
            # make this smarter
            ax = plt.gca()

        data = self.label_regions

        norm_br = matplotlib.colors.Normalize(vmin=.5, vmax=np.max(data), clip=False)
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

        data = self.label_regions

        norm_br = matplotlib.colors.Normalize(vmin=.5, vmax=np.max(data), clip=False)
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
        theta_indx = int((np.mod(theta, 2 * np.pi) / (2 * np.pi)) * self.label_regions.shape[0])
        label = self.label_regions[theta_indx, frame_num]
        return self.height_map[label]

    def _set_height(self, fr, height, overwrite=False):

        label = fr.region
        assert not np.isnan(label), 'label should not be nan'
        # we can't do anything with points in the un-labeled regions of the image
        if label == 0:
            return
        if not overwrite and not np.isnan(self.height_map[label]):
            # we won't over-write at this region already has a height
            # TODO make this a custom exception
            raise RuntimeError("already set the height of this region!")

        self.height_map[label] = height
        for _fr in self.region_fringes[label]:
            _fr.abs_height = height

    def _boot_strap_frame(self, j):

        FR = self.fring_rings[j]
        for fr in FR.fringes:
            fr.abs_height = self.get_height(j, fr.phi)

        invalid_fringes = [fr for fr in FR.fringes if np.isnan(fr.abs_height) and fr.region > 0]

        try_again_flag = len(invalid_fringes) > 0
        while try_again_flag:
            try_again_flag = False
            invalid_fringes = [fr for fr in invalid_fringes if np.isnan(fr.abs_height) and fr.region > 0]
            for fr in invalid_fringes:
                if np.isnan(fr.abs_height):
                    # we need to try to figure out the height
                    prev = fr.prev_P
                    nexp = fr.next_P
                    if prev.valid_follower(fr):
                        # if previous to current is a valid combination
                        p_h = prev.abs_height
                        if not np.isnan(p_h):
                            # only set the height if the fringes on either side agree
                            dh_prev = prev.forward_dh
                            self._set_height(fr, p_h + dh_prev)
                            try_again_flag = True

                    elif fr.valid_follower(nexp):
                        n_h = nexp.abs_height
                        if not np.isnan(n_h):
                            dh_next = fr.forward_dh
                            self._set_height(fr, n_h - dh_next)
                            try_again_flag = True

    def boot_strap(self):
        re_boot = True
        while re_boot:
            re_boot = False
            pre_un_lab_count = np.sum(np.isnan(self.height_map))
            for j in range(self.label_regions.shape[1]):
                if j % 1000 == 0:
                    print j
                self._boot_strap_frame(j)
            post_unlab_count = np.sum(np.isnan(self.height_map))
            if pre_un_lab_count != post_unlab_count:
                re_boot = True

    def seed_frame0(self):
        FR = self.fring_rings[0]
        first_frame_dh, ff_phi = [np.array(_) for _ in zip(*[(fr.forward_dh, fr.phi) for fr in FR])]
        invalid_steps, = np.where(np.isnan(first_frame_dh))

        if len(invalid_steps) > 2:
            # this needs more work to deal with periodic runs properly
            best_run_start = np.argmax(np.diff(invalid_steps))
            run_rng = range(invalid_steps[best_run_start] + 1, invalid_steps[best_run_start + 1])
        elif len(invalid_steps) == 2:
            if invalid_steps[1] - invalid_steps[0] > invalid_steps[0] + len(FR) - invalid_steps[1]:
                run_rng = range(invalid_steps[0] + 1, invalid_steps[1])
            else:
                run_rng = range(invalid_steps[1] + 1, len(FR)) + range(0, invalid_steps[0])
        elif len(invalid_steps) == 1:
            run_rng = range(invalid_steps[0] + 1, len(FR)) + range(0, invalid_steps[0])
        elif len(invalid_steps) == 0:
            run_rng = range(0, len(FR))

        h = 0
        for j in run_rng:
            print 'h: ', h
            try:
                self._set_height(FR[j], h)
            except RuntimeError:
                eh = self.get_height(0, ff_phi[j])
                print eh, h
                h = eh

            h += first_frame_dh[j]

    def write_to_hdf(self, out_file, md_dict):

        # this will blow up if the file exists
        with closing(h5py.File(out_file.format, 'w-')) as h5file:

            h5file.attrs['ver'] = '0.1'
            # store all the md passed in
            for key, val in md_dict:
                try:
                    h5file.attrs[key] = val
                except TypeError:
                    print 'key: ' + key + ' can not be gracefully shoved into'
                    print ' an hdf object, please reconsider your life choices'
            h5file.attrs['thresh'] = self.thresh
            h5file.attrs['size_cut'] = self.size_cut
            if self.structure is not None:
                h5file.attrs['structure'] = np.asarray(self.structure)

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
            # the regions of space
            h5file.create_dataset('label_regions',
                                  self.label_regions.shape,
                                  self.label_regions.dtype,
                                  compression='szip')
            h5file['label_regions'][:] = self.label_regions

            fr_grp = h5file.create_group('fringe_rings')
            for FR in self.fring_rings:
                f_cls = np.vstack([fr.f_cls for fr in FR])
                f_loc = np.vstack([fr.f_cls for fr in FR])
                frame_number = fr.frame_number
                dset_names = ["f_class_{frn:02}".format(frn=frame_number),
                              "f_loc_{frn:02}".format(frn=frame_number)]
                for n, v, t in izip(dset_names, [f_cls, f_loc], [np.int8, np.float]):
                    fr_grp.create_dataset(n,
                                          v.shape,
                                          t)

    @classmethod
    def from_hdf(cls, in_file):
        pass

    def error_check(self):
        error_count = 0
        correct_count = 0
        error_regions = defaultdict(int)
        for FR in self.fring_rings:
            for fr in FR:
                fdh = fr.forward_dh

                if (not np.isnan(fdh)) and (not np.isnan(fr.abs_height)) and (not np.isnan(fr.next_P.abs_height)):
                    if int(fr.next_P.abs_height - fr.abs_height) != fdh:

                        error_count += 1
                        ra, rb = fr.region, fr.next_P.region

                        error_regions[(ra, rb)] += 1

                    else:
                        correct_count += 1

        return error_count, error_regions

    def format_factory(self, xscale=1, yscale=1):
        numrows, numcols = self.label_regions.shape

        def format_coord(x, y):
            col = int(x / xscale)
            row = int(y / yscale)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                region = self.label_regions[row, col]
                return "x:{x}, y:{y}, r:{reg}, h:{h}".format(x=col * xscale,
                                                             y=row * yscale,
                                                             reg=region,
                                                             h=self.height_map[region])
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        return format_coord
