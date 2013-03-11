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
from itertools import tee, izip, cycle, product, islice
from collections import namedtuple, defaultdict, deque
import os.path

from matplotlib import cm
import fractions
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import numpy as np

from infra import Point1D_circ

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


# this should be replaced with a look-up table
def get_valid_between(a, b):
    '''Returns a tuple of the valid fringe types between fringes a and b '''
    v_follow = set(valid_follow_dict[a])
    v_prec = set(valid_precede_dict[b])
    return tuple(v_follow & v_prec)


### atomic validity tests
def is_valid_3(a, b, c):
    '''
    returns if this is a valid sequence of fringes
    '''
    return a in valid_precede_dict[b] and c in valid_follow_dict[b]


def is_valid_run(args):
    '''Returns if this *non-periodic* run is a valid'''
    return all([p in valid_precede_dict[f] for p, f in pairwise(args)])


def is_valid_run_periodic(args):
    '''Returns if this *periodic* run is a valid'''
    return all([p in valid_precede_dict[f] for p, f in pairwise_periodic(args)])


### iterator toys

# ganked from docs
def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C

    copied from documentation
    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


#ganked from docs
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ...

    copied from documentation
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def pairwise_periodic(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..., (sn,s0)

    modified from example in documentation
    """
    a, b = tee(iterable)
    b = cycle(b)
    next(b, None)
    return izip(a, b)


def triple_wise_periodic(iterable):
    """s -> (s0,s1,s2), (s1,s2,s3), ..., (sn,s0,s1)

    modified from example in documentation
    """
    a, _b = tee(iterable)
    b, c = tee(cycle(_b))
    next(b, None)
    next(c, None)
    next(c, None)

    return izip(a, b, c)


def triple_wise(iterable):
    """s -> (s0,s1,s2), (s1,s2,s3),

    modified from example in documentation
    """
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)

    return izip(a, b, c)

##### find problematic runs

def find_bad_runs(d):
    '''
    Takes in a list of bools, returns a list of tuples that are continuous False runs.

    Does not know if the list in periodic, so a run that is split across the beginging/end will
    be 2 slices -> rotate so that first and last are valid before
    '''
    d_len = len(d)

    j = 0
    res = list()
    while j < d_len:
        if d[j]:
            #
            j += 1
        else:
            k_start = j
            while j < d_len and not d[j]:
                j += 1
            k_end = j
            j += 1
            res.append((k_start, k_end))

    return res

def _hinting_only(run, locs):
    '''Only try to guess hinting, return empty lists if this will require  '''
    j = 0

    zero_runs = []
    while j < len(run):

        if run[j].charge != 0:
            j += 1
            continue
        elif run[j].charge == 0:
            run_start = j
            k = j + 1
            while k < len(run) and run[k].charge == 0:
                k += 1
            run_end = k
            j = k + 1
            zero_runs.append((run_start, run_end))

    valid_runs = []
    valid_locs = []
    # the product of all the possible combinations of all of the zero runs (this should be 2 ** N),
    # we have to do this in this funny broken up way because we could have 1 0 0 1 0 1 which will show up
    # as a single long run of invalid fringes

    # do the loop with _only_ hinting correction
    for trial_hints in product(*[product([-1, 1], repeat=z_r[1] - z_r[0]) for z_r in zero_runs]):
        # make local copy of the list
        trial_run = list(run)
        trial_locs = list(locs)
        # apply all of the hints
        for hint, z_r in izip(trial_hints, zero_runs):
            cur_slice = slice(*z_r)
            trial_run[cur_slice] = [_f._replace(hint=th) for _f, th in izip(trial_run[cur_slice], hint)]
        d = list(is_valid_3(*p) for p in triple_wise(trial_run))

        if all(d):
            # just setting the hints was enough and the configuration is happy
            valid_runs.append(trial_run)
            valid_locs.append(trial_locs)
            continue


    return valid_runs, valid_locs


def _valid_run_fixes_with_hinting(run, locs):
    '''

    This takes a run of un-hinted invalid fringes and return valid runs that include both hinting and added fringes

    '''
    j = 0

    zero_runs = []
    while j < len(run):

        if run[j].charge != 0:
            j += 1
            continue
        elif run[j].charge == 0:
            run_start = j
            k = j + 1
            while k < len(run) and run[k].charge == 0:
                k += 1
            run_end = k
            j = k + 1
            zero_runs.append((run_start, run_end))

    valid_runs = []
    valid_locs = []
    # the product of all the possible combinations of all of the zero runs (this should be 2 ** N),
    # we have to do this in this funny broken up way because we could have 1 0 0 1 0 1 which will show up
    # as a single long run of invalid fringes

    # do the loop with _only_ hinting correction
    for trial_hints in product(*[product([-1, 1], repeat=z_r[1] - z_r[0]) for z_r in zero_runs]):
        # make local copy of the list
        trial_run = list(run)
        trial_locs = list(locs)
        # apply all of the hints
        for hint, z_r in izip(trial_hints, zero_runs):
            cur_slice = slice(*z_r)
            trial_run[cur_slice] = [_f._replace(hint=th) for _f, th in izip(trial_run[cur_slice], hint)]
        d = list(is_valid_3(*p) for p in triple_wise(trial_run))

        if all(d):
            # just setting the hints was enough and the configuration is happy
            valid_runs.append(trial_run)
            valid_locs.append(trial_locs)
            continue

    # if we found hinting-only valid runs, then return
    if len(valid_runs) > 0:
        return valid_runs, valid_locs

    # we made it this far, so we need to do hinting + corrections
    for trial_hints in product(*[product([-1, 1], repeat=z_r[1] - z_r[0]) for z_r in zero_runs]):
        # make local copy of the list
        trial_run = list(run)
        trial_locs = list(locs)
        # apply all of the hints
        for hint, z_r in izip(trial_hints, zero_runs):
            cur_slice = slice(*z_r)
            trial_run[cur_slice] = [_f._replace(hint=th) for _f, th in izip(trial_run[cur_slice], hint)]
        d = list(is_valid_3(*p) for p in triple_wise(trial_run))

        # add the edges back (that got cut-off by the triple iterator
        d = ([trial_run[1] in valid_follow_dict[trial_run[0]]] +
             d +
             [trial_run[-2] in valid_precede_dict[trial_run[-1]]])
        res = find_bad_runs(d)
        slices = [slice(*_r) for _r in res]

        valid_sub_regions = [_valid_run_fixes(trial_run[s_], trial_locs[s_]) for s_ in slices]
        v_sub_runs, v_locs = zip(*valid_sub_regions)
        for vsr_prod in product(*v_sub_runs):
            for vsr, s_, vsl in izip(vsr_prod[::-1], slices[::-1], v_locs[::-1]):
                # do this loop backwards so slices stay valid
                l_trial_runs = list(trial_run)
                l_trial_locs = list(trial_locs)

                l_trial_runs[s_] = vsr
                l_trial_locs[s_] = vsl

            if is_valid_run(l_trial_runs):
                valid_runs.append(l_trial_runs)
                valid_locs.append(l_trial_locs)

    if len(valid_locs) > 0:
        return valid_runs, valid_locs

    # so, we have made it this far and not found a valid
    # configuration, hueristically do something stupid related to the
    # fact that some of our 0 charge fringes are miss-identified (really should fix this in
    # the location/classify step, but alea iacta est.

    striped_run, striped_locs = zip(*[(_r, _l) for _r, _l in zip(run, locs) if _r.charge != 0])
    d = list(is_valid_3(*p) for p in triple_wise(striped_run))

    # add the edges back (that got cut-off by the triple iterator
    d = ([trial_run[1] in valid_follow_dict[trial_run[0]]] +
         d +
         [trial_run[-2] in valid_precede_dict[trial_run[-1]]])
    res = find_bad_runs(d)
    slices = [slice(*_r) for _r in res]

    valid_sub_regions = [_valid_run_fixes(striped_run[s_], striped_locs[s_]) for s_ in slices]
    v_sub_runs, v_locs = zip(*valid_sub_regions)
    for vsr_prod in product(*v_sub_runs):
        for vsr, s_, vsl in izip(vsr_prod[::-1], slices[::-1], v_locs[::-1]):
            # do this loop backwards so slices stay valid
            l_trial_runs = list(striped_run)
            l_trial_locs = list(striped_locs)

            l_trial_runs[s_] = vsr
            l_trial_locs[s_] = vsl

        if is_valid_run(l_trial_runs):
            valid_runs.append(l_trial_runs)
            valid_locs.append(l_trial_locs)

    return valid_runs, valid_locs
    #


def _valid_run_fixes(run, locs):
    '''
    This takes a run of fringes with no hinting issues and returns possible
    '''

    possible_insertions = []
    possible_locs = []
    for (a, b), (a_loc, b_loc) in izip(pairwise(run), pairwise(locs)):
        # make this a look up!
        pi = set(valid_follow_dict[a]) & set(valid_precede_dict[b])
        possible_insertions.append(pi)
        phi_dist = (a_loc[1] - b_loc[1])
        if phi_dist > np.pi:
            phi_dist = 2 * np.pi - phi_dist
        phi_dist /= 2

        possible_locs.append(fringe_loc(0, b_loc[1] + phi_dist))

    valid_runs = []

    for pi in product(*possible_insertions):
        pos_run = list(roundrobin(run, pi))

        if is_valid_run(pos_run):
            valid_runs.append(pos_run)

    new_locs = list(roundrobin(locs, possible_locs))

    return valid_runs, new_locs


class Fringe(Point1D_circ):
    '''
    Version of :py:class:`Point1D_circ` for representing fringes

    '''

    def __init__(self, f_class, f_loc, frame_number):
        Point1D_circ.__init__(self, q=f_loc.q, phi=f_loc.phi)                  # initialize first base class

        self.f_class = f_class            #: fringe class

        # linked list for time
        self.next_T = None   #: next fringe in time
        self.prev_T = None   #: prev fringe in time
        # linked list for space
        self.next_P = None   #: next fringe in angle
        self.prev_P = None   #: prev fringe in angle

        # properties of fringe shape
        # self.q and self, phi are set by Point1D_circ __init__

        self.f_dh = None     #: dh figured going forward
        self.r_dh = None     #: dh figured going backwards
        self.f_cumh = None   #: the cumulative shift counting forward
        self.r_cumh = None   #: the cumulative shift counting forward



        self.slope_f = None  #: the slope going forward
        self.slope_r = None  #: the slope going backward
        self.slope = None    #: the 'average' slope at this point

        self.frame_number = frame_number    #: the frame that this fringe belongs to

        self.region = np.nan #: region of the khymograph
        self.abs_height = None   #: the height of this fringe as given by tracking in time

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

        self.remove_from_track(self.track)

    def valid_follower(self, other):
        return other.f_class in valid_follow_dict[self.f_class]

    def forward_dh(self):
        other = self.next_P
        if self.valid_follower(other):
            return forward_dh_dict[(self.f_class, other.f_class)]
        return np.nan

    def reverse_dh(self):
        other = self.prev_P
        if other.valid_follower(self):
            return forward_dh_dict[(other.f_class, self.f_class)]
        return np.nan


t_format_dicts = [{1:'B', -1:'D'},
                  {1:'L', 0:'_', -1:'R'},
                  {1:'P', 0:'_', -1:'S'}]


def format_fringe_txt(f):
    return ''.join([_d[_f] for _d, _f in izip(t_format_dicts, f)])


class FringeRing(object):
    '''
    A class to carry around Fringe data
    '''
    def __init__(self, mbe, reclassify=False):
        '''Extracts the data from the mbe, cleans up the fringes, and constructs the `Fringe` objects needed for tracking. '''
        self.frame_number = mbe.frame_number
        self.curve = mbe.curve

        f_classes, f_locs = _get_fc_lists(mbe, reclassify)

        self.fringes = [Fringe(fcls, floc, self.frame_number) for fcls, floc in izip(f_classes, f_locs)]
        for a, b in pairwise_periodic(self.fringes):
            a.insert_ahead(b)

    def __iter__(self):
        return self.fringes.__iter__()

    def __len__(self):
        return len(self.fringes)


def _clean_fringes(f_classes, f_locs):
    d = list(is_valid_3(*p) for p in triple_wise_periodic(f_classes))
    # rotate to deal with offset of 1 in is_valid_3
    d = d[-1:] + d[:-1]
    # if there are any in-valid fringes, we need to try and patch them up
    if not all(d):
        # if an invalid run spans the loop over point, rotate everything and has at least 2 fringes padding
        while d[0] is False or d[-1] is False or d[-1] is False or d[-2] is False:
            d = d[-1:] + d[:-1]
            f_classes = f_classes[-1:] + f_classes[:-1]
            f_locs = f_locs[-1:] + f_locs[:-1]

        # get the bad runs
        bad_runs = find_bad_runs(d)
        # provide some more padding
        bad_runs = [(br[0], br[1]) for br in bad_runs]
        fixes = list()
        for br in bad_runs:
            slc = slice(*br)
            working_classes = f_classes[slc]
            working_locs = f_locs[slc]
            if any([f.charge == 0 for f in working_classes]):
                # we have at least one zero fringe
                fix_ = _valid_run_fixes_with_hinting(working_classes, working_locs)
                fixes.append(fix_)
            else:
                fix_ = _valid_run_fixes(working_classes, working_locs)

                fixes.append([fix_[0],
                              [fix_[1]] * len(fix_[0])])

        best_lst = None
        min_miss = np.inf
        zero_count = 0
        prop_c_lsts, prop_l_lsts = zip(*fixes)

        for p_cls, p_loc in izip(product(*prop_c_lsts), product(*prop_l_lsts)):
            # get a copy of the list so we can mutate it
            working_class_lst = list(f_classes)
            working_locs_lst = list(f_locs)

            # apply the proposed changes
            for p, loc, br in zip(p_cls[::-1], p_loc[::-1], bad_runs[::-1]):
                slc = slice(*br)
                working_class_lst[slc] = p
                working_locs_lst[slc] = loc

            # count up how much we missed by
            miss_count = np.sum([forward_dh_dict[a, b] for a, b in pairwise_periodic(working_class_lst)])

            if miss_count == 0:
                zero_count += 1
            # if better than current best, store this one
            if np.abs(miss_count) < min_miss:
                min_miss = np.abs(miss_count)
                best_lst = (working_class_lst, working_locs_lst)
        print zero_count, '/', np.prod([len(_p) for _p in prop_l_lsts])
        f_classes, f_locs = best_lst
    return f_classes, f_locs


def _get_fc_lists(mbe, reclassify):

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
            top_lab = lab_regions[0,j]
            bot_lab = lab_regions[-1,j]
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

            fringe_rings.append(FringeRing(mbe, reclassify=reclassify))
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
        self.height_img = np.ones(self.label_regions.shape, dtype=np.float32) * np.nan
        self.height_map = np.ones(nb_br + nb_dr, dtype=np.float32) * np.nan
        self.region_fringes = [[] for j in range(nb_br + nb_dr)]
        self.working_img = working_img

        self.thresh = thresh
        self.structure = structure
        self.size_cut = size_cut

        for FR in self.fring_rings:
            for fr in FR:
                theta_indx = int((np.mod(fr.phi, 2 * np.pi) / (2 * np.pi)) * self.label_regions.shape[0])
                label = self.label_regions[theta_indx, fr.frame_number]
                fr.region = label
                fr.abs_height = np.nan
                self.region_fringes[label].append(fr)

    def display_height(self, ax=None, cmap='jet', bckgnd=True, alpha=.65, t_scale=1, t_units=''):
        height_img = self.height_img
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

        ax.set_xlabel(' '.join([r'$\tau$', t_units.strip()]) )
        ax.set_ylabel(r'$\theta$')

        data = self.label_regions


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
            origin='bottom',alpha=alpha)

        ax.figure.canvas.draw()


    def display_region(self, n, ax=None):

        if ax is None:
            # make this smarter
            ax = plt.gca()

        norm_br = matplotlib.colors.Normalize(vmin=.5, vmax=np.max(data), clip=False)
        my_cmap = cm.get_cmap('jet')
        my_cmap.set_under(alpha=0)

        ax.imshow(n * (data == n),
                  cmap=my_cmap,
                  norm=norm_br,
                  aspect='auto')
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
        self.height_img[self.label_regions == label] = height
        for _fr in self.region_fringes[label]:
            _fr.abs_height = height

    def set_height(self, frame_num, theta, height, overwrite=False):
        theta_indx = int((np.mod(theta, 2 * np.pi) / (2 * np.pi)) * self.label_regions.shape[0])

        label = self.label_regions[theta_indx, frame_num]
        # we can't do anything with points in the un-labeled regions of the image
        if label == 0:
            print 'label is 0'
            return
        if not overwrite and not np.isnan(self.height_map[label]):
            # we won't over-write at this region already has a height
            # TODO make this a custom exception
            raise RuntimeError("already set the height of this region!")

        self.height_map[label] = height
        self.height_img[self.label_regions == label] = height

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
                            dh_prev = prev.forward_dh()
                            self._set_height(fr, p_h + dh_prev)
                            try_again_flag = True

                    elif fr.valid_follower(nexp):
                        n_h = nexp.abs_height
                        if not np.isnan(n_h):
                            dh_next = fr.forward_dh()
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
        first_frame_dh, ff_phi = [np.array(_) for _ in zip(*[(fr.forward_dh(), fr.phi) for fr in FR])]
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
                self.set_height(0, ff_phi[j], h)
            except RuntimeError:
                eh = self.get_height(0, ff_phi[j])
                print eh, h
                h = eh

            h += first_frame_dh[j]

    def write_to_hdf(self, out_file, md_dict):

        # this will blow up if the file exists
        h5file = h5py.File(out_file.format, 'w-')
        try:
            file_out.attrs['ver'] = '0.1'
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



        finally:
            # make sure than no matter what we clean up after our selves
            h5file.close()

    @classmethod
    def from_hdf(cls, in_file):
        pass


def format_frac(fr):
    sp = str(fr).split('/')
    if len(sp) == 1:
        return sp[0]
    else:
        return r'$\frac{%s}{%s}$' % tuple(sp)
