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


def cleanup_fringes(fringe_list, fringe_locs):
    '''
    Assume that we only have (color, charge, hint) tuples, adapt this
    to objects later once we know _how_ to do this
    '''
    # step one, hint the zeros
    fringe_list = list(fringe_list)  # make a copy, and make sure it is really a list
    ln_fr_lst = len(fringe_list)
    zero_hints = hint_zero_charges(fringe_list)
    multiple_hint_region = set()  # multiple valid hint configurations
    invalid_hint_region = set()  # no valid hint configurations, need to add a fringe
    for s, res in zero_hints.iteritems():
        print s, res
        if len(res) == 1:
            # replace the fringes in the fringe list with the deduced hinting
            # the fringe_cls objects are immuatble, so things that refer to this are now broken
            if s[1] < ln_fr_lst:
                sl = slice(*s)
                fringe_list[sl] = [_f._replace(hint=th) for _f, th in izip(fringe_list[sl], res[0])]
            else:
                raise NotImplementedError("need to write this code")

        elif len(res) == 0:
            invalid_hint_region.add(s)
        elif len(res) > 1:
            multiple_hint_region.add(s)
        else:
            raise RuntimeError("Should never hit this")
    if all(d):
        # we are done, awesome
        print 'woo'
        return fringe_list, fringe_locs


    ### TODO add check to make sure that the invalid fringe does not span the circular buffer edge

    ### filter out the runs that have multiple valid hinting configurations (still not sure that can really exist
    slices = [slice(*j) for j in bad_runs]
    trial_corrections = []
    for s_ in slices:
        work_run = None


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
    print 'zero runs', zero_runs
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


        # this section still needs work
        d = [True] + d + [True]  # add edges back on
        res = find_bad_runs(d)
        slices = [slice(*_r) for _r in res]

        valid_sub_regions = [_valid_run_fixes(trial_run[s_], trial_locs[s_], add_zcharge=False) for s_ in slices]
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

    return valid_runs, valid_locs


def _valid_run_fixes(run, locs, add_zcharge=True):
    '''
    This takes a run of fringes with no hinting issues and returns possible
    '''
    possible_insertions = []
    possible_locs = []
    for (a, b), (a_loc, b_loc) in izip(pairwise(run), pairwise(locs)):
        pi = set(valid_follow_dict[a]) & set(valid_precede_dict[b])
        if not add_zcharge:
            pi = [_p for _p in pi if _p.charge != 0]
        possible_insertions.append(pi)
        phi_dist = (a_loc[1] - b_loc[1])
        if phi_dist > np.pi:
            phi_dist = 2 * np.pi - phi_dist
        phi_dist /= 2

        possible_locs.append(fringe_loc(0, b_loc[1] + phi_dist))

    vaild_runs = []

    for pi in product(*possible_insertions):
        pos_run = list(roundrobin(run, pi))

        if is_valid_run(pos_run):
            vaild_runs.append(pos_run)

    new_locs = list(roundrobin(locs, possible_locs))
    return vaild_runs, new_locs


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

        self.abs_height = None   #: the height of this fringe as given by tracking in time

        self.slope_f = None  #: the slope going forward
        self.slope_r = None  #: the slope going backward
        self.slope = None    #: the 'average' slope at this point

        self.frame_number = frame_number    #: the frame that this fringe belongs to

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

t_format_dicts = [{1:'B', -1:'D'},
                  {1:'L', 0:'_', -1:'R'},
                  {1:'P', 0:'_', -1:'S'}]


def format_fringe_txt(f):
    return ''.join([_d[_f] for _d, _f in izip(t_format_dicts, f)])


class FringeRing(object):
    '''
    A class to carry around Fringe data
    '''
    def __init__(self, mbe):
        '''Extracts the data from the mbe, cleans up the fringes, and constructs the `Fringe` objects needed for tracking. '''
        self.frame_number = mbe.frame_number
        self.curve = mbe.curve

        f_classes, f_locs = _get_fc_lists(mbe)

        d = deque(is_valid_3(*p) for p in triple_wise_periodic(f_classes))
        # rotate to deal with offset of 1 in is_valid_3
        d.rotate(1)
        # if there are any in-valid fringes, we need to try and patch them up
        if not all(d):
            # if an invalid run spans the loop over point, rotate everything and has at least 2 fringes padding
            while d[0] is False or d[-1] is False or d[-1] is False or d[-2] is False:
                d.rotate(1)
                f_classes = f_classes[-1:] + f_classes[:-1]
                f_locs = f_locs[-1:] + f_locs[:-1]

            # get the bad runs
            bad_runs = find_bad_runs(d)
            # provide some more padding
            bad_runs = [(br[0] - 1, br[1] + 1) for br in bad_runs]
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

                    fixes.append([fix_[0], [fix_[1]] * len(fix_[0])])

            best_lst = None
            min_miss = np.inf

            prop_c_lsts, prop_l_lsts = zip(*fixes)
            print [len(_p) for _p in prop_l_lsts]
            for p_cls, p_loc in izip(product(*prop_c_lsts), product(*prop_l_lsts)):
                # get a copy of the list so we can mutate it
                working_class_lst = list(f_classes)
                working_locs_lst = list(f_locs)

                # apply the proposed changes
                for p, loc, br in zip(p_cls[::-1], p_loc[::-1], bad_runs[::-1]):
                    slc = slice(*br)
                    working_class_lst[slc] = p
                    working_locs_lst[slc] = loc

                print is_valid_run_periodic(working_class_lst)

                # count up how much we missed by
                miss_count = np.sum([forward_dh_dict[a, b] for a, b in pairwise_periodic(working_class_lst)])
                # if better than current best, store this one
                if np.abs(miss_count) < min_miss:
                    min_miss = np.abs(miss_count)
                    best_lst = (working_class_lst, working_locs_lst)

            print min_miss
            f_classes, f_locs = best_lst

        self.fringes = [Fringe(fcls, floc, self.frame_number) for fcls, floc in izip(f_classes, f_locs)]


def _get_fc_lists(mbe):

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

    for color, trk_lst in izip(colors, mbe.trk_lst):
        for t in trk_lst:
            t.classify2()
            if t.charge is not None:
                f_locs.append(fringe_loc(t.q, np.mod(t.phi + th_offset, 2 * np.pi)))
                f_classes.append(fringe_cls(color, t.charge, 0))
            else:
                junk_fringes.append(t)

    f_classes, f_locs = zip(*sorted(zip(f_classes, f_locs), key=lambda x: x[1][1]))
    # TODO: deal with junk fringes in sensible way

    return list(f_classes), list(f_locs)
