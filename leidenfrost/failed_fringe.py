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
from itertools import izip, product, tee, cycle, islice
from leidenfrost.fringes import valid_follow_dict, valid_precede_dict, pairwise_periodic, fringe_loc, forward_dh_dict

import numpy as np


#ganked from docs
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ...

    copied from documentation
    """
    a, b = tee(iterable)
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


class FringeRing(object):
    '''
    A class to carry around Fringe data
    '''
    def __init__(self, tau):
        self.fringes = []
        self.tau = tau
        self.invalid_fringes = []
        self.curve = None

    def link_to_next_ring(self, other):
        pass

    def set_forward_deltas(self):
        for f in self.fringes:
            f.determine_forward_dh()
        return np.cumsum([f.f_dh for f in self.fringes])

    def find_invalid_sequence(self):
        list_tmp = []
        for f in self.fringes:
            valid, configs = f.is_valid_order()
            if not valid:
                pre_f = f.prev_P
                invalid_fringe = Fringe(0, pre_f.phi + pre_f.distance(f) / 2, frame_number=f.frame_number)
                f.insert_behind(invalid_fringe)
                # if there is only one option, just set it
                if len(configs) == 1:
                    invalid_fringe.set_color_charge(*configs[0])
                    list_tmp.append(invalid_fringe)
                else:
                    self.invalid_fringes.append((invalid_fringe, configs))

        self.fringes.extend([f for f, _ in self.invalid_fringes])
        self.fringes.extend(list_tmp)
        self.fringes.sort(key=lambda x: x.phi)

    def set_reverse_deltas(self):
        raise NotImplementedError()

    def compute_cumulative_forward(self):
        return np.cumsum([f.f_dh for f in self.fringes])

    def compute_cumulative_reverse(self):
        raise NotImplementedError()

    def plot_fringes(self, ax):
        colors = ['r', 'b']
        shapes = ['^', 'o', 'v']
        lines = []
        for color, c in zip([-1, 1], colors):
            for charge, s in zip([-1, 0, 1], shapes):
                phi, q = zip(*[(fr.phi, fr.q) for fr in self.fringes if fr.color == color and fr.charge == charge])
                x, y = self.curve.q_phi_to_xy(q, phi)
                lines.extend(ax.plot(x, y, linestyle='none', marker=s, color=c))
        return lines

    def return_tracking_lists(self):
        return [[fr for fr in self.fringes if fr.color == color and fr.charge == charge] for (color, charge) in
                itertools.product([-1, 1], repeat=2)]

def get_rf(hf, j):
    mbe = hf[j]
    th_offset = mbe.get_theta_offset()
    rf = FringeRing(mbe.res[0][1],
                    mbe.res[0][0],
                    th_offset=th_offset,
                    ringID=j)
    return rf


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
            print best_accum, len(source), len(dest), len(cur_order)
            print cur_order
            # we have a winner
            best_order = cur_order[:]      # get a copy
            best_order.extend([1] * len(dest))
            return best_order, tmp_accum
        else:
            # old way is still best
            return best_order, best_accum

    if len(dest) == 0:
        tmp_accum = cur_accum + len(source) * max_search_range
        if tmp_accum < best_accum:
            print best_accum, len(source), len(dest), len(cur_order)
            print cur_order
            best_order = cur_order[:]      # get a copy
            best_order.extend([1] * len(source))
            return best_order, tmp_accum

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


def link_run(source, dest, max_search_range):
    '''
    wrapper function for _link_run that handles setting up and parsing the output

    assumes the objects that come in have a `set_next` method
    '''
    best_accum = ((len(source) + len(dest)) / 4) * max_search_range
    best_order = []
    cur_accum = 0
    cur_order = []
    wsource = list(source)
    wdest = list(dest)

    best_order, best_accum = _link_run(best_accum, best_order, cur_accum, cur_order, wsource, wdest, max_search_range)

    wsource = list(source)
    wdest = list(dest)
    res = []
    print best_order
    for r in best_order:
        if r == 0:
            res.append((wsource.pop(0), wdest.pop(0)))
        elif r == -1:
            res.append((None, wdest.pop(0)))
        elif r == 1:
            res.append((wsource.pop(0), None))
        else:
            raise RuntimeError("should never reach this")

    return res, best_accum
