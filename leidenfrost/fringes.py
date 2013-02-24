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

fringe_c = namedtuple('fringe_c', ['color', 'charge', 'hint'])

# set up all the look-up dictionaries
# list of the needed combinations
fringe_type_list = [fringe_c(1, 0, 1),
                    fringe_c(1, 0, -1),
                    fringe_c(1, 1, 0),
                    fringe_c(1, -1, 0),
                    fringe_c(-1, 0, 1),
                    fringe_c(-1, 0, -1),
                    fringe_c(-1, 1, 0),
                    fringe_c(-1, -1, 0)]

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


### iterator toys

# ganked from docs
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
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



def find_bad_runs(d):
    '''
    Takes in a list of bools, returns a list of tuples that are continuous False runs.

    Does not know if the list in periodic, so a run that is split across the beginging/end will
    be 2 slices
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
    fringe_list = list(fringe_list) # make a copy, and make sure it is really a list
    ln_fr_lst = len(fringe_list)
    zero_hints = hint_zero_charges(fringe_list)
    multiple_hint_region = set() # multiple valid hint configurations
    invalid_hint_region = set()  # no valid hint configurations, need to add a fringe
    for s, res in zero_hints.iteritems():
        print s, res
        if len(res) == 1:
            # replace the fringes in the fringe list with the deduced hinting
            # the fringe_c objects are immuatble, so things that refer to this are now broken
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

    d = list(is_valid_3(*p) for p in triple_wise_periodic(fringe_list))
    d = d[:-1] + d[1:]  # to account for the offset in is_valid_3
    if all(d):
        # we are done, awesome
        print 'woo'
        return fringe_list, fringe_locs

    bad_runs = find_bad_runs(d)
    ### TODO add check to make sure that the invalid fringe does not span the circular buffer edge

    ### filter out the runs that have multiple valid hinting configurations (still not sure that can really exist
    slices = [slice(*j) for j in bad_runs]
    trial_corrections = []
    for s_ in slices:
        work_run = None


def is_valid_3(a, b, c):
    '''
    returns if this is a valid sequence of fringes
    '''
    return a in valid_precede_dict[b] and c in valid_follow_dict[b]


def is_valid_run(args):
    return all([p in valid_precede_dict[f] for p, f in pairwise(args)])


def get_valid_between(a, b):
    v_follow = set(valid_follow_dict[a])
    v_prec = set(valid_precede_dict[b])
    return list(v_follow & v_prec)


#ganked from docs
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def pairwise_periodic(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..., (sn,s0)"
    a, b = tee(iterable)
    b = cycle(b)
    next(b, None)
    return izip(a, b)


def triple_wise_periodic(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), ..., (sn,s0,s1)"
    a, _b = tee(iterable)
    b, c = tee(cycle(_b))
    next(b, None)
    next(c, None)
    next(c, None)

    return izip(a, b, c)


def triple_wise(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), "
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)

    return izip(a, b, c)


def hint_zero_charges_run(a, zero_run, c):
    valid_runs = []
    for trial_hints in product([1, -1], repeat=len(zero_run)):
        trial_run = [a] + [_f._replace(hint=th) for _f, th in izip(zero_run, trial_hints)] + [c]
        if is_valid_run(trial_run):
            valid_runs.append(trial_hints)
    return valid_runs


def hint_zero_charges(fringes):
    j = 0
    f_count = len(fringes)
    #k -> j, v -> hint or None
    hints_dict = {}
    while j < f_count:
        b = fringes[j]
        if b.charge == 0:
            zero_run = deque([b])
            start_j = j
            a = fringes[(start_j - 1) % f_count]
            if len(zero_run) < f_count:
                while a.charge == 0:
                    start_j -= 1
                    zero_run.appendleft(a)
                    if len(zero_run) == f_count:
                        break

                    a = fringes[(start_j - 1) % f_count]

            end_j = j + 1
            c = fringes[(end_j) % f_count]
            if len(zero_run) < f_count:
                while c.charge == 0:
                    zero_run.append(c)
                    if len(zero_run) == f_count:
                        break
                    end_j += 1
                    c = fringes[(end_j) % f_count]

            hints_dict[(start_j, end_j)] = hint_zero_charges_run(a, zero_run, c)
        j += 1
    return hints_dict
