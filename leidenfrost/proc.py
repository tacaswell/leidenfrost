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
from __future__ import division, print_function
import signal
import time

import copy
import logging
import os
import datetime

import numpy as np

import leidenfrost
import leidenfrost.db as ldb
import leidenfrost.fringes as lf
import leidenfrost.backends as lb
import leidenfrost.file_help as lffh


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def proc_cine_to_h5(ch, hdf_fname_template, params, seed_curve):
    """
    Processes a cine path -> h5

    Parameters
    ----------
    ch : str
        Hash of the cine_fname

    hdf_fname_template: FilePath
        Template for where to put the output file + log files

    params : dict
        Parameters to use to process the cine file

    seed_curve : int
        The first frame to process

    """
    i_disk_dict = {0: u'/media/tcaswell/leidenfrost_a', 1: u'/media/tcaswell/leidenfrost_c'}
    # make data base communication object
    db = ldb.LFmongodb(i_disk_dict=i_disk_dict)
    # set up logging stuff
    logger = logging.getLogger('proc_cine_frame_' + str(os.getpid()))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # make a copy so we don't export side effects
    params = copy.copy(params)

    # convert disk
    hdf_fname_template = leidenfrost.convert_base_path(hdf_fname_template,
                                                       i_disk_dict)
    cine_md = db.get_movie_md(ch)
    cine_fname = leidenfrost.FilePath.from_db_dict(cine_md['fpath'], i_disk_dict)

    # sort out output files names
    h5_fname = hdf_fname_template._replace(
        fname=cine_fname.fname.replace('cine', 'h5'))
    # get _id from DB
    _id, h5_fname = db.start_proc(ch, params, seed_curve, h5_fname)
    lh = logging.FileHandler(hdf_fname_template._replace(
        fname=h5_fname.fname.replace('h5', 'log')).format)

    start_frame = params.pop('start_frame', 0)
    end_frame = params.pop('end_frame', -1)
    if end_frame > 0 and end_frame < start_frame:
        raise Exception("invalid start and end frames")

    max_circ_change_frac = params.pop('max_circ_change', None)

    if os.path.isfile(h5_fname.format):
        print ('panic!')
        logger.error("file already exists")
        db.remove_proc(_id)
        return

    stack = lb.ProcessBackend.from_args(cine_fname, **params)
    stack.gen_stub_h5(h5_fname.format, seed_curve, start_frame)
    hfb = lb.HdfBackend(h5_fname,
                        cine_base_path=cine_fname.base_path,
                        mode='rw')
    file_out = hfb.file
    logger.info('created file')

    if end_frame == -1:
        end_frame = len(stack)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        lh.setFormatter(formatter)
        logger.addHandler(lh)

        for j in xrange(start_frame, end_frame):
            # set a 30s window, if the frame does not finish on 30s, kill it
            if hfb.contains_frame(j):
                logger.warn('deleting existing frame {0}'.format(j))
                hfb._del_frame(j)

            signal.alarm(45)
            start = time.time()
            mbe, new_seed_curve = stack.process_frame(j, seed_curve)
            if max_circ_change_frac is not None:
                # check if we are limiting how much the circumference
                # can change between frames
                old_circ = seed_curve.circ
                new_circ = new_seed_curve.circ
                # if it changes little enough, adopt the new seed
                # curve
                if (np.abs(old_circ - new_circ) / old_circ
                    < max_circ_change_frac):
                    seed_curve = new_seed_curve
            else:
                seed_curve = new_seed_curve
            signal.alarm(0)
            elapsed = time.time() - start
            logger.info('completed frame %d in %fs', j, elapsed)
            # set alarm to 0

            mbe.write_to_hdf(file_out)

            if j % 500 == 0:
                file_out.flush()
            del mbe
            #gc.collect()
    except TimeoutException:
        # handle the time out error
        logger.warn('timed out')
        # tell the DB we timed out
        db.timeout_proc(_id)
    except Exception as e:
        # handle all exceptions we should get
        logger.warn(str(e))
        db.error_proc(_id)
    except:
        # handle everything else
        logger.warn('raised exception not derived from Exception')
        db.error_proc(_id)
    else:
        # if we ran through the full movie, mark it done (yay)
        db.finish_proc(_id)
    finally:
        # make sure that no matter what the output file gets cleaned up
        file_out.close()
        # reset the alarm
        signal.alarm(0)
        # rest the signal handler
        signal.signal(signal.SIGALRM, old_handler)
        logger.removeHandler(lh)

    return None


def proc_h5_to_RM_tofile(h5_fname,
                         output_file_template, RM_params,
                         fname_mutator=None, cine_base_path=None):
    """
    Runs the RM code on an h5 file and writes the result to disk.

    Parameters
    ----------
    h5_fname : FilePath
        the h5_file to load

    output_file_template : FilePath
        Template for the output file, fname will be replaced

    RM_params : dict
        the parameters to be passed to Region_map.from_backend

    frame_mutator : function or None
        function that takes one string as it's argument and returns a fname
        for the RM output based on it.  If None, then '.RM' is appended.

    cine_base_path : str or None
        Base path for the cine file.  If None, assumed to be the base_path in
        h5_fname


    """
    if fname_mutator is None:
        fname_mutator = lambda x: x + '.RM'

    if cine_base_path is None:
        cine_base_path = h5_fname.base_path

    out_fname = output_file_template._replace(
        fname=fname_mutator(h5_fname.fname))

    RM, h5_backend = proc_h5_to_RM(h5_fname, RM_params, cine_base_path)

    RM.write_to_hdf(out_fname, md_dict=RM_params)


def proc_h5_to_RM(h5_name, RM_params, cine_base_path=None):
    if cine_base_path is None:
        cine_base_path = h5_name.base_path

    h5_backend = lb.HdfBackend(h5_name, cine_base_path=cine_base_path)

    return _proc_h5_to_RM(h5_backend, RM_params)


def proc_h5list_to_RM(h5_list, RM_params, cine_base_path):
    h5_backend = lb.MultiHdfBackend(h5_list, cine_base_path)

    return _proc_h5_to_RM(h5_backend, RM_params)


def _proc_h5_to_RM(h5_backend, RM_params, cine_base_path=None):

    RM = lf.Region_map.from_backend(h5_backend, **RM_params)

    return RM, h5_backend


def process_split_RM(k, cache_path, RM_params,
                     i_disk_dict,
                     section_per_sec=2):
    """
    Processes fringes -> region_map by segment and saves the result to
    disk

    Snarfs all exceptions

    Parameters
    ----------
    k : cine_hash
        The move to work on

    RM_params : dict
        Paramaters to pass to Region_map.from_backend

    i_disk_dict : dict
       mapping between disk number -> disk path

    section_per_sec : int, optional
       The number of segments per second
    """
    db = ldb.LFmongodb(i_disk_dict=i_disk_dict)
    v = db.get_movie_md(k)

    # make the hdf backend object to
    hbe = lb.hdf_backend_factory(k, i_disk_dict=i_disk_dict)
    # set up steps to half second chunks
    N_step = S_step = hbe.frame_rate // section_per_sec
    out_path_template = leidenfrost.FilePath(cache_path,
                                 datetime.date.today().isoformat(),
                                 '')
    # make sure the path exists
    lffh.ensure_path_exists(out_path_template.format)
    for j in xrange(hbe.first_frame, hbe.last_frame - S_step, N_step):
        # make output name
        out_path = out_path_template._replace(
            fname='RM_{}_{:06}-{:06}_{}.h5'.format(k,
                                                   j,
                                                   j+N_step,
                                                   v['fpath']['fname'][:-5]))

        # compute the RM
        _rm = lf.Region_map.from_backend(hbe,
                                         f_slice=slice(j, j + S_step),
                                         **RM_params)
        # write it out to disk
        _rm.write_to_hdf(out_path)
