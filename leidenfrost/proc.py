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
import gc

import leidenfrost.db as ldb
import leidenfrost.fringes as lf
import leidenfrost.backends as lb

import logging

import os
import numpy as np


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def proc_cine_to_h5(cine_fname, ch, hdf_fname_template, params, seed_curve):
    db = ldb.LFmongodb()
    logger = logging.getLogger('proc_cine_frame_' + str(os.getpid()))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    h5_fname = hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'h5'))

    lh = logging.FileHandler(hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'log')).format)

    lh.setFormatter(formatter)
    logger.addHandler(lh)

    _id, h5_fname = db.start_proc(ch, params, seed_curve.to_dict(), h5_fname)

    start_frame = params.pop('start_frame', 0)
    max_circ_change_frac = params.pop('max_circ_change_frac', None)

    if not os.path.isfile(h5_fname.format):
        print ('panic')
        logger.error("file already exists")
        db.remove_proc(_id)
        return

    stack = lb.ProcessBackend.from_args(cine_fname, **params)
    stack.gen_stub_h5(h5_fname.format, seed_curve)
    hfb = lb.HdfBackend(h5_fname, cine_base_path=cine_fname.base_path, mode='rw')
    file_out = hfb.file
    logger.info('created file')

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        for j in xrange(start_frame, len(stack)):
            # set a 30s window, if the frame does not finish on 30s, kill it
            if hfb.contains_frame(j):
                logger.warn('deleting existing frame {0}'.format(j))
                hfb._del_frame(j)

            signal.alarm(45)
            start = time.time()
            mbe, new_seed_curve = stack.process_frame(j, seed_curve)
            if max_circ_change_frac is not None:
                # check if we are limiting how much the circumference can change
                # between frames
                old_circ = seed_curve.circ
                new_circ = new_seed_curve.circ
                # if it changes little enough, adopt the new seed curve
                if np.abs(old_circ - new_circ) / old_circ < max_circ_change_frac:
                    seed_curve = new_seed_curve
            else:
                seed_curve = new_seed_curve
            signal.alarm(0)
            elapsed = time.time() - start
            logger.info('completed frame %d in %fs', j, elapsed)
            # set alarm to 0

            mbe.write_to_hdf(file_out)

            del mbe
            file_out.flush()
            gc.collect()
    except TimeoutException:
        # handle the time out error
        logger.warn('timed out')
        db.timeout_proc(_id)
    except Exception as e:
        # handle all exceptions we should get
        logger.warn(str(e))
    except:
        # handle everything else
        logger.warn('raised exception not derived from Exception')
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


def proc_h5_to_RM(h5_fname, output_file_template, RM_params, fname_mutator=None, cine_base_path=None):
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

    out_fname = output_file_template._replace(fname=fname_mutator(h5_fname.fname))

    h5_backend = lb.HdfBackend(h5_fname, cine_base_path=cine_base_path)

    RM = lf.Region_map.from_backend(h5_backend, **RM_params)

    RM.write_to_hdf(out_fname, md_dict=RM_params)
