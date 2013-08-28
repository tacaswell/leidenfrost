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
import leidenfrost.backends as lfbe
import logging

import os
import numpy as np


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def proc_cine_to_h5(cine_fname, ch, hdf_fname_template, params, seed_curve, _id=None):
    logger = logging.getLogger('proc_cine_frame_' + str(os.getpid()))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    h5_fname = hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'h5'))

    lh = logging.FileHandler(hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'log')).format)

    lh.setFormatter(formatter)
    logger.addHandler(lh)

    start_frame = params.pop('start_frame', 0)

    max_circ_change_frac = params.pop('max_circ_change_frac', None)

    if not os.path.isfile(h5_fname.format):
        stack = lfbe.ProcessBackend.from_args(cine_fname, **params)
        stack.gen_stub_h5(h5_fname.format, seed_curve)
        hfb = lfbe.HdfBackend(h5_fname, cine_base_path=cine_fname.base_path, mode='rw')
        file_out = hfb.file
        logger.info('created file')
    else:
        hfb = lfbe.HdfBackend(h5_fname, cine_base_path=cine_fname.base_path, mode='rw')
        logger.info('opened file')
        file_out = hfb.file
        # make sure that we continue with the same parameters
        params = dict((k, hfb.proc_prams[k]) for k in params)
        stack = lfbe.ProcessBackend.from_args(cine_fname, ver=hfb.ver, **params)

    if _id is not None:
        db = ldb.LFmongodb()
        db.store_proc(ch, _id, h5_fname)

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
        logger.warn('timed out')
    except Exception as e:
        logger.warn(str(e))
    except:
        logger.warn('raised exception not derived from Exception')

    finally:
        # make sure that no matter what the output file gets cleaned up
        file_out.close()
        # reset the alarm
        signal.alarm(0)
        # rest the signal handler
        signal.signal(signal.SIGALRM, old_handler)
        logger.removeHandler(lh)

    return None
