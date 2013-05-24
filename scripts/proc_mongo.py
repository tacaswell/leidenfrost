from multiprocessing import Process, JoinableQueue
import cine
import argparse
import signal
import logging
import time
import itertools

import os
from leidenfrost import FilePath
import leidenfrost.infra as li
import leidenfrost.db as ldb
import leidenfrost.backends as lfbe
import h5py


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class worker(Process):
    """Worker class for farming out the work of doing the least
    squares fit

    """

    def __init__(self,
                 work_queue):
        """
        Work queue is a joinable queue, res_queue can be any sort of thing that supports put()
        """
        # background set up that must be done
        Process.__init__(self)
        self.daemonic = True
        self.work_queue = work_queue

    def run(self):
        """
        The assumption is that these will be run daemonic and reused for multiple work sessions
        """
        while True:
            work_arg = self.work_queue.get()
            if work_arg is None:          # poison pill
                self.work_queue.task_done()
                return
            try:
                cine_fname, hdf_fname_template = work_arg
                proc_cine_fname(cine_fname, hdf_fname_template)
            except Exception as E:
                # we want to catch _EVERYTHING_ so errors don't blow up the other computations with it
                print E

            self.work_queue.task_done()


def proc_cine_fname(cine_fname, hdf_fname_template):
    logger = logging.getLogger('proc_cine_frame_' + str(os.getpid()))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    db = ldb.LFmongodb()

    ch = cine.Cine(cine_fname.format).hash
    config_dict_lst = db.get_unproced_configs(ch)

    for config_dict in config_dict_lst:
        print cine_fname

        h5_fname = hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'h5'))
        db.store_proc(ch, config_dict['_id'], h5_fname)

        lh = logging.FileHandler(hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'log')).format)

        lh.setFormatter(formatter)
        logger.addHandler(lh)

        seed_curve = li.SplineCurve.from_pickle_dict(config_dict['curves']['0'])

        params = config_dict['config']
        stack = lfbe.ProcessBackend.from_args(cine_fname, **params)
        stack.gen_stub_h5(h5_fname.format, seed_curve)

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        file_out = h5py.File(h5_fname.format, 'r+')
        try:
            for j in range(len(stack)):

                # set a 10s window, if the frame does not finish on 10s, kill it
                signal.alarm(30)
                start = time.time()
                mbe, seed_curve = stack.process_frame(j, seed_curve)
                signal.alarm(0)
                elapsed = time.time() - start
                logger.info('completed frame %d in %fs', j, elapsed)
                # set alarm to 0

                mbe.write_to_hdf(file_out)
                del mbe
                file_out.flush()
        except TimeoutException:
            logger.warn('timed out')

        finally:
            # make sure that no matter what the output file gets cleaned up
            file_out.close()
            # reset the alarm
            signal.alarm(0)
            # rest the signal handler
            signal.signal(signal.SIGALRM, old_handler)
            logger.removeHandler(lh)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_path", help="path, relative to base path of hdf file")
    parser.add_argument("cine_base_path",
                        help="The base path for where the (cine) data files are located",
                        nargs='*')
    parser.add_argument("--hdf_base_path", help="The base path for where the hdf data file are located.  If not given, assumed to be the same as cine_base_path")
    parser.add_argument("--N", help="number of files to process simultaneously")

    args = parser.parse_args()

    if args.N:
        N = args.N
    else:
        N = 8

    cine_fnames = []
    for cine_base_path in args.cine_base_path:

        if args.hdf_base_path:
            hdf_base_path = args.hdf_base_path
        else:
            hdf_base_path = cine_base_path

        if args.cine_search_path:
            search_path = cine_base_path + '/' + args.cine_search_path
        else:
            search_path = cine_base_path + '/' + 'leidenfrost'

        # template for the output file names

        for dirpath, dirnames, fnames in os.walk(search_path):
            cine_fnames.extend(zip([FilePath(cine_base_path, dirpath[len(cine_base_path) + 1:], f) for f in fnames if 'cine' in f],
                                   itertools.repeat(FilePath(hdf_base_path, args.hdf_path, ''))))

    WORK_QUEUE = JoinableQueue()
    PROCS = [worker(WORK_QUEUE) for j in range(N)]
    for p in PROCS:
        p.start()

    for cf, hdf_fname_template in cine_fnames:
        print 'adding', cf
        WORK_QUEUE.put((cf, hdf_fname_template))

    # poison the worker processes
    for j in range(len(PROCS)):
        WORK_QUEUE.put(None)

    # wait for everything to finish
    WORK_QUEUE.join()
