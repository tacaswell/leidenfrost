from multiprocessing import Process, JoinableQueue
from itertools import izip
import argparse
import signal
import logging
import time
import itertools

import os
from leidenfrost import FilePath
import leidenfrost.infra as li
import leidenfrost.file_help as lffh
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
        Work queue is a joinable queue, res_queue can be any sort of
        thing that supports put()

        """
        # background set up that must be done
        Process.__init__(self)
        self.daemonic = True
        self.work_queue = work_queue

    def run(self):
        """
        The assumption is that these will be run daemonic and reused for
        multiple work sessions

        """
        while True:
            work_arg = self.work_queue.get()
            if work_arg is None:          # poison pill
                self.work_queue.task_done()
                return
            try:
                cine_fname, cine_hash, hdf_fname_template = work_arg
                proc_cine_fname(cine_fname, cine_hash, hdf_fname_template)
            except Exception as E:
                # we want to catch _EVERYTHING_ so errors don't blow
                # up the other computations with it
                print "on file {}".format(cine_fname.fname)
                print E

            self.work_queue.task_done()


def proc_cine_fname(cine_fname, ch, hdf_fname_template):
    logger = logging.getLogger('proc_cine_frame_' + str(os.getpid()))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    db = ldb.LFmongodb()

    config_dict_lst = db.get_unproced_configs(ch)

    for config_dict in config_dict_lst:
        print cine_fname

        h5_fname = hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'h5'))

        lh = logging.FileHandler(hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'log')).format)

        lh.setFormatter(formatter)
        logger.addHandler(lh)

        seed_curve = li.SplineCurve.from_pickle_dict(config_dict['curves']['0'])

        params = config_dict['config']
        start_frame = params.pop('start_frame', 0)
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
            stack = lfbe.ProcessBackend.from_args(cine_fname, **params)

        db.store_proc(ch, config_dict['_id'], h5_fname)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)

        # move the logging to the top

        try:
            for j in xrange(start_frame, len(stack)):
                # set a 30s window, if the frame does not finish on 30s, kill it
                if hfb.contains_frame(j):
                    logger.warn('deleting existing frame {0}'.format(j))
                    hfb._del_frame(j)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_path",
                        help="path, relative to base path of hdf file")
    parser.add_argument("cine_base_path",
                        help="The base path for where the (cine) " +
                              "data files are located",
                        nargs='*')
    parser.add_argument("--hdf_base_path",
                        help="The base path for " +
                        "where the hdf data file are located.  If " +
                        "not given, assumed to be the " +
                        "same as cine_base_path")
    parser.add_argument("--cine_search_path",
                        help="Path relative to cine_base_path to " +
                        "look for cine files")
    parser.add_argument("--N",
                        help="number of files to process simultaneously")

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
            search_path = args.cine_search_path
        else:
            search_path = 'leidenfrost'

        cines = lffh.get_cine_hashes(cine_base_path, search_path)
        cine_fnames.extend(zip(cines,
                                itertools.repeat(FilePath(hdf_base_path,
                                                          args.hdf_path, ''))))

    # don't start more processes than we could ever use
    N = max(N, len(cine_fnames))

    # stet up queues
    WORK_QUEUE = JoinableQueue()
    PROCS = [worker(WORK_QUEUE) for j in range(N)]
    # start workers
    for p in PROCS:
        p.start()

    # put the work in the queue
    for (cf, ch), hdf_fname_template in cine_fnames:
        print 'adding {}/{} to work queue'.format(cf.path, cf.fname)
        WORK_QUEUE.put((cf, ch, hdf_fname_template))

    # poison the worker processes
    for j in range(len(PROCS)):
        WORK_QUEUE.put(None)

    # wait for everything to finish
    WORK_QUEUE.join()
