from multiprocessing import Process, JoinableQueue
import cine
import argparse

import os
from leidenfrost import FilePath
import leidenfrost.infra as li
import leidenfrost.db as ldb
import leidenfrost.backends as lfbe
import h5py


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
                return
            try:
                cine_fname, hdf_fname_template = work_arg
                proc_cine_fname(cine_fname, hdf_fname_template)
            except Exception as E:
                # we want to catch _EVERYTHING_ so errors don't blow up the other computations with it
                print E

            self.work_queue.task_done()


def proc_cine_fname(cine_fname, hdf_fname_template):
    db = ldb.LFmongodb()

    ch = cine.Cine(cine_fname.format).hash
    config_dict_lst = db.get_unproced_configs(ch)

    for config_dict in config_dict_lst:
        print cine_fname

        h5_fname = hdf_fname_template._replace(fname=cine_fname.fname.replace('cine', 'h5'))
        db.store_proc(ch, config_dict['_id'], h5_fname)

        seed_curve = li.SplineCurve.from_pickle_dict(config_dict['curves']['0'])

        params = config_dict['config']
        stack = lfbe.ProcessBackend.from_args(cine_fname, **params)
        stack.gen_stub_h5(h5_fname.format, seed_curve)

        file_out = h5py.File(h5_fname.format, 'r+')
        try:
            for j in range(len(stack)):
                mbe, seed_curve = stack.process_frame(j, seed_curve)
                mbe.write_to_hdf(file_out)
                del mbe
                file_out.flush()

        finally:
            # make sure that no matter what the output file gets cleaned up
            file_out.close()

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cine_base_path", help="The base path for where the (cine) data files are located")
    parser.add_argument("hdf_path", help="path, relative to base path of hdf file")
    parser.add_argument("--cine_search_path", help="The path to search for cine to process.  If not given, assumed to be cine_base_path")
    parser.add_argument("--hdf_base_path", help="The base path for where the hdf data file are located.  If not given, assumed to be the same as cine_base_path")
    parser.add_argument("--N", help="number of files to process simultaneously")

    args = parser.parse_args()

    cine_base_path = args.cine_base_path
    if args.hdf_base_path:
        hdf_base_path = args.hdf_base_path
    else:
        hdf_base_path = cine_base_path

    if args.cine_search_path:
        search_path = cine_base_path + '/' + args.cine_search_path
    else:
        search_path = cine_base_path + '/' + 'leidenfrost'

    if args.N:
        N = args.N
    else:
        N = 8

    # template for the output file names
    hdf_fname_template = FilePath(hdf_base_path, args.hdf_path, '')

    cine_fnames = []
    for dirpath, dirnames, fnames in os.walk(search_path):
        cine_fnames.extend([FilePath(cine_base_path, dirpath[len(cine_base_path) + 1:], f) for f in fnames if 'cine' in f])

    # special check
    cine_fnames = [cf for cf in cine_fnames if '320C_round_mode1.cine' != cf.fname]

    WORK_QUEUE = JoinableQueue()
    PROCS = [worker(WORK_QUEUE) for j in range(N)]
    for p in PROCS:
        p.start()

    for cf in cine_fnames:
        print 'adding', cf
        WORK_QUEUE.put((cf, hdf_fname_template))

    # poison the worker processes
    for j in range(len(PROCS)):
        WORK_QUEUE.put(None)

    # wait for everything to finish
    WORK_QUEUE.join()
