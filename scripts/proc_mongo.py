from multiprocessing import Pool
import cine
import argparse
import itertools
import os
from leidenfrost import FilePath
import leidenfrost.infra as li
import leidenfrost.db as ldb
import leidenfrost.backends as lfbe
import h5py


def proc_cine_fname(arg):
    db = ldb.LFmongodb()
    cine_fname, hdf_fname_template = arg
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
                mbe,seed_curve = stack.process_frame(j, seed_curve)
                mbe.write_to_hdf(file_out)
                del mbe
                file_out.flush()

        finally:
            # make sure that no matter what the output file gets cleaned up
            file_out.close()


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
    p = Pool(N)
    p.map(proc_cine_fname, zip(cine_fnames, itertools.repeat(hdf_fname_template)), chunksize=1)
