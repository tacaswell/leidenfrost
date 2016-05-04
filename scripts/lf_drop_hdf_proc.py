from __future__ import print_function
import argparse

import leidenfrost.infra as li
from leidenfrost import FilePath
import h5py

import find_peaks.peakdetect as pd

pd.init_procs(8)

parser = argparse.ArgumentParser()
parser.add_argument("cine_base_path", help="The base path for where the (cine) data files are located")
parser.add_argument("hdf_path", help="path, relative to base path of hdf file")
parser.add_argument("hdf_fname", help="name of the hdf file")
parser.add_argument("--hdf_base_path", help="The base path for where the hdf data file are located.  If not given, assumed to be the same as cine_base_path")

args = parser.parse_args()

cine_base_path = args.cine_base_path
if args.hdf_base_path:
    hdf_base_path = args.hdf_base_path
else:
    hdf_base_path = cine_base_path

hdf_fname = FilePath(hdf_base_path, args.hdf_path, args.hdf_fname)
print(hdf_fname.format)


stack, seed_curve = li.ProcessBackend.from_hdf_file(cine_base_path, hdf_fname)
file_out = h5py.File(hdf_fname.format, 'r+')

for j in range(len(stack)):
    mbe, seed_curve = stack.process_frame(j, seed_curve)
    mbe.write_to_hdf(file_out)
    del mbe
    file_out.flush()


file_out.close()

pd.kill_procs()
