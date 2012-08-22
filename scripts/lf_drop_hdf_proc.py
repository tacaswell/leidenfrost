import argparse
import os
import lf_drop.infra as li
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("cine_base_path",help="The base path for where the (cine) data files are located")
parser.add_argument("hdf_path",help="path, relative to base path of hdf file")
parser.add_argument("hdf_fname",help="name of the hdf file")
parser.add_argument("--hdf_base_path",help="The base path for where the hdf data file are located.  If not given, assumed to be the same as cine_base_path")

args = parser.parse_args()

cine_base_path = args.cine_base_path
if args.hdf_base_path:
    hdf_base_path = args.hdf_base_path
else:
    hdf_base_path = cine_base_path

hdf_fname = li.FilePath(hdf_base_path,args.hdf_path,args.hdf_fname)

stack,seed_curve = li.ProcessBackend.from_hdf_file(cine_base_path,hdf_fname)
file_out = h5py.File('/'.join(hdf_fname),'r+')

for j in range(len(stack)):
    mbe,seed_curve = stack.process_frame(j,seed_curve)
    mbe.write_to_hdf(file_out)
    del mbe
    file_out.flush()


file_out.close()
