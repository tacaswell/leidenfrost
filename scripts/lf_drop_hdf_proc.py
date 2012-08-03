import argparse
import os
import lf_drop.infra as li

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

stack = li.ProcessStack.from_hdf_file(cine_base_path,hdf_fname)
try:
    stack.get_frame(len(stack))
finally:
    del stack
