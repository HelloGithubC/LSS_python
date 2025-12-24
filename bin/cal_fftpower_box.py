#!python -u

import numpy as np 
from LSS_python.mesh import Mesh 
from LSS_python.fftpower import FFTPower
import joblib, os
import argparse, sys

def read_data(filename, data_type, weight_key=None, weight_set_one=False, delimiter=None):
    if weight_key is None:
        if weight_set_one:
            with_weight = True
            result_cols = 4 
        else:
            with_weight = False
            result_cols = 3 
    else:
        with_weight = True 
        result_cols = 4
    
    if data_type == "ascii":
        source = np.loadtxt(filename, delimiter=delimiter)
        if source.shape[1] < result_cols:
            raise ValueError(f"The cols of {filename} is not enough")
        result_array = np.zeros(shape=(source.shape[0], result_cols), dtype=source.dtype) 
        if weight_set_one:
            result_array[:, :3] = source[:, :3]
            result_array[:,3] = 1.0
        else:
            result_array[:,:result_cols] = source[:, :result_cols]
    elif data_type == "binary":
        source = np.load(filename)
        if source.dtype.names is None:
            result_array = np.zeros(shape=(source.shape[0], result_cols), dtype=source.dtype)
            if weight_set_one:
                result_array[:,:3] = source[:, :3]
                result_array[:,3] = 1.0
            else:
                result_array[:,:result_cols] = source[:, :result_cols]
        else:
            result_array = np.empty(shape=(source.shape[0], result_cols), dtype=source["X"].dtype)
            result_array[:,0] = source["X"]
            result_array[:,1] = source["Y"]
            result_array[:,2] = source["Z"]
             
            if with_weight:
                if weight_set_one:
                    result_array[:,3] = 1.0
                else:
                    result_array[:,3] = source[weight_key]
    else:
        raise ValueError(f"{data_type} is not supported")
    
    return result_array
        
    
parser = argparse.ArgumentParser(
    prog="cal_tpCF_box", usage="To cal tpCF with Corrfunc for simulation box"
)

parser.add_argument("data_filename", help="Need suffix to support the automatic naming rules. Or use --output_filename and --rr_filename to specify the relevant files.", type=str)
parser.add_argument("random_filename", help="Need suffix to ... (the same as data_filename)", type=str)
parser.add_argument("boxsize", type=float, help="If <=0, will be calculated from data or random (only when --only-run-rr set)")
parser.add_argument("nthreads", type=int)

parser.add_argument("--data_type", "-type", type=str, default="binary", help="Only support ascii and binary. Only support npy format in binary, whether is structed array or not.")
parser.add_argument("--weight_key", "-wei", type=str, default="weight", help="Only support npy format in binary with structed array.")

parser.add_argument("-smin", type=float, default=0.0, help="Default 0.0")
parser.add_argument("-smax", type=float, default=150.0, help="Default 150.0")
parser.add_argument("-sbin", type=int, default=150, help="Default 150")
parser.add_argument("-mubin", type=int, default=120, help="Default 120")

parser.add_argument("--with_weight", "-ww", action="store_true", help="If set, will load the weight from file or set to one with -weightone")
parser.add_argument("--with_random_weight", "-wrw", action="store_true", help="If set, will use random weight and the weight key is the same as data")
parser.add_argument("--weight_power", "-weipow", type=str, help="Only be valid when -ww set.", default="1")
parser.add_argument("--set_weight_to_one", "-weightone", action="store_true", help="Only be valid when --with_weight set. The weight will be forced to be 1.")
parser.add_argument("--xyz_refine_factors", "-refine", nargs=3, type=int, default=[2, 2, 1])

parser.add_argument("--output_filename", "-o", help="Only support specifying the filename of tpCF or RR (when --only_run_rr set). The dir is needed.", default=None)
parser.add_argument("--rr_filename", "-rr", help="Force to specify the rr file.", default=None)

parser.add_argument("--only_run_rr", "-only_rr", action="store_true")
parser.add_argument("--save_dd_dr", "-savetemp", help="If set save DD and DR to re-use.", action="store_true")
parser.add_argument("-force", "-f", action="store_true", help="Force to run tpCF, DD and DR, even if the result exists.")
parser.add_argument("--force_rr", "-forcerr", action="store_true", help="Force to run RR, even if the result exists.")

parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")

if len(sys.argv) == 1:
    parser.print_help()
    exit(0)

argv = parser.parse_args()

data_fileallname = argv.data_filename
random_fileallname = argv.random_filename
boxsize = argv.boxsize

nthreads = argv.nthreads

data_type = argv.data_type
weight_key = argv.weight_key

smin = argv.smin
smax = argv.smax
sbin = argv.sbin
mubin = argv.mubin
sedges = np.linspace(smin, smax, sbin+1)

with_weight = argv.with_weight
with_random_weight = argv.with_random_weight
weight_power = argv.weight_power
weight_set_one = argv.set_weight_to_one
xyz_refine_factors = argv.xyz_refine_factors

output_filename = argv.output_filename
force_rr_filename = argv.rr_filename

only_run_rr = argv.only_run_rr
save_dd_dr = argv.save_dd_dr 
force = argv.force
force_rr = argv.force_rr

verbose = argv.verbose