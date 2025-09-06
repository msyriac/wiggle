import os, sys
from time import time
try:
    nthreads = int(os.environ["OMP_NUM_THREADS"])
except:
    import multiprocessing
    nthreads = multiprocessing.cpu_count()
import numpy as np
import argparse
from pywiggle.utils import Benchmark
parser = argparse.ArgumentParser(
    description="Benchmark wiggle, pspy, ducc or Namaster",
    epilog="""
    
    Example:
    bench.py wiggle --lmax 512 --spin 0 --binned
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("code", help="One of wiggle, ducc, pspy, nmt.")
parser.add_argument("--lmax", type=int, default=512, help="Maximum multipole")
parser.add_argument("--repeat", type=int, default=5, help="Repetitions.")
parser.add_argument("--spin", type=int, default=0, help="Spin 0 or 2")
parser.add_argument("--skipinit", action="store_true", help="Whether to skip single initial call.")
parser.add_argument("--binned", action="store_true", help="Whether to bin.")

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
args = parser.parse_args()
    
bin_edges = None
if args.binned:
    bin_edges = np.arange(20,args.lmax,20)
b = Benchmark(lmax=args.lmax,nthreads=nthreads)
if not(args.skipinit):
    b.get_mcm(args.code,spin=args.spin,bin_edges = bin_edges)

N = args.repeat
start = time()
for i in range(N):
    b.get_mcm(args.code,spin=args.spin,bin_edges = bin_edges)
end = time()

print((end-start)/N)
