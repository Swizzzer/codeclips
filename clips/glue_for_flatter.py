from subprocess import check_output
from re import findall
from sage.all import matrix
import os
def flatter(M):
    # compile https://github.com/keeganryan/flatter and put it in $PATH
    z = "[[" + "]\n[".join(" ".join(map(str, row)) for row in M) + "]]"
    env = os.environ.copy()
    # change this on MacBook to avoid processing on E-Cores
    # 4~8 is better
    env['OMP_NUM_THREADS'] = '8'
    ret = check_output(["flatter"], input=z.encode(), env=env)
    return matrix(M.nrows(), M.ncols(), map(int, findall(rb"-?\d+", ret)))
