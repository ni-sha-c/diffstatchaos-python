#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys

sys.path.insert(0, '../examples/')
from kuznetsov import *

from pylab import *
from numpy import *
from mpi4py import MPI
from time import time
from util import *

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("test")



n_steps = 100000
n_runup = 50000
u = rand(d)
u = Step(u,s0,n_runup)
ds = zeros(p)
dJ = zeros(d)
v0 = rand(d)
w0 = rand(d)
v0 /= norm(v0)
w0 /= norm(w0)
ds1 = copy(ds)
ds1[0] = 1.0
v = zeros(d)
for i in arange(n_steps):
    v0 = tangent_step(v0,u,s0,ds)
    w0 = adjoint_step(w0,u,s0,dJ)
    v0 /= norm(v0)
    w0 /= norm(w0)
   
    dfds = DfDs(u,s0)[:,0]
    dfds,_= decompose_tangent(dfds,v0,w0)
    v = tangent_step(v,u,s0,dfds)
    u = Step(u,s0,1) 


