#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from pylab import *
from numpy import *
from mpi4py import MPI
from time import time


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("test")
