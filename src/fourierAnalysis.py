#Statistical sensitivity analysis algorithm for maps.
from __future__ import division
from __future__ import print_function
import sys
import pdb
from pylab import *
from numpy import *
from time import clock
from util import *

from numba import jitclass
from numba import int64, float64
spec = []

@jitclass(spec)
class FourierAnalysis:
    def __init__(self):
        pass

    def solve_primal(self, solver_map, u_init, n_steps, s):
        u = empty((n_steps, u_init.size))
        u[0] = u_init
        for i in range(1,n_steps):
            u[i] = solver_map.primal_step(u[i-1],s,1)
        return u
    
    
    def solve_unstable_direction(self, solver_map,\
            u, v_init, n_steps, s):
        v = empty((n_steps, v_init.size))
        v[0] = v_init
        v[0] /= norm(v[0])
        for i in range(1,n_steps):
            v[i] = dot(solver_map.gradFs(u[i-1],s),v[i-1])
            v[i] /= linalg.norm(v[i])
        return v
    
    
    def solve_unstable_adjoint_direction(self, solver_map,\
            u, w_init, n_steps, s):
        w = empty((n_steps, w_init.size))
        w[-1] = w_init
        w[-1] /= norm(w[-1])
        for i in range(n_steps-1,0,-1):
            w[i-1] = solver_map.adjoint_step(w[i],u[i-1],s,0)
            w[i-1] /= norm(w[i-1])
        return w
    
    
    
    def compute_source_tangent(self, solver_map, \
            u, n_steps, s0):
        param_dim = s0.size
        dFds = zeros((n_steps,param_dim,u.shape[1]))
        for i in range(n_steps):
            dFds[i] = solver_map.DFDs(u[i],s0)
        return dFds




    def compute_fourier_transform(self,u,J,xi):
        pass


    def compute_correlation(self,f,g,n):
        for i in range(f.shape[0]):
            


