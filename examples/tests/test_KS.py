import sys
sys.path.insert(0, '../')
from KS import *
from matplotlib.pyplot import *
from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

solver = Solver()
fig = figure()
ax = fig.add_subplot(111)
L = solver.L
state_dim = solver.state_dim
x = linspace(0.,L,state_dim+2)
u0 = solver.u_init
s0 = solver.s0
n_steps = 5000
t = linspace(0, n_steps, n_steps)
u_trj = solve_primal(solver, u0, s0, n_steps)
u_trj_padded = hstack([zeros((n_steps,1)),\
        u_trj, zeros((n_steps,1))])
im = ax.contourf(x, t, u_trj_padded)

