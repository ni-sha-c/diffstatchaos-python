import sys
sys.path.insert(0, '../')
from KS import *
from matplotlib.pyplot import *
from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

solver = Solver()
u0 = solver.u_init
s0 = solver.s0
def plot_solution():
    fig = figure()
    ax = fig.add_subplot(111)
    L = solver.L
    dt = solver.dt
    state_dim = solver.state_dim
    x = linspace(0.,L,state_dim+2)
    n_steps = 5000
    t = linspace(0, n_steps, n_steps)
    u_trj = solve_primal(solver, u0, s0, n_steps)
    u_trj_padded = hstack([zeros((n_steps,1)),\
        u_trj, zeros((n_steps,1))])
    im = ax.contourf(x, t*dt, u_trj_padded)
    return fig, ax

#def compute_ensemble_average():
if __name__=="__main__":
    n_samples = 10
    n_steps = 2000
    n_runup = 1000
    n_c = 25
    u_mean = empty(n_c) 
    c = linspace(0.,2.,n_c)
    for i, c_i in enumerate(c):
        for k in range(n_samples):
            mean_noise_init = rand()
            u_init = u0 + 0.1*mean_noise_init
            u_init = solver.primal_step(u_init,\
                    c_i, n_runup)
            u_trj_k = solve_primal(solver, \
                u_init, c_i, n_steps)
            u_mean[i] += mean(u_trj_k)/n_samples
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(c, u_mean, 'o-')
    
    

