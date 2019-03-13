import sys
sys.path.insert(0, '../examples/')
import kuznetsov_poincare as kp
from objective import *
import scipy
from scipy.interpolate import griddata
import map_sens as map_sens
from time import clock
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
@jit(nopython=True)
def solve_tangent_equation(solver_map,\
            u, v_init, dFds, n_steps, s):
    v = empty((n_steps, v_init.size))
    v[0] = v_init
    for i in range(1,n_steps):
        v[i] = dot(solver_map.gradFs(u[i-1],s),v[i-1]) + \
                dFds[i-1]
    return v

@jit(nopython=True)
def gradient_J(u, index):
    gradJ = zeros_like(u)
    gradJ[:,index] = 1.0
    return gradJ


@jit(nopython=True)
def compute_dJds(gradJ, v):
    dJds = diag(dot(gradJ, v.T))
    return dJds

#def compute_sensitivity():
if __name__ == "__main__":
    solver = kp.Solver()
    n_steps = 50
    s3_map = map_sens.Sensitivity(solver,n_steps)
    n_runup = 500
    u_init = solver.u_init
    s0 = solver.s0
    state_dim = solver.state_dim
    param_dim = s0.size
    u_init = solver.primal_step(u_init, s0, n_runup)
    u_trj = s3_map.solve_primal(solver,\
            solver.u_init, n_steps, solver.s0)
    dFds = s3_map.compute_source_tangent(solver,\
            u_trj, n_steps, s0)[:,0,:]
    v0_init = zeros_like(u_init)
    v_trj = solve_tangent_equation(solver, u_trj, v0_init, dFds, n_steps, s0)
    dJ_trj = gradient_J(u_trj, 0)
    dJds = compute_dJds(dJ_trj, v_trj)
    fig = figure(figsize=[15,10])
    ax = fig.add_subplot(111)
    ax.semilogy(abs(dJds),linewidth=3,color='r')
    ax.set_xlabel("i", fontsize=24)
    ax.set_ylabel("| $dJ/ds$ |", fontsize=24)
    ax.set_title("Sensitivity of x in Plykin system wrt s",fontsize=24)
    ax.tick_params(axis='both',labelsize=24)
    ax.grid(True)
    savefig('../examples/plots/plykin_tangent_blowup.png')


