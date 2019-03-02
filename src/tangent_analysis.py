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
def objective(u):
    return u[0]

@jit(nopython=True)
def compute_dJds(gradJ, v):
    dJds = diag(dot(gradJ, v.T))
    return dJds

#@jit(nopython=True)
def projection(v, u):
    for i, ui in enumerate(u):
        print(i)
        ui /= norm(ui)
        v[i] -= dot(outer(ui,ui),v[i])
    return v



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
    J_trj = objective(u_trj.T)

