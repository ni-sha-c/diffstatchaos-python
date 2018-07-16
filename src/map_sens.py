'''
Statistical sensitivity analysis algorithm for maps.
'''
from __future__ import division
from __future__ import print_function
import sys
import pdb

from pylab import *
from numpy import *
from time import clock
from util import *



@jit(nopython=True)
def compute_objective(u,s0,n_steps,n_theta=25,n_phi=25):
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
    phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)
    J_theta_phi = zeros((n_steps,n_theta,n_phi))
    for i in arange(n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                J_theta_phi[i,t_ind,p_ind] += objective(u[i],
                        s0,theta0,dtheta,phi0,dphi)
    return J_theta_phi 

@jit(nopython=True)
def preprocess_objective(J_theta_phi):
    n_steps = J_theta_phi.shape[0]
    integral_J = zeros(J_theta_phi.shape)
    integral_J[-1] = copy(J_theta_phi[-1])
    for i in range(n_steps-1,0,-1):
        integral_J[i-1] = integral_J[i] + J_theta_phi[i-1]
    return integral_J


@jit(nopython=True)
def compute_gradient_objective(u,s0,n_steps,n_theta=25,n_phi=25):
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
    phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)

    DJ_theta_phi = zeros((n_steps,n_theta,n_phi,state_dim))
    for i in arange(n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                DJ_theta_phi[i,t_ind,p_ind] += Dobjective(u[i],
                        s0,theta0,dtheta,phi0,dphi)
    return DJ_theta_phi 


@jit(nopython=True)
def solve_primal(u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[0] = u_init
    for i in range(1,n_steps):
        u[i] = step(u[i-1],s)
    return u

@jit(nopython=True)
def solve_unstable_direction(u, v_init, n_steps, s):
    v = empty((n_steps, v_init.size))
    log_v_mag = empty((n_steps,))
    v[0] = v_init
    log_v_mag[0] = log(norm(v[0]))
    v[0] /= norm(v[0])
    for i in range(1,n_steps):
        v[i] = dot(gradFs(u[i-1],s),v[i-1])
        log_v_mag[i] = log_v_mag[i-1] + log(norm(v[i]))
        v[i] /= linalg.norm(v[i])
    return v, log_v_mag

@jit(nopython=True)
def solve_unstable_adjoint_direction(u, w_init, n_steps, s):
    w = empty((n_steps, w_init.size))
    log_w_mag = empty((n_steps,))
    w[-1] = w_init
    log_w_mag[-1] = log(norm(w[-1]))
    w[-1] /= norm(w[-1])
    for i in range(n_steps-1,0,-1):
        w[i-1] = adjoint_step(w[i],u[i-1],s,0)
        log_w_mag[i-1] = log_w_mag[i] + log(norm(w[i-1]))
        w[i-1] /= norm(w[i-1])
    return w, log_w_mag


@jit(nopython=True)
def compute_source_tangent(u, n_steps, s0):
    param_dim = s0.size
    dFds = zeros((n_steps,param_dim,state_dim))
    for i in range(n_steps):
        dFds[i] = DFDs(u[i],s0)
    return dFds

@jit(nopython=True)
def compute_source_forward_adjoint(u, n_steps, s0):
    dgf = zeros((n_steps,state_dim))
    for i in range(n_steps):
        dgf[i] = divGradFsinv(u[i],s0)
    return dgf

@jit(nopython=True)
def compute_source_sensitivity(u, n_steps, s0):
    param_dim = s0.size
    t_ddFds_dFinv = zeros((n_steps,param_dim))
    for i in range(n_steps):
        t_ddFds_dFinv[i] = trace_gradDFDs_gradFsinv(u[i],s0)
    return t_ddFds_dFinv

@jit(nopython=True)
def compute_forward_adjoint(u,v0,w0,source_forward_adjoint):
    n_steps = u.shape[1]
    assert(v0.shape==w0.shape==source_forward_adjoint.shape==u.shape)
    
@jit(nopython=True)
def compute_sensitivity(u,s,v0,w0,J,dJ,dFds,dJds_0,source_forward_adjoint,N,n_runup_forward_adjoint):

    t0 = clock()
    u = solve_primal(u_init, n_steps, s0)
    t1 = clock()
    v0 = solve_poincare_unstable_direction(u, v0_init, n_steps, s0)
    t2 = clock()


    g = zeros(n_runup_forward_adjoint)
    w_inv = zeros((n_steps,state_dim))
    v = zeros(state_dim)
    n_points_theta = J.shape[1]
    n_points_phi = J.shape[2]
    gsum_mean = 0.0
    gsum_history = zeros(n_steps)
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dFds_unstable = zeros(state_dim)
    n_samples = N - 2*n_runup_forward_adjoint
    for n in range(N-1):
        b = dJds_0[n]
        q = source_forward_adjoint[n]
        nablaFs = gradFs_poincare(u[n],s)   
        w_inv[n+1] = -1.0*q + solve(nablaFs.T, w_inv[n])
        w_inv[n+1] = dot(w_inv[n+1], v0[n+1]) * v0[n+1]
        g[n % n_runup_forward_adjoint] = dot(w_inv[n+1],dFds_unstable) - b   
        v = dot(nablaFs,v) + dFds[n]
        gsum_history[n] = sum(g)
        gsum_mean += sum(g)

        v, dFds_unstable = decompose_tangent(v, v0[n+1], w0[n+1])
        if(n>=2*n_runup_forward_adjoint):
            for binno_t in range(n_points_theta):
                for binno_p in range(n_points_phi):
                    dJds_unstable[binno_t,binno_p] += \
                            J[n+1,binno_t,binno_p]*\
                            (sum(g)-gsum_mean/n)/n_samples
                    dJds_stable[binno_t,binno_p] += \
                            dot(dJ[n+1,binno_t,binno_p], v)/n_samples
    return dJds_stable, dJds_unstable, w_inv, gsum_history


