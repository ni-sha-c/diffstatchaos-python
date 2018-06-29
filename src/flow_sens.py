#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys
import pdb

sys.path.insert(0, '../examples/')
from kuznetsov import *

from pylab import *
from numpy import *
from time import clock
from util import *

@jit(nopython=True)
def solve_primal(u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[0] = u_init
    for i in range(1,n_steps):
        u[i] = primal_step(u[i-1],s)
    return u

@jit(nopython=True)
def solve_unstable_direction(u, v_init, n_steps, s, ds):
    v = empty((n_steps, v_init.size))
    v[0] = v_init
    for i in range(1,n_steps):
        v[i] = tangent_step(v[i-1],u[i-1],s,ds)
        v[i] /= linalg.norm(v[i])
    return v

@jit(nopython=True)
def solve_unstable_adjoint_direction(u, w_init, n_steps, s, dJ):
    w = empty((n_steps, w_init.size))
    w[-1] = w_init
    for i in range(n_steps-1,0,-1):
        w[i-1] = adjoint_step(w[i],u[i-1],s,dJ)
        w[i-1] /= norm(w[i-1])
    return w

@jit(nopython=True)
def solve_dfds(u, s0, n_steps):
    param_dim = s0.size
    dfds = zeros((n_steps,param_dim,state_dim))
    for i in range(n_steps):
        dfds[i] = DfDs(u[i],s0)
    return dfds

@jit(nopython=True)
def compute_objective(u,s0,n_steps,n_theta=25,n_phi=25):
    theta_bin_centers = linspace(0,pi,n_theta)
    phi_bin_centers = linspace(-pi,pi,n_phi)
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    J_theta_phi = zeros((n_steps,n_theta,n_phi))
    for i in arange(1,n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                J_theta_phi[i,t_ind,p_ind] += objective(u[i-1],
                        s0,theta0,dtheta,phi0,dphi)/n_steps
    return J_theta_phi 


if __name__ == '__main__':
    n_steps = int(T / dt) * 1000
    n_runup = int(T / dt) * 100
    n_points_theta = 20
    n_points_phi = 20
    dtheta = pi/(n_points_theta-1)
    dphi = 2*pi/(n_points_phi-1)
    u = zeros((n_steps,state_dim))
    random.seed(0)
    u_init = rand(state_dim)
    u_init[3] = 0
    u_init = primal_step(u_init,s0,n_runup)
    param_dim = s0.size
    ds0 = zeros(param_dim)
    dJ0 = zeros(state_dim)
    
    w0 = zeros((n_steps,state_dim))
    w0_init = rand(state_dim)
    w0_init /= linalg.norm(w0_init)
    
    v0_init = rand(state_dim)
    v0_init /= linalg.norm(v0_init)
    
    J_theta_phi = zeros((n_steps,n_points_theta,n_points_phi))
    theta_bin_centers = linspace(dtheta/2.0,pi - dtheta/2.0, n_points_theta)
    phi_bin_centers = linspace(-pi-dphi/2.0,pi - dphi/2.0, n_points_phi)
    v = zeros(state_dim)
    w_inv = zeros(state_dim)
    dfds = zeros((n_steps,param_dim,state_dim))
    divdfds = zeros((n_steps,param_dim))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    
    t0 = clock()
    u = solve_primal(u_init, n_steps, s0)
    t1 = clock()
    v0 = solve_unstable_direction(u, v0_init, n_steps, s0, ds0)
    t2 = clock()
    dfds = solve_dfds(u,s0,n_steps) 
    t3 = clock()
    w0 = solve_unstable_adjoint_direction(u, w0_init, n_steps, s0, dJ0)
    t4 = clock()
    J_theta_phi = compute_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t5 = clock()
    print(n_steps, t1 - t0, t2 - t1, t3 - t2, t4 - t3,t5 - t4)
    
    
    
     
    '''
    for i in arange(1,n_steps):
        divdfds[:,i] = divDfDs(u[:,i-1],s0)
    
    for i in arange(1,n_steps-1):
        source_tangent = DfDs(u[:,i],s0)[:,0]
        source_adjoint = divGradfs(u[:,i],s0)
        v = tangent_step(v,u[:,i],s0,ds0) + source_tangent*dt
        v,_= decompose_tangent(v,v0[:,i+1],w0[:,i+1])
        w_inv = adjoint_step(w_inv,u[:,i],s0,dJ0) + source_adjoint*dt
        w_inv,_= decompose_adjoint(w_inv,v0[:,i+1],w0[:,i+1])
        for i1 in arange(n_points_theta):
            for j1 in arange(n_points_phi):
                theta0 = theta_bin_centers[i1]
                phi0 = theta_bin_centers[j1]
                dJ_theta_phi = Dobjective(u[:,i+1],s0,theta0,dtheta,
                                    phi0,dphi)
                dJds_stable[i1,j1] += dot(dJ_theta_phi,v)/n_steps
                dJds_unstable[i1,j1] -= J_theta_phi[i1,j1]*(divdfds[0,i+1] +
                        dot(dfds[:,0,i+1],w_inv))
    
    
    n_samples = 100000
    um = rand(state_dim)*2.0 - 1.0
    up = rand(state_dim)*2.0 - 1.0
    sm = copy(s0)
    sp = copy(s0)
    sm[0] -= epsi
    sp[0] += epsi
    up = Step(up,sp,n_runup)
    um = Step(um,sm,n_runup)
    epsi = 1.e-4
    J_sum_m = zeros((n_points_theta,n_points_phi))
    J_sum_p = zeros((n_points_theta,n_points_phi))
    dJds_fd = zeros((n_points_theta,n_points_phi))
    for i in arange(n_samples):
        um = Step(um,sm,1)
        up = Step(up,sp,1)
        for i1 in arange(n_points_theta):
            for i2 in arange(n_points_phi):
                theta0 = theta_bin_centers[i1]
                phi0 = phi_bin_centers[i2]
                J_sum_p[i1,i2] += objective(um,s0,theta0,dtheta,phi0,dphi)
                J_sum_m[i1,i2] += objective(um,s0,theta0,dtheta,phi0,dphi)
                dJds_fd[i1,i2] += (J_sum_p[i1,i2]-J_sum_m[i1,i2]) \
                                / (2.0*epsi)/n_samples
    '''
