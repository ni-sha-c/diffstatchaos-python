#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys
import pdb

sys.path.insert(0, '../examples/')
from kuznetsov import *

sys.path.insert(0, '../examples/tests')
from test_kuznetsov import *
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
def compute_source_tangent(u, n_steps, s0):
    param_dim = s0.size
    dfds = zeros((n_steps,param_dim,state_dim))
    for i in range(n_steps):
        dfds[i] = DfDs(u[i],s0)
    return dfds

@jit(nopython=True)
def compute_source_inverse_adjoint(u, n_steps, s0):
    dgf = zeros((n_steps,state_dim))
    for i in range(n_steps):
        dgf[i] = divGradfs(u[i],s0)
    return dgf


@jit(nopython=True)
def compute_source_sensitivity(u, n_steps, s0):
    param_dim = s0.size
    ddfds = zeros((n_steps,param_dim))
    for i in range(n_steps):
        ddfds[i] = divDfDs(u[i],s0)
    return ddfds


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
    theta_bin_centers = linspace(0,pi,n_theta)
    phi_bin_centers = linspace(-pi,pi,n_phi)
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    DJ_theta_phi = zeros((n_steps,n_theta,n_phi,state_dim))
    for i in arange(1,n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                DJ_theta_phi[i,t_ind,p_ind] += Dobjective(u[i-1],
                        s0,theta0,dtheta,phi0,dphi)
    return DJ_theta_phi 

@jit(nopython=True)    
def compute_finite_difference_sensitivity(n_samples,s0,n_points_theta, \
        n_points_phi):

    um = (rand(state_dim)*(boundaries[1]-boundaries[0]) + 
                      boundaries[0])
    up = copy(um)
    sm = copy(s0)
    sp = copy(s0)
    epsi = 1.e-4
    sm[0] -= epsi
    sp[0] += epsi
    up = primal_step(up,sp,n_runup)
    um = primal_step(um,sm,n_runup)
    epsi = 1.e-4
    J_sum_m = zeros((n_points_theta,n_points_phi))
    J_sum_p = zeros((n_points_theta,n_points_phi))
    dJds_fd = zeros((n_points_theta,n_points_phi))
    n_steps = 100000
     
    for i in arange(n_samples):
        um_traj = solve_primal(um,n_steps,sm)
        up_traj = solve_primal(up,n_steps,sp)
        
        J_sum_m += (compute_objective(um_traj,sm,n_steps,\
                n_points_theta,\
                n_points_phi)).sum(0)/n_samples
        
        J_sum_p += (compute_objective(up_traj,sp,n_steps,\
                n_points_theta,\
                n_points_phi)).sum(0)/n_samples
        dJds_fd += (J_sum_p-J_sum_m) \
                                / (2.0*epsi)/n_samples
        um = copy(um_traj[-1])
        up = copy(up_traj[-1])
    return dJds_fd

@jit(nopython=True)
def compute_correlation_Jg(cumsum_J, \
        ddfds,n_samples):
    temp_array = zeros(cumsum_J.shape)
    for i in range(n_samples):
        temp_array[i] = cumsum_J[i]*ddfds[i]
    return temp_array[:n_samples].sum(0)


@jit(nopython=True)
def compute_sensitivity(u,s0,v0,w0,dJ,ds,dfds,cumsumJ,N,Ninf):
    N_padded = u.shape[0]
    v = zeros(state_dim)
    dJ0 = zeros(state_dim)
    w_inv = zeros(state_dim)
    dJds_stable = zeros(cumsumJ.shape[1:])
    dJds_unstable = zeros(cumsumJ.shape[1:])
    ntheta = DJ_theta_phi.shape[1]
    nphi = DJ_theta_phi.shape[2]
    for i in range(N_padded-1):
        v = tangent_step(v,u[i],s0,ds) 
        v,_= decompose_tangent(v,v0[i+1],w0[i+1])
        w_inv = adjoint_step(w_inv,u[i],s0,dJ0) - source_inverse_adjoint[i]*dt
        w_inv,_= decompose_adjoint(w_inv,v0[i+1],w0[i+1]) 
        if(i < N):
            for t1 in range(ntheta):
                for p1 in range(nphi):
                    dJds_stable += dot(DJ_theta_phi[i+1,t1,p1],v)/N
            dJds_unstable -= cumsumJ[i]*( \
                            dot(dfds[i+1],w_inv))/N
    return dJds_stable,dJds_unstable 


if __name__ == '__main__':
    n_samples = int(T / dt) * 1000
    n_runup = int(T / dt) * 100
    n_converge = int(T / dt)* 10
    n_steps = n_samples + n_converge
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
    DJ_theta_phi = zeros((n_steps,n_points_theta,n_points_phi,state_dim))  
    theta_bin_centers = linspace(dtheta/2.0,pi - dtheta/2.0, n_points_theta)
    phi_bin_centers = linspace(-pi-dphi/2.0,pi - dphi/2.0, n_points_phi)
    v = zeros(state_dim)
    w_inv = zeros(state_dim)
    dfds = zeros((n_steps,param_dim,state_dim))
    source_inverse_adjoint = zeros((n_steps,state_dim))
    source_tangent = zeros((n_steps,param_dim,state_dim))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    divdfds = zeros(n_steps)

    t0 = clock()
    u = solve_primal(u_init, n_steps, s0)
    t1 = clock()
    v0 = solve_unstable_direction(u, v0_init, n_steps, s0, ds0)
    t2 = clock()
    source_tangent = compute_source_tangent(u,n_steps,s0)[:,0,:] 
    t3 = clock()
    w0 = solve_unstable_adjoint_direction(u, w0_init, n_steps, s0, dJ0)
    t4 = clock()
    #J_theta_phi = compute_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t5 = clock()
    #DJ_theta_phi = compute_gradient_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t6 = clock()
    source_inverse_adjoint = compute_source_inverse_adjoint(u,n_steps,s0)
    t7 = clock()
    t8 = clock()
    divdfds = (compute_source_sensitivity(u,n_steps,s0))[:,0]
    t9 = clock()

    print('='*50)
    print("Pre-computation times for {:>10d} steps".format(n_samples))
    print('='*50)
    print('{:<35s}{:>16.10f}'.format("primal", t1-t0))
    print('{:<35s}{:>16.10f}'.format("tangent",t2 - t1)) 
    print('{:<35s}{:>16.10f}'.format("tangent source", t3 - t2))
    print('{:<35s}{:>16.10f}'.format("adjoint ", t4 - t3))
    print('{:<35s}{:>16.10f}'.format("inverse adjoint source", t7 - t6))
    print('{:<35s}{:>16.10f}'.format("objective ", t5 - t4)) 
    print('{:<35s}{:>16.10f}'.format("gradient objective ", t6 - t5))
    print('{:<35s}{:>16.10f}'.format("divergence tangent source ", t9 - t8))
    print('*'*50)
    print("End of pre-computation")
    print('*'*50)

    stop
    
    print('='*50)
    print("Computation times for sensitivity source terms")
    print('='*50)
    t10 = clock()
    cumsum_J_theta_phi = preprocess_objective(J_theta_phi) 
    t11 = clock()
    correlation_J_divDfDs = compute_correlation_Jg(cumsum_J_theta_phi, \
            divdfds,n_samples)
    t12 = clock()
    print('{:<35s}{:>16.10f}'.format("Preprocessing objective", t11-t10))
    print('{:<35s}{:>16.10f}'.format("objective source correlation", t12-t11))
    print('*'*50)
    print('End of sensitivity source term computation')
    print('*'*50)

    
    ds1 = copy(ds0)
    ds1[0] = 1.0
    print('Starting stable-(adjoint-unstable) split evolution...')
    t13 = clock()
    dJds_stable, dJds_unstable = compute_sensitivity(u,s0,v0,w0,DJ_theta_phi,\
            ds1,source_tangent,cumsum_J_theta_phi,n_samples)
    t14 = clock()
    print('{:<35s}{:>16.10f}'.format("time taken",t14-t13))
    print('End of computation...')
    dJds_unstable -= correlation_J_divDfDs

    dJds = dJds_stable + dJds_unstable
    theta = linspace(0,pi,n_points_theta)
    phi = linspace(-pi,pi,n_points_phi)
    contourf(phi,theta,dJds,100)
    xlabel(r"$\phi$")
    ylabel(r"$\theta$")
    colorbar()
    savefig("../examples/plots/plykin_main1")      
