#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys
import pdb

sys.path.insert(0, '../examples/')
from kuznetsov import *

from pylab import *
from numpy import *
from time import time
from util import *

n_steps = int(T / dt) * 1000
n_runup = int(T / dt) * 100
n_bins_theta = 20
n_bins_phi = 20
dtheta = pi/n_bins_theta
dphi = 2*pi/n_bins_phi
u = zeros((state_dim,n_steps))
random.seed(0)
u_init = rand(state_dim)
u_init[3] = 0
u_init = primal_step(u_init,s0,n_runup)

ds0 = zeros(s0.size)
dJ0 = zeros(state_dim)

w0 = zeros((state_dim,n_steps))
w0[:,n_steps-1] = rand(state_dim)
w0[:,n_steps-1] /= norm(w0[:,n_steps-1])

v0_init = rand(state_dim)
v0_init /= linalg.norm(v0_init)

J_theta_phi = zeros((n_bins_theta,n_bins_phi))
theta_bin_centers = linspace(dtheta/2.0,pi - dtheta/2.0, n_bins_theta)
phi_bin_centers = linspace(-pi-dphi/2.0,pi - dphi/2.0, n_bins_phi)
param_dim = s0.size
v = zeros(state_dim)
w_inv = zeros(state_dim)
dfds = zeros((state_dim,param_dim,n_steps))
divdfds = zeros((param_dim,n_steps))
dJds_stable = zeros((n_bins_theta,n_bins_phi))
dJds_unstable = zeros((n_bins_theta,n_bins_phi))

@jit(nopython=True)
def solve_primal(u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[0] = u_init
    for i in arange(1,n_steps):
        u[i] = primal_step(u[i-1],s)
    return u

@jit(nopython=True)
def solve_tangent(u, v_init, n_steps, s, ds):
    v = empty((n_steps, v_init.size))
    v[0] = v_init
    for i in arange(1,n_steps):
        v[i] = tangent_step(v[i-1],u[i-1],s,ds)
        v[i] /= linalg.norm(v[i])
    return v

t0 = time()
u = solve_primal(u_init, n_steps, s0)
t1 = time()
v0 = solve_tangent(u, v0_init, n_steps, s0, ds0)
t2 = time()

print(n_steps, t1 - t0, t2 - t1)

def visualize_primal(u):
    stero_real, stero_imag = stereographic_projection(u.T)
    plot(stero_real, stero_imag, '.', ms=1)

def extrapolate(a0, a1, multiplier):
    return a0 + (a1 - a0) * multiplier

def visualize_tangent(u, v):
    EPS = 1E-8
    u_plus, u_minus = u + v * EPS, u - v * EPS
    stero_real, stero_imag = stereographic_projection(u.T)
    stero_real_plus, stero_imag_plus = stereographic_projection(u_plus.T)
    stero_real_minus, stero_imag_minus = stereographic_projection(u_minus.T)
    stero_real_plus = extrapolate(stero_real, stero_real_plus, 1E6)
    stero_real_minus = extrapolate(stero_real, stero_real_minus, 1E6)
    stero_imag_plus = extrapolate(stero_imag, stero_imag_plus, 1E6)
    stero_imag_minus = extrapolate(stero_imag, stero_imag_minus, 1E6)
    plot([stero_real_plus, stero_real_minus],
         [stero_imag_plus, stero_imag_minus], '-k', ms=1)

visualize_tangent(u[::int(T/dt)], v0[::int(T/dt)])


'''
for i in arange(1,n_steps):
    dfds[:,:,i] = DfDs(u[:,i-1],s0)

for i in arange(1,n_steps):
    for t_ind, theta0 in enumerate(theta_bin_centers):
        for p_ind, phi0 in enumerate(phi_bin_centers):
            J_theta_phi[t_ind,p_ind] += objective(u[:,i-1],
                    s0,theta0,dtheta,phi0,dphi)/n_steps

for i in arange(n_steps-1,0,-1):
    w0[:,i-1] = adjoint_step(w0[:,i],u[:,i-1],s0,dJ0)
    w0[:,i-1] /= norm(w0[:,i-1])

for i in arange(1,n_steps):
    divdfds[:,i] = divDfDs(u[:,i-1],s0)

for i in arange(1,n_steps-1):
    source_tangent = DfDs(u[:,i],s0)[:,0]
    source_adjoint = divGradfs(u[:,i],s0)
    v = tangent_step(v,u[:,i],s0,ds0) + source_tangent*dt
    v,_= decompose_tangent(v,v0[:,i+1],w0[:,i+1])
    w_inv = adjoint_step(w_inv,u[:,i],s0,dJ0) + source_adjoint*dt
    w_inv,_= decompose_adjoint(w_inv,v0[:,i+1],w0[:,i+1])
    for i1 in arange(n_bins_theta):
        for j1 in arange(n_bins_phi):
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
J_sum_m = zeros((n_bins_theta,n_bins_phi))
J_sum_p = zeros((n_bins_theta,n_bins_phi))
dJds_fd = zeros((n_bins_theta,n_bins_phi))
for i in arange(n_samples):
    um = Step(um,sm,1)
    up = Step(up,sp,1)
    for i1 in arange(n_bins_theta):
        for i2 in arange(n_bins_phi):
            theta0 = theta_bin_centers[i1]
            phi0 = phi_bin_centers[i2]
            J_sum_p[i1,i2] += objective(um,s0,theta0,dtheta,phi0,dphi)
            J_sum_m[i1,i2] += objective(um,s0,theta0,dtheta,phi0,dphi)
            dJds_fd[i1,i2] += (J_sum_p[i1,i2]-J_sum_m[i1,i2]) \
                            / (2.0*epsi)/n_samples
'''
