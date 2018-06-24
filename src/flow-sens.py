#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys

sys.path.insert(0, '../examples/')
from kuznetsov import *

from pylab import *
from numpy import *
from time import time
from util import *

n_steps = 100
n_runup = 10000
n_bins_theta = 20
n_bins_phi = 20
dtheta = pi/n_bins_theta
dphi = 2*pi/n_bins_phi
u = zeros((d,n_steps))
random.seed(0)
u0 = rand(d)
u0[3] *= T
u0 = Step(u0,s0,n_runup)
u[:,0] = copy(u0)

ds0 = zeros(p)
dJ0 = zeros(d)

v0 = zeros((d,n_steps))
w0 = zeros((d,n_steps))

v0[:,0] = rand(d)
v0[:,0] /= norm(v0[:,0])
w0[:,n_steps-1] = rand(d)
w0[:,n_steps-1] /= norm(w0[:,n_steps-1])

J_theta_phi = zeros((n_bins_theta,n_bins_phi))
theta_bin_centers = linspace(dtheta/2.0,pi - dtheta/2.0, n_bins_theta)
phi_bin_centers = linspace(-pi-dphi/2.0,pi - dphi/2.0, n_bins_phi)
t_ind = -1
p_ind = -1
v = zeros(d)
w_inv = zeros(d)
dfds = zeros((d,p,n_steps))
divdfds = zeros((p,n_steps))
dJds_stable = zeros((n_bins_theta,n_bins_phi))
dJds_unstable = zeros((n_bins_theta,n_bins_phi))

# inputs: u[:,0], s0, ds0, v0[:,0]
for i in arange(1,n_steps):
    v0[:,i] = tangent_step(v0[:,i-1],u[:,i-1],s0,ds0)
    v0[:,i] /= norm(v0[:,i])
    u[:,i] = Step(u[:,i-1],s0,1)

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


'''
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
um = rand(d)*2.0 - 1.0
up = rand(d)*2.0 - 1.0
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
