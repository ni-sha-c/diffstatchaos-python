#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import sys

sys.path.insert(0, '../examples/')
from kuznetsov import *

from pylab import *
from numpy import *
from mpi4py import MPI
from time import time
from util import *

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("test")



n_steps = 100000
n_runup = 10000
n_bins_theta = 20
n_bins_phi = 20
dtheta = pi/n_bins_theta
dphi = 2*pi/n_bins_phi
u = zeros((d,n_steps))
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
theta_bin_centres = linspace(dtheta/2.0,pi - dtheta/2.0,
						n_bins_theta)
phi_bin_centres = linspace(-pi-dphi/2.0,pi - dphi/2.0,
						n_bins_phi)
t_ind = -1
p_ind = -1
v = zeros(d)
w_inv = zeros(d)

for i in arange(1,n_steps):
	v0[:,i] = tangent_step(v0[:,i-1],u[:,i-1],s0,ds0)
	v0[:,i] /= norm(v0[:,i])
	u[:,i] = Step(u[:,i-1],s0,1)
	t_ind = -1
	for theta0 in theta_bin_centres:
		t_ind += 1
		p_ind = 0
		for phi0 in phi_bin_centres:
			J_theta_phi[t_ind,p_ind] += objective(u[:,i-1],
								s0,theta0,dtheta,phi0,dphi)/n_steps 
			p_ind += 1
			

for i in arange(n_steps-1,0,-1):
	w0[:,i-1] = adjoint_step(w0[:,i],u[:,i-1],s0,dJ0)
	w0[:,i-1] /= norm(w0[:,i-1])


for i in arange(1,n_steps-1):
	source_tangent = DfDs(u[:,i],s0)[:,0]
	source_adjoint = divGradfs(u[:,i],s0)
	v = tangent_step(v,u[:,i],s0,ds0) + source_tangent*dt
	v,_= decompose_tangent(v,v0[:,i+1],w0[:,i+1])
	w_inv = adjoint_step(w_inv,u[:,i],s0,dJ0) + source_adjoint*dt
	w_inv,_= decompose_adjoint(w_inv,v0[:,i+1],w0[:,i+1])

	#dJ_ds_stable += dot(dJ_theta_phi,v)/n_steps
	
	

