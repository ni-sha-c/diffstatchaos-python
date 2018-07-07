#Statistical sensitivity analysis algorithm for maps.
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



if __name__ == "__main__":
#def compute_sensitivity()

	n_converge = 10
	n_adjoint_converge = 10
	n_samples = 50000
	n_runup = 1000

	n_steps = n_samples + n_converge +\
            n_adjoint_converge + 1 
			
    n_points_theta = 20
    n_points_phi = 20
    dtheta = pi/(n_points_theta-1)
    dphi = 2*pi/(n_points_phi-1)
    u = zeros((n_steps,state_dim))
    random.seed(0)
    u_init = rand(state_dim)
    u_init[3] = 0
    u_init = poincare_step(u_init,s0,n_runup)
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
    J_theta_phi = compute_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t5 = clock()
    DJ_theta_phi = compute_gradient_objective(u,s0,n_steps,n_points_theta,n_points_phi)
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

		
	 = zeros(n_bins_theta,n_bins_phi)
	
	g = zeros(n_converge,param_dim)
	ϕhat = zeros(d)
	v0 = rand(d)
	v = zeros(d,param_dim)
	
	t = linspace(-pi,pi-binsize_t,n_bins_t)
	r = linspace(0., rmax - binsize_r, n_bins_r)
		
	dFds = zeros(d,param_dim)
	dFds_unstable = zeros(d,param_dim)
	nablaFsinv = zeros(d,d)
	gsum_mean = zeros(param_dim)
	gsum_history = zeros(n_steps,param_dim)
	for n = 1:n_steps

		dFds = ∂F∂s(u,s) 
		nablaFs = gradFs(u,s)
		b = ∇Fsinvcolon∂Fs∂s(u,s)
		q = div∇Fsinv(u,s)
		ϕhat = -1.0*q + 
			transpose(\(nablaFs', ϕhat'))
		v0 = nablaFs*v0
		v0 /= norm(v0)
		ϕhat = dot(ϕhat', v0) * v0'
		u = Step(u,s,1)
		for ip=1:param_dim
			
			g[(n-1)%n_converge+1,iparam_dim] = ϕhat*dFds_unstable[:,ip] - b[ip]   
			gsum_mean[ip] += sum(g[:,ip])
			v[:,ip] = nablaFs*v[:,ip] + dFds[:,ip]
			gsum_history[n,ip] = sum(g[:,ip])
			v[:,ip], dFds_unstable[:,ip] = decompose(v[:,ip], u, v0)
			if(n>=n_converge+n_adjoint_converge+1)
			
				for binno_t=1:n_bins_t
					for binno_r=1:n_bins_r
						φ[binno_t,binno_r] = objective(u,s,t[binno_t],binsize_t,
											r[binno_r],binsize_r)	
						dΦds_unstable[binno_t,binno_r,ip] += 
								#φ[binno_t,binno_r]*(sum(g[:,ip])
								#		-gsum_mean[ip]/n)/n_samples
								φ[binno_t,binno_r]*(sum(g[:,ip]))/n_samples

						gradφ = nablaφ(u,s,t[binno_t],binsize_t,
								r[binno_r],binsize_r)
						dΦds_stable[binno_t,binno_r,ip] += dot(gradφ, v[:,ip])/n_samples
						
				
			
			
		
	

