#!/usr/bin/env python
'''
Statistical sensitivity analysis algorithm for odes.
'''

from __future__ import division
from __future__ import print_function
import sys
import pdb


from pylab import *
from numpy import *
from time import clock
from util import *

from numba import jitclass
from numba import int64, float64

spec = [
    ('n_samples', int64),
    ('n_runup', int64),
    ('n_runup_foradj', int64),
    ('n_steps_corr',int64),
    ('n_steps',int64),
    ('v0',float64[:,:]),
    ('w0',float64[:,:]),
    ('J',float64[:,:,:]),
    ('dJ',float64[:,:,:,:]),
    ('source_sens',float64[:]),
    ('source_tangent',float64[:,:]),
    ('unstable_source_tangent',float64[:,:]),
    ('stable_source_tangent',float64[:,:]),
    ('source_foradj',float64[:,:])


]

@jitclass(spec)
class Sensitivity:
    def __init__(self,solver,n_steps):
        self.n_samples = 100000 
        self.n_runup = 600
        self.n_runup_foradj = 10
        self.n_steps_corr = 10
        self.n_steps = n_steps
        state_dim = solver.state_dim
        n_theta = solver.n_theta
        n_phi = solver.n_phi
        self.v0 = zeros((n_steps,state_dim))
        self.w0 = zeros((n_steps,state_dim)) 
        self.J = zeros((n_steps,n_theta,n_phi))
        self.dJ = zeros((n_steps,n_theta,n_phi,state_dim))
        self.source_sens = zeros(n_steps)
        self.source_tangent = zeros((n_steps,state_dim))
        self.source_foradj = zeros((n_steps,state_dim))

    
    #@jit(nopython=True)
    def solve_primal(self,solver_ode,u_init, n_steps, s):
        u = empty((n_steps, u_init.size))
        u[0] = u_init
        for i in range(1,n_steps):
            u[i] = solver_ode.primal_step(u[i-1],s,1)
        return u


    #@jit(nopython=True)
    def solve_unstable_direction(self, solver_ode, u, v_init, n_steps, s):
        v = empty((n_steps, v_init.size))
        v[0] = v_init
        for i in range(1,n_steps):
            v[i] = solver_ode.tangent_step(v[i-1],u[i-1],s,\
                    zeros(v_init.size))
            v[i] /= linalg.norm(v[i])
        return v
    
    #@jit(nopython=True)
    def solve_unstable_adjoint_direction(self, solver_ode, \
            u, w_init, n_steps, s):
        w = empty((n_steps, w_init.size))
        w[-1] = w_init
        dJ = zeros(w_init.size)
        for i in range(n_steps-1,0,-1):
            w[i-1] = solver_ode.adjoint_step(w[i],u[i-1],s,dJ)
            w[i-1] /= norm(w[i-1])
        return w
    
    #@jit(nopython=True)
    def compute_source_tangent(self, solver_ode, u, n_steps, s0):
        param_dim = s0.size
        state_dim = u.shape[1]
        dfds = zeros((n_steps,param_dim,state_dim))
        DfDs = solver_ode.DfDs
        for i in range(n_steps):
            dfds[i] = DfDs(u[i],s0)
        return dfds
    
    #@jit(nopython=True)
    def compute_source_forward_adjoint(self, solver_ode, u, n_steps, s0):
        state_dim = solver_ode.state_dim
        dgf = zeros((n_steps,state_dim))
        for i in range(n_steps):
            dgf[i] = solver_ode.divGradfs(u[i],s0)
        return dgf
    
    
    #@jit(nopython=True)
    def compute_source_sensitivity(self, solver_ode, u, n_steps, s0):
        param_dim = s0.size
        ddfds = zeros((n_steps,param_dim))
        for i in range(n_steps):
            ddfds[i] = solver_ode.divDfDs(u[i],s0)
        return ddfds
    
    
    #@jit(nopython=True)
    def compute_objective(self, solver_ode, u, s0, n_steps,\
            n_theta=25,n_phi=25):
        theta_bin_centers = linspace(0,pi,n_theta)
        phi_bin_centers = linspace(-pi,pi,n_phi)
        dtheta = pi/(n_theta-1)
        dphi = 2*pi/(n_phi-1)
        objective = solver_ode.objective
        J_theta_phi = zeros((n_steps,n_theta,n_phi))
        for i in arange(1,n_steps):
            for t_ind, theta0 in enumerate(theta_bin_centers):
                for p_ind, phi0 in enumerate(phi_bin_centers):
                    J_theta_phi[i,t_ind,p_ind] += objective(u[i-1],
                            s0,theta0,dtheta,phi0,dphi)/n_steps
        return J_theta_phi 
    
    #@jit(nopython=True)
    def preprocess_objective(self,J_theta_phi):
        n_steps = J_theta_phi.shape[0]
        integral_J = zeros(J_theta_phi.shape)
        integral_J[-1] = copy(J_theta_phi[-1])
        for i in range(n_steps-1,0,-1):
            integral_J[i-1] = integral_J[i] + J_theta_phi[i-1]
        return integral_J
    
    
    #@jit(nopython=True)
    def compute_gradient_objective(self,solver_ode,\
            u,s0,n_steps,n_theta=25,n_phi=25):
        theta_bin_centers = linspace(0,pi,n_theta)
        phi_bin_centers = linspace(-pi,pi,n_phi)
        dtheta = pi/(n_theta-1)
        dphi = 2*pi/(n_phi-1)
        Dobjective = solver_ode.Dobjective
        state_dim = solver_ode.state_dim
        DJ_theta_phi = zeros((n_steps,n_theta,n_phi,state_dim))
        for i in arange(1,n_steps):
            for t_ind, theta0 in enumerate(theta_bin_centers):
                for p_ind, phi0 in enumerate(phi_bin_centers):
                    DJ_theta_phi[i,t_ind,p_ind] += Dobjective(u[i-1],
                            s0,theta0,dtheta,phi0,dphi)
        return DJ_theta_phi 
    
        
    #@jit(nopython=True)
    def compute_correlation_Jg(self,cumsum_J, \
            ddfds,n_samples):
        temp_array = zeros(cumsum_J.shape)
        for i in range(n_samples):
            temp_array[i] = cumsum_J[i]*ddfds[i]
        return temp_array[:n_samples].sum(0)
    
    
    #@jit(nopython=True)
    def precompute_sources(self,solver,u):
        n_steps = u.shape[0]
        state_dim = solver.state_dim
        s0 = solver.s0
        v0_init = rand(state_dim)
        v0_init /= norm(v0_init)
        w0_init = rand(state_dim)
        w0_init /= norm(w0_init)
        n_theta = self.J.shape[1]
        n_phi = self.J.shape[2]
        self.v0 = self.solve_unstable_direction(solver,\
                u, v0_init, n_steps, s0)
        self.source_tangent = self.compute_source_tangent(solver,\
                u, n_steps, s0)[:,0,:] 
        self.w0 = self.solve_unstable_adjoint_direction(solver,\
                u, w0_init, n_steps, s0)
        self.J = self.compute_objective(solver,\
                u, s0, n_steps, n_theta, n_phi)
        self.dJ = self.compute_gradient_objective(solver,\
                u, s0, n_steps, n_theta, n_phi)
        self.source_foradj = \
                self.compute_source_forward_adjoint(solver,\
                u, n_steps, s0)
        self.source_sens = (self.compute_source_sensitivity(solver,\
                u, n_steps, s0))[:,0]
       
       
             
    
    
    if __name__ == '__main__':
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
        source_foradj = zeros((n_steps,state_dim))
        source_tangent = zeros((n_steps,param_dim,state_dim))
        dJds_stable = zeros((n_points_theta,n_points_phi))
        dJds_unstable = zeros((n_points_theta,n_points_phi))
        source_sens = zeros(n_steps)
    
            
        print('='*50)
        print("Pre-computation times for {:>10d} steps".format(n_samples))
        print('='*50)
        print('{:<35s}{:>16.10f}'.format("primal", t1-t0))
        print('{:<35s}{:>16.10f}'.format("tangent",t2 - t1)) 
        print('{:<35s}{:>16.10f}'.format("tangent source", t3 - t2))
        print('{:<35s}{:>16.10f}'.format("adjoint ", t4 - t3))
        print('{:<35s}{:>16.10f}'.format("forward adjoint source", t7 - t6))
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
        v = tangent_step(v,u[i],s0,ds) 
        v,_= decompose_tangent(v,v0[i+1],w0[i+1])
        w_inv = adjoint_step(w_inv,u[i],s0,dJ0) - source_foradj[i]*dt
        w_inv,_= decompose_adjoint(w_inv,v0[i+1],w0[i+1]) 
        if(i < N):
            for t1 in range(ntheta):
                for p1 in range(nphi):
                    dJds_stable += dot(DJ_theta_phi[i+1,t1,p1],v)/N
            dJds_unstable -= cumsumJ[i]*( \
                            dot(dfds[i+1],w_inv))/N

        
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
