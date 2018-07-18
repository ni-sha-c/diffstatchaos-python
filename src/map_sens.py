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
    ('winv',float64[:,:]),
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
        self.n_runup = 1000
        self.n_runup_foradj = 10
        self.n_steps_corr = 10
        self.n_steps = n_steps
        self.n_samples = self.n_steps - \
                self.n_steps_corr - \
                self.n_runup_foradj - 1
        state_dim = solver.state_dim
        n_theta = solver.n_theta
        n_phi = solver.n_phi
        self.v0 = zeros((n_steps,state_dim))
        self.w0 = zeros((n_steps,state_dim))
        self.winv = zeros((n_steps,state_dim)) 
        self.J = zeros((n_steps,n_theta,n_phi))
        self.dJ = zeros((n_steps,n_theta,n_phi,state_dim))
        self.source_sens = zeros(n_steps)
        self.source_tangent = zeros((n_steps,state_dim))
        self.unstable_source_tangent = zeros((n_steps,state_dim))
        self.stable_source_tangent = zeros((n_steps,state_dim))
        self.source_foradj = zeros((n_steps,state_dim))




    def compute_objective(self,solver_map,\
            u,s0,n_steps,n_theta=25,n_phi=25):
        dtheta = pi/(n_theta-1)
        dphi = 2*pi/(n_phi-1)
        theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
        phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)
        J_theta_phi = zeros((n_steps,n_theta,n_phi))
        objective = solver_map.objective
        for i in arange(n_steps):
            for t_ind, theta0 in enumerate(theta_bin_centers):
                for p_ind, phi0 in enumerate(phi_bin_centers):
                    J_theta_phi[i,t_ind,p_ind] += objective(u[i],
                            s0,theta0,dtheta,phi0,dphi)
        return J_theta_phi 
    
    
    def preprocess_objective(self,J_theta_phi):
        n_steps = J_theta_phi.shape[0]
        integral_J = zeros(J_theta_phi.shape)
        integral_J[-1] = copy(J_theta_phi[-1])
        for i in range(n_steps-1,0,-1):
            integral_J[i-1] = integral_J[i] + J_theta_phi[i-1]
        return integral_J
    
    
    
    def compute_gradient_objective(self,solver_map,\
            u,s0,n_steps,n_theta=25,n_phi=25):
        dtheta = pi/(n_theta-1)
        dphi = 2*pi/(n_phi-1)
        theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
        phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)
        Dobjective = solver_map.Dobjective
        state_dim = solver_map.state_dim
        DJ_theta_phi = zeros((n_steps,n_theta,n_phi,state_dim))
        for i in arange(n_steps):
            for t_ind, theta0 in enumerate(theta_bin_centers):
                for p_ind, phi0 in enumerate(phi_bin_centers):
                    DJ_theta_phi[i,t_ind,p_ind] = Dobjective(u[i],
                            s0,theta0,dtheta,phi0,dphi)
        return DJ_theta_phi 
    
    
    
    def solve_primal(self, solver_map, u_init, n_steps, s):
        u = empty((n_steps, u_init.size))
        u[0] = u_init
        for i in range(1,n_steps):
            u[i] = solver_map.primal_step(u[i-1],s,1)
        return u
    
    
    def solve_unstable_direction(self, solver_map,\
            u, v_init, n_steps, s):
        v = empty((n_steps, v_init.size))
        v[0] = v_init
        v[0] /= norm(v[0])
        for i in range(1,n_steps):
            v[i] = dot(solver_map.gradFs(u[i-1],s),v[i-1])
            v[i] /= linalg.norm(v[i])
        return v
    
    
    def solve_unstable_adjoint_direction(self, solver_map,\
            u, w_init, n_steps, s):
        w = empty((n_steps, w_init.size))
        w[-1] = w_init
        w[-1] /= norm(w[-1])
        for i in range(n_steps-1,0,-1):
            w[i-1] = solver_map.adjoint_step(w[i],u[i-1],s,0)
            w[i-1] /= norm(w[i-1])
        return w
    
    
    
    def compute_source_tangent(self, solver_map, \
            u, n_steps, s0):
        param_dim = s0.size
        dFds = zeros((n_steps,param_dim,u.shape[1]))
        for i in range(n_steps):
            dFds[i] = solver_map.DFDs(u[i],s0)
        return dFds
    
    
    def compute_source_forward_adjoint(self, solver_map,\
            u, n_steps, s0):
        dgf = zeros((n_steps,u.shape[1]))
        for i in range(n_steps):
            dgf[i] = solver_map.divGradFsinv(u[i],s0)
        return dgf
    
    
    def compute_source_sensitivity(self, solver_map, \
            u, n_steps, s0):
        param_dim = s0.size
        t_ddFds_dFinv = zeros((n_steps,param_dim))
        trace_gradDFDs_gradFsinv = solver_map.\
                trace_gradDFDs_gradFsinv
        for i in range(n_steps):
            t_ddFds_dFinv[i] = trace_gradDFDs_gradFsinv(u[i],s0)
        return t_ddFds_dFinv
    
    
    def solve_forward_adjoint(self, solver_map,\
            u, n_steps, s0, source_foradj, v0):
        state_dim = solver_map.state_dim
        w_inv = zeros((n_steps,state_dim))
        for i in range(n_steps - 1):
            nablaFs = solver_map.gradFs(u[i],\
                    solver_map.s0)
            w_inv[i+1] = - source_foradj[i] + \
                    linalg.solve(nablaFs.T, w_inv[i])
            w_inv[i+1] = dot(w_inv[i+1], v0[i+1])*\
                    v0[i+1]
        return w_inv


    def compute_decomposed_source_tangent(self, source_tangent,\
            v0, w0):
        n_steps = source_tangent.shape[0]
        state_dim = v0.shape[1]
        uns_src_tan = zeros((n_steps,state_dim))
        s_src_tan = zeros((n_steps,state_dim))
        for i in range(n_steps-1):
            s_src_tan[i], uns_src_tan[i] =\
                    decompose_tangent(source_tangent[i],\
                    v0[i+1], w0[i+1])
        return s_src_tan, uns_src_tan
            

    
    def precompute_sources(self,solver,u):
        n_steps = u.shape[0]
        s0 = solver.s0
        state_dim = solver.state_dim
        v0_init = rand(state_dim)
        v0_init /= norm(v0_init)
        w0_init = rand(state_dim)
        w0_init /= norm(w0_init)
        n_theta = self.J.shape[1]
        n_phi = self.J.shape[2]

        self.v0 = self.solve_unstable_direction(solver, u, \
                v0_init, n_steps, s0)
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
        self.winv = self.solve_forward_adjoint(solver,\
                u, n_steps, s0, self.source_foradj, self.v0)
        self.stable_source_tangent, self.unstable_source_tangent = \
                self.compute_decomposed_source_tangent(\
                self.source_tangent, self.v0, self.w0)
     
    def compute_stable_sensitivity(self,solver,u):
        state_dim = solver.state_dim
        n_theta = solver.n_theta
        n_phi = solver.n_phi
        n_samples = self.n_samples
        n_steps = u.shape[0]
        dfds_st = self.stable_source_tangent
        v = zeros(state_dim)
        dJds_s = zeros((n_theta,n_phi))
        for i in range(n_steps-1):
            v = dot(solver.gradFs(u[i],solver.s0),v)
            v += dfds_st[i]
            v = v - dot(v,self.w0[i+1])*self.w0[i+1]
            if(i >= self.n_runup_foradj + self.n_steps_corr):
                for bin_t in range(n_theta):
                    for bin_p in range(n_phi):
                        dJds_s[bin_t,bin_p] += dot(v,self.dJ[i+1,\
                            bin_t,bin_p])/n_samples
        return dJds_s


    def compute_timeseries_unstable_sensitivity(self,solver):
        g = zeros((self.n_steps))
        winv = self.winv
        dfds_ust = self.unstable_source_tangent
        for i in range(self.n_steps-1):
            g[i] = dot(winv[i+1],dfds_ust[i]) - \
                    self.source_sens[i]
        return g
        

    def compute_correlation_sum(self,g,f,n_ignore):
        n_corr_max = self.n_steps_corr
        n_total = g.shape[0]
        g_mean = 0.0
        fg_corr_sum = 0.0
        for i in range(n_ignore + n_corr_max,n_total):
            g_mean += sum(g[i-n_corr_max:i])
            fg_corr_sum += f[i]*\
                    (sum(g[i-n_corr_max:i])-g_mean/(i-n_ignore-
                        n_corr_max + 1))/\
                    self.n_samples
        return fg_corr_sum


    def compute_unstable_sensitivity(self,solver):
        n_theta = solver.n_theta
        n_phi = solver.n_phi
        g = self.compute_timeseries_unstable_sensitivity(\
                solver)
        n_ignore = self.n_runup_foradj
        dJds_us = zeros((n_theta,n_phi))
        for bin_t in range(n_theta):
            for bin_p in range(n_phi):
                dJds_us[bin_t,bin_p] += \
                        self.compute_correlation_sum(g,\
                        self.J[:,bin_t,bin_p],n_ignore)
        return dJds_us




        
        
    '''
        g = zeros(n_runup_forward_adjoint)
        w_inv = zeros((n_steps,state_dim))
        v = zeros(state_dim)
        n_points_theta = J.shape[1]
        n_points_phi = J.shape[2]
        gsum_mean = 0.0
        gsum_history = zeros(n_steps)
        dFds_unstable = zeros(state_dim)
        n_samples = N - 2*n_runup_forward_adjoint
        for n in range(N-1):
            b = dJds_0[n]
            q = source_forward_adjoint[n]
            nablaFs = gradFs(u[n],s)   
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
    '''
        
    
    
