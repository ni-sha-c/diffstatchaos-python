#Statistical sensitivity analysis algorithm for maps.
from pylab import *
from numpy import *
from numba import jitclass
from numba import float64, int64
spec = []

@jitclass(spec)
class FourierAnalysis:
    def __init__(self):
        pass

    def solve_primal(self, solver_map, u_init, n_steps, s):
        u = empty((n_steps, u_init.size))
        u[-1] = u_init
        for i in range(n_steps):
            u[i] = solver_map.primal_step(u[i-1],s,1)
        return u
    
    
    def solve_unstable_direction(self, solver_map,\
            u, v_init, n_steps, s):
        v = empty((n_steps, v_init.size))
        v[0] = v_init
        v[0] /= linalg.norm(v[0])
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




    def compute_fourier_transform_pullback(self,u,f,xi,\
            n,n_samples,decorr_len):
        g = exp(1j*dot(u,xi))
        expf_expg = mean(f)*mean(g)
        return self.compute_correlation_function\
                (f,g,n,n_samples,decorr_len) - \
                expf_expg
        


    def compute_correlation(self,f,g,n,n_samples,decorr_len):
        corr = 0.0
        N = f.shape[0]-n
       
        fbar = mean(f)
        gbar = mean(g)
        for i in range(0,N,decorr_len):
            corr += (f[i] - fbar)*(g[n+i] - gbar)/n_samples
        return corr

    def compute_correlation_function(self,f,g,n_max,\
            n_samples,decorr_len):
        corr_fg = zeros(n_max)
        for n in range(n_max):
            corr_fg[n] = self.compute_correlation(f,g,n,\
                    n_samples,decorr_len)
        return corr_fg

            
    def compute_timeshifted_objective(self,solver,J,u0,n):
        n_samples = u0.shape[0]
        objective_trj = zeros((n_samples,n))
        s0 = solver.s0
        for i in range(n_samples):
            u_trj = self.solve_primal(solver,u0[i],\
                    n,s0).T
            objective_trj[i] = J(u_trj)
        return objective_trj


    def compute_directional_derivative_timeshifted_objective(self,solver,DJ,u0,x,n_steps):
        n_samples = u0.shape[0]
        dJdx_trj = zeros((n_samples,n_steps))
        s0 = solver.s0
        v = zeros((n_steps,u0.shape[1]))
        for i in range(n_samples):
            u_trj = self.solve_primal(solver,u0[i],\
                    n_steps,s0)
            v[0] = x[i]
            dJdx_trj[i,0] = dot(DJ(u_trj[0]),v[0])
            for n in range(1,n_steps):
                v[n] = dot(solver.gradFs(u_trj[n-1],s0),\
                        v[n-1])
                dJdx_trj[i,n] = dot(DJ(u_trj[n]),\
                        v[n])
        return dJdx_trj



    
        

