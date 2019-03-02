from pylab import *
from numpy import *
from numba import jitclass
from numba import int64, float64
spec = [
    ('L', float64),
    ('dt', float64),
    ('s0', float64[:]),
    ('boundaries', float64[:,:]),
    ('state_dim', int64),
    ('param_dim', int64),
    ('u_init',float64[:]),
]
@jitclass(spec)
class Solver:

    def __init__(self):
        self.dt = 2.e-3
        self.state_dim = 127
        self.param_dim = 5
        self.boundaries = ones((2,self.state_dim))        
        self.boundaries[0] = -0.5
        self.boundaries[1] = 0.5 
        self.u_init = rand(self.state_dim)*(self.boundaries[1]- \
                     self.boundaries[0]) + self.boundaries[0]
        u_init_norm = norm(self.u_init)
        self.u_init /= u_init_norm
        self.s0 = zero(param_dim)
        s0[0] = 0.5
        self.u_init[-1] = 0.0

    
    def primal_step(self,u0,s,n=1):
        u = copy(u0)
        for i in range(n):
            u += dt*self.primal_vector_field(u,s)
        return u

    def primal_vector_field(self,u,s):
        state_dim = u.shape[0]
        dx = self.L/(state_dim+1)
        dx_inv = 1./dx
        dx_inv_sq = dx_inv*dx_inv
        dx_inv_4 = dx_inv_sq*dx_inv_sq
        A = (0.5*c*dx_inv + dx_inv_sq - 4.0*dx_inv_4)*\
                diag(ones(state_dim-1),1)  
        A += (-0.5*c*dx_inv + dx_inv_sq - 4.0*dx_inv_4)*\
                diag(ones(state_dim-1),-1)
        A += (-2.0*dx_inv_sq + 6.0*dx_inv**4.0)*\
                diag(state_dim)
        A += dx_inv_4*diag(ones(state_dim-2),2)
        A += dx_inv_4*diag(ones(state_dim-2),-2)
        u_sq = u*u
        Bu = zeros_like(u_sq)
        Bu[1:-1] = u_sq[2:]-u_sq[:-3]
        Bu[0] = u_sq[1]
        Bu[-1] = -u_sq[-2]
        Bu *= 4.0*dx_inv
        dudt = dot(A,u) + Bu

    
