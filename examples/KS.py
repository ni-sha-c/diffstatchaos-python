from pylab import *
from numpy import *
from numba import jitclass
from numba import int64, float64
from numba import jit
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
        self.dt = 5.e-2
        self.L = 128
        self.state_dim = 127
        self.param_dim = 5
        self.boundaries = ones((2,self.state_dim))        
        self.boundaries[0] = -0.5
        self.boundaries[1] = 0.5 
        self.u_init = rand(self.state_dim)*(self.boundaries[1]- \
                     self.boundaries[0]) + self.boundaries[0]
        u_init_norm = norm(self.u_init)
        self.u_init /= u_init_norm
        self.s0 = zeros(self.param_dim)
        self.s0[0] = 0.8

    
    def primal_step(self,u0,s,n=1):
        u = copy(u0)
        for i in range(n):
            u += self.dt*self.primal_vector_field(u,s)
        return u

    def primal_implicit_vector_field(self,u,s):
        state_dim = u.shape[0]
        dx = self.L/(state_dim+1)
        c = self.s0[0]
        dx_inv = 1./dx
        dx_inv_sq = dx_inv*dx_inv
        dx_inv_4 = dx_inv_sq*dx_inv_sq
        diffusion = 1
        super_diffusion = 1
        super_diagonal_matrix = \
                diag(ones(state_dim -1), 1)
        sub_diagonal_matrix = \
                diag(ones(state_dim -1), -1)
        diagonal_matrix = \
                diag(ones(state_dim))
        super_super_diagonal_matrix = \
                diag(ones(state_dim-2), 2)
        sub_sub_diagonal_matrix = \
                diag(ones(state_dim-2), -2)


        diffusion_coeff = -1.*dx_inv_sq
        super_diffusion_coeff = -1.*dx_inv_4

       
        super_diagonal_matrix_coeff = (diffusion*diffusion_coeff + \
                super_diffusion*super_diffusion_coeff*(-4.0))
        sub_diagonal_matrix_coeff = (diffusion*diffusion_coeff + \
                super_diffusion*super_diffusion_coeff*(-4.))
        diagonal_matrix_coeff = (diffusion*diffusion_coeff*(-2.0) + \
            super_diffusion*super_diffusion_coeff*(6.0))
        super_super_diagonal_matrix_coeff = \
                super_diffusion*super_diffusion_coeff
        sub_sub_diagonal_matrix_coeff = \
                super_diffusion*super_diffusion_coeff


        linear_matrix = super_diagonal_matrix_coeff*super_diagonal_matrix + \
               sub_diagonal_matrix_coeff*sub_diagonal_matrix + \
               diagonal_matrix_coeff*diagonal_matrix + \
               super_super_diagonal_matrix_coeff*super_super_diagonal_matrix + \
               sub_sub_diagonal_matrix_coeff*sub_sub_diagonal_matrix

        linear_matrix[0, 0] += super_diffusion_coeff
        linear_matrix[-1, -1] += super_diffusion_coeff

        dudt = dot(linear_matrix, u)
        
        return dudt

    def primal_explicit_vector_field(self, u, s):
        
        state_dim = u.shape[0]
        dx = self.L/(state_dim+1)
        c = self.s0[0]
        dx_inv = 1./dx

        advection = 1
        nonlinear = 1
        advection_coeff = 0.5*dx_inv
        nonlinear_coeff = 0.25*dx_inv

        super_diagonal_matrix = \
                diag(ones(state_dim -1), 1)
        sub_diagonal_matrix = \
                diag(ones(state_dim -1), -1)
        
        diff_matrix = super_diagonal_matrix - \
                sub_diagonal_matrix

        linear_contrib = dot(diff_matrix, u)
        linear_contrib *= advection_coeff

        u_sq = u*u
        nonlinear_contrib = dot(diff_matrix, u_sq)
        dudt = linear_contrib + nonlinear_contrib

        return dudt

@jit(nopython=True)
def solve_primal(solver, u0, s, n_steps):
    state_dim = u0.shape[0]
    u_trj = zeros((n_steps, state_dim))
    u_trj[0] = u0
    for n in range(1,n_steps):
        u_trj[n] = solver.primal_step(u_trj[n-1],\
                s, 1)
    return u_trj

