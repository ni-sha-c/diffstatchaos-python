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
        self.dt = 1.e-1
        self.L = 8
        self.state_dim = 7
        self.param_dim = 5
        self.boundaries = ones((2,self.state_dim))        
        self.boundaries[0] = -0.5
        self.boundaries[1] = 0.5 
        self.u_init = rand(self.state_dim)*(self.boundaries[1]- \
                     self.boundaries[0]) + self.boundaries[0]
        u_init_norm = norm(self.u_init)
        self.u_init /= u_init_norm
        self.s0 = zeros(self.param_dim)
        self.s0[0] = 0.5
        self.u_init[-1] = 0.0

    
    def primal_step(self,u0,s,n=1):
        u = copy(u0)
        for i in range(n):
            u += self.dt*self.primal_vector_field(u,s)
        return u

    def primal_vector_field(self,u,s):
        state_dim = u.shape[0]
        dx = self.L/(state_dim+1)
        c = self.s0[0]
        dx_inv = 1./dx
        dx_inv_sq = dx_inv*dx_inv
        dx_inv_4 = dx_inv_sq*dx_inv_sq
        advection = 1
        diffusion = 1
        nonlinear = 1
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


        advection_coeff = -1.*c*0.5*dx_inv
        diffusion_coeff = -1.*dx_inv_sq
        super_diffusion_coeff = -1.*dx_inv_4
        nonlinear_coeff = -1.*0.25*dx_inv

       
        super_diagonal_matrix_coeff = (advection*advection_coeff + \
                diffusion*diffusion_coeff + \
                super_diffusion*super_diffusion_coeff*(-4.0))
        sub_diagonal_matrix_coeff = (advection*advection_coeff*(-1) + \
                diffusion*diffusion_coeff + \
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

        linear_contrib = dot(linear_matrix, u)
        
         
        u_sq = u*u
        nonlinear_contrib = zeros_like(u_sq)
        nonlinear_contrib[1:-1] = u_sq[2:]-u_sq[:-2]
        nonlinear_contrib[0] = u_sq[1]
        nonlinear_contrib[-1] = -u_sq[-2]
        nonlinear_contrib *= nonlinear_coeff

        dudt = linear_contrib + nonlinear_contrib
        return dudt


@jit(nopython=True)
def solve_primal(solver, u0, s, n_steps):
    state_dim = u0.shape[0]
    u_trj = zeros((n_steps, state_dim))
    u_trj[0] = u0
    for n in range(1,n_steps+1):
        u_trj[i-1] = solver.primal_step(u_trj[i-1],\
                s, 1)
    return u_trj

