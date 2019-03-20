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
    ('n_stage', int64),
    ('A_imp', float64[:,:]),
    ('A_exp', float64[:,:]),
    ('b_exp',float64[:]),
    ('b_imp',float64[:])
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
        self.n_stage = 4
        self.A_exp = self.populate_A_exp()
        self.A_imp = self.populate_A_imp()
        self.b_exp = self.populate_b_exp()
        self.b_imp = self.populate_b_imp()
        

    def populate_A_exp(self):
        n_stage = self.n_stage
        A = zeros((n_stage, n_stage))
        A[1, 0] = 1./3.
        A[2, 1] = 1.
        A[3, 1] = 3./4.
        A[3, 2] = 1./4.
        return A

    def populate_A_imp(self):
        n_stage = self.n_stage
        A = zeros((n_stage, n_stage))
        A[1, 1] = 1./3.
        A[2, 1] = 1./2.
        A[2, 2] = 1./2.
        A[3, 1] = 3./4.
        A[3, 2] = -1./4.
        A[3, 3] = 1./2.
        return A

    def populate_b_exp(self):
        n_stage = self.n_stage
        b = zeros(n_stage)
        b[1] = 3./4.
        b[2] = -1./4.
        b[3] = 1./2.
        return b

    def populate_b_imp(self):
        n_stage = self.n_stage
        b = zeros(n_stage)
        b[1] = 3./4.
        b[2] = -1./4.
        b[3] = 1./2.
        return b

    def populate_implicit_matrix(self,stage_no):
        an = self.A_imp[stage_no, stage_no]
        state_dim = self.state_dim
        dt = self.dt
        A = self.primal_implicit_vector_field(self.u_init, self.s0)
        B = eye(state_dim) - dt*an*A
        return inv(B)

    def primal_step(self,u0,s,n=1):
        n_stage = self.n_stage
        dt = self.dt
        state_dim = self.state_dim
        evf = self.primal_explicit_vector_field
        ivf = self.primal_implicit_vector_field(u0,s)
        dudt_exp = zeros((n_stage,state_dim))
        dudt_imp = zeros((n_stage,state_dim))
        A_exp = self.A_exp
        A_imp = self.A_imp
        b_exp = self.b_exp
        b_imp = self.b_imp
        implicit_matrix = zeros((n_stage, state_dim,\
                state_dim))
        for n in range(1,n_stage):
            implicit_matrix[n] = self.populate_implicit_matrix(n)
        ui = copy(u0)
        for i in range(n):
            u = copy(ui)
            dudt_exp[0] = evf(u,s)
            dudt_imp[0] = dot(ivf,u)
            for n in range(1,n_stage):
                dudt_exp[n] = evf(u,s)
                dudt_imp[n] = dot(implicit_matrix[n],\
                        dot(ivf,u))
                u = u + dt*sum(A_exp[n,:n]*(dudt_exp[:n].T), 1) + \
                        dt*sum(A_imp[n,:n]*(dudt_imp[:n].T), 1) 
            ui += dt*sum(b_exp*(dudt_exp.T), 1) + \
                  dt*sum(b_imp*(dudt_imp.T), 1)
        print(ui.shape)
        return ui

    def primal_implicit_vector_field(self,u,s):
        print("inside implicit")
        state_dim = u.shape[0]
        dx = self.L/(state_dim+1)
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


        diffusion_coeff = dx_inv_sq
        super_diffusion_coeff = dx_inv_4

       
        super_diagonal_matrix_coeff = (diffusion*diffusion_coeff + \
                super_diffusion*super_diffusion_coeff*(-4.0))
        sub_diagonal_matrix_coeff = (diffusion*diffusion_coeff + \
                super_diffusion*super_diffusion_coeff*(-4.0))
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
        linear_matrix *= -1.0
        return linear_matrix

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
        linear_contrib *= c*advection_coeff

        u_sq = u*u
        nonlinear_contrib = dot(diff_matrix, u_sq)
        dudt = -1.0*(linear_contrib + nonlinear_contrib)

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

