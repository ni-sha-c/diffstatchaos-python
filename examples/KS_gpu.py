from pylab import *
from numpy import *
from numba import cuda
from numba import int64, float64
        

n_stage = 4
A_exp_host = array([zeros(n_stage),\
		[1./3,0.,0.,0.],\
		[0.,1.,0.,0.],\
	    [0.,3./4,1./4.,0.]])
A_imp_host = array([zeros(n_stage),\
		[0.,1./3,0.,0.],\
		[0.,1./2,1/2.,0.],\
	    [0.,3./4,-1./4.,1/2.]])
b_exp_host = array([0., 3./4., -1./4., 1./2])
b_imp_host = array([0., 3./4., -1./4., 1./2])
A_exp = cuda.to_device(A_exp_host)
A_imp = cuda.to_device(A_imp_host)
b_exp = cuda.to_device(b_exp_host)
b_imp = cuda.to_device(b_imp_host)
dt_host = 1.e-1
dt = cuda.to_device(dt_host)
L = 128
state_dim = 127
dx_host = L/(state_dim + 1)
dx = cuda.to_device(dx_host)

@cuda.jit(device=True)
def primal_implicit_update(u, stage_no, C):
	an = A_imp[stage_no, stage_no]
	state_dim = state_dim
	B = eye(state_dim) - dt*an*A
	C = linalg.solve(B,dot(A,u))

@cuda.jit(device=True)
def primal_implicit_matrix(u, A):
	state_dim = u.shape[0]
	dx_inv = 1./dx
	dx_inv_sq = dx_inv*dx_inv
	dx_inv_4 = dx_inv_sq*dx_inv_sq
	A = (dx_inv_sq - 4.0*dx_inv_4)*diag(ones(state_dim-1),1) + \
		(dx_inv_sq - 4.0*dx_inv_4)*diag(ones(state_dim-1),-1) + \
		(-2.0*dx_inv_sq + 6.0*dx_inv_4)*eye(state_dim) + \
		 dx_inv_4*diag(ones(state_dim-2),2) + \
		 dx_inv_4*diag(ones(state_dim-2),-2)

	A[0, 0] += dx_inv_4
	A[-1, -1] += dx_inv_4
	A *= -1.0



'''
@cuda.jit(device=True)
def primal_step(u0,s,n_steps):
	n_stage = n_stage
	dt = dt
	state_dim = state_dim
	evf = primal_explicit_vector_field
	ivf = primal_implicit_vector_field
	iu = primal_implicit_update
	dudt_exp = zeros((n_stage,state_dim))
	dudt_imp = zeros((n_stage,state_dim))
	A_exp = A_exp
	A_imp = A_imp
	b_exp = b_exp
	b_imp = b_imp
	u = copy(u0)
	for i in range(n_steps):
		explicit_field = copy(u)
		implicit_field = iu(explicit_field,0)
		explicit_field = evf(explicit_field + \
				dt*A_imp[0,0]*implicit_field,s)
		u += dt*b_imp[0]*implicit_field + \
				dt*b_exp[0]*explicit_field
		for n in range(1,n_stage):
			explicit_field = u + \
					dt*(A_imp[n,n-1] - b_imp[n-1])*implicit_field + \
					dt*(A_exp[n,n-1] - b_exp[n-1])*explicit_field
			implicit_field = iu(explicit_field,n)
			explicit_field = evf(explicit_field + dt*A_imp[n,n]*implicit_field,s)
			u += dt*b_imp[n]*implicit_field + \
				 dt*b_exp[n]*explicit_field
	return u


@cuda.jit(device=True)
def primal_implicit_vector_field(u,s):
	implicit_matrix = primal_implicit_matrix(u,s)
	return dot(implicit_matrix, u)

@cuda.jit(device=True)
def primal_explicit_vector_field( u, s):
	
	state_dim = u.shape[0]
	dx = L/(state_dim+1)
	c = s[0]
	dx_inv = 1./dx

	advection = 1
	nonlinear = 1
	advection_coeff = 0.5*dx_inv
	nonlinear_coeff = 0.25*dx_inv

	
	linear_contrib = dot(diag(ones(state_dim-1),1) - \
			diag(ones(state_dim-1),-1), u)
	linear_contrib *= c*advection_coeff

	u_sq = u*u
	nonlinear_contrib = dot(diag(ones(state_dim-1),1)-\
			diag(ones(state_dim-1),-1), u_sq)
	nonlinear_contrib *= nonlinear_coeff
	dudt = -1.0*(linear_contrib + nonlinear_contrib)

	return dudt

'''
