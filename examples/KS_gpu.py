from pylab import *
from numpy import *
from numba import cuda
from numba import int64, float64, int32, float32
        

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

@cuda.jit
def primal_step(u, s, n_steps, dt):
	state_dim = u.shape[0]
	lu = cuda.local.array(state_dim, dtype=float32)
	t = grid(1)
	for i in range(state_dim):
		lu[i] = u[t,i]	
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
