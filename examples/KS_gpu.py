from pylab import *
from numpy import *
from numba import cuda
from numba import int64, float64, int32, float32
import cupy as cp
import numpy as np
@cuda.jit
def imexrk342r(u_all,A,A_imp,A_exp,brk):
	u = cuda.shared.array(shape=state_dim,dtype=float64)  
	u_imp = cuda.shared.array(shape=state_dim,dtype=float64)  
	u_buf = cuda.shared.array(shape=state_dim,dtype=float64)  
	t = cuda.threadIdx.x
	b = cuda.blockIdx.x
	u[t] = u_all[b, t]
	u_imp[t] = u_all[b,t]
	u_buf[t] = u_all[b,t]

	if(t==0):   
		f = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[0] + \
                coeff1*u[1] + coeff2*u[2]
		g = coeffl*u[1] + coeffnl*u[1]*u[1]
	elif t==state_dim-1:
		f = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[t] + \
                coeff1*u[t-1] + coeff2*u[t-2]
		g = -coeffl*u[t-1] - coeffnl*u[t-1]*u[t-1]
	elif t==1:
		f = coeff1*u[t+1] + coeff1*u[t-1] + \
                coeff0*u[t] + coeff2*u[t+2]
		g = coeffl*u[t+1] - coeffl*u[t-1] + \
                coeffnl*u[t+1]*u[t+1] - \
                coeffnl*u[t-1]*u[t-1]
	elif t==state_dim-2:
		f = coeff1*u[t+1] + coeff1*u[t-1] + \
                coeff0*u[t] + coeff2*u[t-2]
		g = coeffl*u[t+1] - coeffl*u[t-1] + \
                coeffnl*u[t+1]*u[t+1] - \
                coeffnl*u[t-1]*u[t-1]
	elif t < state_dim-1:
		f = coeff1*u[t+1] + coeff1*u[t-1] + \
                coeff0*u[t] + coeff2*u[t-2] + \
                coeff2*u[t+2]
		g = coeffl*u[t+1] - coeffl*u[t-1] + \
                coeffnl*u[t+1]*u[t+1] - \
                coeffnl*u[t-1]*u[t-1]

	u_buf[t] = u[t] + dt*A_imp[1,0]*f + \
					dt*A_exp[1,0]*g
	cuda.syncthreads()
	u_imp[t] = u_buf[t]
	cuda.syncthreads()

	#	for i in range(state_dim):
	


	




L = 128
state_dim = 127
n_samples = 1000
dx = L/(state_dim + 1)
tpb = state_dim
bpg = n_samples
n_stage = 4
A_exp_host = np.array([zeros(n_stage),\
		[1./3,0.,0.,0.],\
		[0.,1.,0.,0.],\
	    [0.,3./4,1./4.,0.]])
A_imp_host = np.array([zeros(n_stage),\
		[0.,1./3,0.,0.],\
		[0.,1./2,1/2.,0.],\
	    [0.,3./4,-1./4.,1/2.]])
b_host = np.array([0., 3./4., -1./4., 1./2])
A_exp = cuda.to_device(A_exp_host)
A_imp = cuda.to_device(A_imp_host)
brk = cuda.to_device(b_host)

dx_inv = 1/dx
dx_inv_2 = dx_inv*dx_inv
dx_inv_4 = dx_inv_2*dx_inv_2
dt = 1.e-1
coeff0 = 2.0*dx_inv_2 - 6.*dx_inv_4
coeff1 = -dx_inv_2 + 4.*dx_inv_4
coeff2 = -dx_inv_4
coeffl = 0.5*dx_inv
coeffnl = 0.25*dx_inv
A = coeff0*np.diag(np.ones(state_dim),0) + \
    coeff1*np.diag(np.ones(state_dim - 1),1) + \
    coeff1*np.diag(np.ones(state_dim - 1),-1) + \
    coeff2*np.diag(np.ones(state_dim - 2),-2) + \
    coeff2*np.diag(np.ones(state_dim - 2), 2) 
A[0,0] += coeff2
A[-1,-1] += coeff2
A = cuda.to_device(A)

#Imp_0 is eye(state_dim) and Imp_3 = Imp_2 so neither is stored.
Imp_1 = np.linalg.inv(eye(state_dim) - dt*A_imp[1,1]*A)
Imp_2 = np.linalg.inv(eye(state_dim) - dt*A_imp[2,2]*A)
u_all = ones((n_samples,state_dim))
imexrk342r[bpg, tpb](u_all,A,A_imp,A_exp,brk)
