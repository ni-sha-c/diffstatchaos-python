from numba import cuda, float64
import numpy as np
TPB = 16

@cuda.jit
def tridvec(u,dx,Au):
	t = cuda.grid(1)
	dx_inv = 1/dx
	dx_inv_2 = dx_inv*dx_inv
	dx_inv_4 = dx_inv_2*dx_inv_2
	coeff1 = dx_inv_2 - 4.0*dx_inv_4
	coeff0 = -2.0*dx_inv_2 + 6.0*dx_inv_4 
	state_dim = x.shape[0]
	if t==0:
		Au[0] = (-2.0*dx_inv_2 + 7.0*dx_inv_4)*u[0] + \
				coeff1*u[1] + dx_inv_4*u[2]
	elif t==state_dim-1:
		Au[t] = (-2.0*dx_inv_2 + 7.0*dx_inv_4)*u[t] + \
				coeff1*u[t-1] + dx_inv_4*u[t-2]
	elif t==1:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + dx_inv_4*u[t+2]
	elif t==state_dim-2:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + dx_inv_4*u[t-2]
	else:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + dx_inv_4*u[t-2] + \
				dx_inv_4*u[t+2]


		
tpb = 16
L = 128
state_dim = 127
dx = L/(state_dim + 1)
bpg = np.math.ceil(A.shape[0] / threadsperblock[0])
blockspergrid_y = np.math.ceil(A.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
matmul[blockspergrid, threadsperblock](A,B,C)
