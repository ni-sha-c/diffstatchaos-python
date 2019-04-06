from numba import cuda, float64
import numpy as np
TPB = 16

@cuda.jit
def tridvec(u,dx,Au):
	t = cuda.grid(1)
	dx_inv = 1/dx
	dx_inv_2 = dx_inv*dx_inv
	dx_inv_4 = dx_inv_2*dx_inv_2
	coeff1 = -dx_inv_2 + 4.0*dx_inv_4
	coeff0 = 2.0*dx_inv_2 - 6.0*dx_inv_4 
	coeff2 = -dx_inv_4
	state_dim = u.shape[0]
	if t==0:
		Au[0] = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[0] + \
				coeff1*u[1] + coeff2*u[2]
	elif t==state_dim-1:
		Au[t] = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[t] + \
				coeff1*u[t-1] + coeff2*u[t-2]
	elif t==1:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t+2]
	elif t==state_dim-2:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t-2]
	elif t<=state_dim-1:
		Au[t] = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t-2] + \
				coeff2*u[t+2]

@cuda.jit
def advection(u,dx,s,Bu):
	t = cuda.grid(1)
	dx_inv = 1/dx
	coeffl = -0.5*s*dx_inv  
	coeffnl = -0.25*dx_inv
	state_dim = u.shape[0]
	if t==0:
		Bu[0] = coeffl*u[1] + coeffnl*u[1]*u[1]
	elif t==state_dim-1:
		Bu[t] = -coeffl*u[t-1] - coeffnl*u[t-1]*u[t-1]
	elif t < state_dim-1:
		Bu[t] = coeffl*u[t+1] - coeffl*u[t-1] + \
				coeffnl*u[t+1]*u[t+1] - \
				coeffnl*u[t-1]*u[t-1]

		
tpb = 16
L = 512
state_dim = 511
dx = L/(state_dim + 1)
bpg = (state_dim + 1) // tpb
u = np.ones(state_dim)
s = 1.0
Au = np.zeros_like(u) 
tridvec[bpg, tpb](u, dx, Au)
dx_inv = 1/dx
dx_inv_2 = dx_inv*dx_inv
dx_inv_4 = dx_inv_2*dx_inv_2
coeff0 = 2.0*dx_inv_2 - 6.*dx_inv_4
coeff1 = -dx_inv_2 + 4.*dx_inv_4
coeff2 = -dx_inv_4
A = coeff0*np.diag(np.ones(state_dim),0) + \
    coeff1*np.diag(np.ones(state_dim - 1),1) + \
    coeff1*np.diag(np.ones(state_dim - 1),-1) + \
    coeff2*np.diag(np.ones(state_dim - 2),-2) + \
    coeff2*np.diag(np.ones(state_dim - 2), 2) 
A[0,0] += coeff2
A[-1,-1] += coeff2
Au_cpu = np.dot(A,u)
assert(np.linalg.norm(Au - Au_cpu)<=1.e-10)
Bu = np.zeros_like(u)
advection[bpg, tpb](u, dx, s, Bu)
coeffl = 0.5*dx_inv
coeffnl = 0.25*dx_inv
B = np.diag(np.ones(state_dim - 1), 1) - \
    np.diag(np.ones(state_dim - 1), -1)
u_sq = u*u
Bu_cpu = s*coeffl*np.dot(B,u) + \
	coeffnl*np.dot(B,u_sq) 
Bu_cpu *= -1.
assert(np.linalg.norm(Bu - Bu_cpu)<=1.e-10)
