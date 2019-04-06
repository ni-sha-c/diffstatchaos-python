from numba import cuda, float64, int64
import numpy as np

@cuda.jit
def tridvec(u_all,dx,fu_all):
	t = cuda.threadIdx.x
	b = cuda.blockIdx.x
	dx_inv = 1/dx
	dx_inv_2 = dx_inv*dx_inv
	dx_inv_4 = dx_inv_2*dx_inv_2
	coeff1 = -dx_inv_2 + 4.0*dx_inv_4
	coeff0 = 2.0*dx_inv_2 - 6.0*dx_inv_4 
	coeff2 = -dx_inv_4

	u = cuda.shared.array(shape=state_dim,dtype=float64)  
	u[t] = u_all[b, t]
	cuda.syncthreads()

	if(t==0):   
		Au = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[0] + \
				coeff1*u[1] + coeff2*u[2]
	elif t==state_dim-1:
		Au = (2.0*dx_inv_2 - 7.0*dx_inv_4)*u[t] + \
				coeff1*u[t-1] + coeff2*u[t-2]
	elif t==1:
		Au = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t+2]
	elif t==state_dim-2:
		Au = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t-2]
	elif t<=state_dim-1:
		Au = coeff1*u[t+1] + coeff1*u[t-1] + \
				coeff0*u[t] + coeff2*u[t-2] + \
				coeff2*u[t+2]
	fu_all[b,t] = Au
		

@cuda.jit
def advection(u_all,dx,s,gu_all):
	dx_inv = 1/dx
	coeffl = -0.5*s*dx_inv  
	coeffnl = -0.25*dx_inv
	t = cuda.threadIdx.x
	b = cuda.blockIdx.x
	
	u = cuda.shared.array(shape=state_dim,dtype=float64)  

	u[t] = u_all[b, t]
	cuda.syncthreads()

	if t==0:
		Bu = coeffl*u[1] + coeffnl*u[1]*u[1]
	elif t==state_dim-1:
		Bu = -coeffl*u[t-1] - coeffnl*u[t-1]*u[t-1]
	elif t < state_dim-1:
		Bu = coeffl*u[t+1] - coeffl*u[t-1] + \
				coeffnl*u[t+1]*u[t+1] - \
				coeffnl*u[t-1]*u[t-1]
	gu_all[b,t] = Bu

#@cuda.jit
#def imexrk2r(u,n):
#	for i in np.range(n):
#		for k in np.range(4):
			
		
L = 512
tpb = 511
state_dim = tpb
dx = L/(state_dim + 1)
bpg = 1000
n_samples = bpg
u = np.ones((n_samples,state_dim))
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
Au_cpu = np.dot(A,u.T)
Au_cpu = Au_cpu.T
assert(np.linalg.norm(Au - Au_cpu)<=1.e-10)
Bu = np.zeros_like(u)
advection[bpg, tpb](u, dx, s, Bu)
coeffl = 0.5*dx_inv
coeffnl = 0.25*dx_inv
B = np.diag(np.ones(state_dim - 1), 1) - \
    np.diag(np.ones(state_dim - 1), -1)
u_sq = u*u
Bu_cpu = s*coeffl*np.dot(B,u.T) + \
	coeffnl*np.dot(B,u_sq.T) 
Bu_cpu = -1.0*Bu_cpu.T
assert(np.linalg.norm(Bu - Bu_cpu)<=1.e-10)
