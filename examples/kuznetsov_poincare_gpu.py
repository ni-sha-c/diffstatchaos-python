from pylab import *
from numpy import *
from numba import cuda
from numba import int64, float64

@cuda.jit
def solve_primal(solver, u0, s, n_steps):
    state_dim = u0.shape[0]
    u_trj = zeros((n_steps, state_dim))
    u_trj[0] = u0
    for n in range(1,n_steps):
        u_trj[n] = solver.primal_step(u_trj[n-1],\
                s, 1)
    return u_trj

@cuda.jit(device=True)
def primal_step(self,u0,s,n=1):
	u = copy(u0)
	for i in range(n):
		u = self.primal_halfstep(u,s,-1.,-1.)
		u = self.primal_halfstep(u,s,1.,1.)
	u[3] = u0[3]
	return u
   
@cuda.jit(device=True)
def primal_halfstep(self,u,s0,sigma,a):
	emmu = exp(-s0[1])
	x = u[0]
	y = u[1]
	z = u[2]
	T = self.T
	r2 = (x**2.0 + y**2.0 + z**2.0)
	r = sqrt(r2)
	rxy2 = x**2.0 + y**2.0
	rxy = sqrt(rxy2)
	em2erxy2 = exp(-2.0*s0[0]*rxy2)
	emerxy2 = exp(-s0[0]*rxy2)
	term = pi*0.5*(z*sqrt(2) + 1)
	sterm = sin(term)
	cterm = cos(term)

	coeff1 = 1.0/((1.0 - emmu)*r + emmu)
	coeff2 = rxy/sqrt((x**2.0)*em2erxy2 + \
		(y**2.0))

	u1 = copy(u)
	u1[0] = coeff1*a*z
	u1[1] = coeff1*coeff2*(sigma*x*emerxy2*sterm + \
		y*cterm)
	u1[2] = coeff1*coeff2*(-a*x*emerxy2*cterm + \
		a*sigma*y*sterm)
	u1[3] = (u[3] + T/2.0)%T

	return u1
   
