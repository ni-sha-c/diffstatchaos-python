#Statistical sensitivity analysis algorithm for maps.
from __future__ import division
from __future__ import print_function
import sys
import pdb


from pylab import *
from numpy import *
from time import clock
from util import *

from numba import jitclass
from numba import int64, float64

@jitclass
class FourierAnalysis:
	def init(self):
		pass

	def solve_poincare_primal(u_init, n_steps, s):
		u = empty((n_steps, u_init.size))
		u[0] = u_init
		for i in range(1,n_steps):
			u[i] = poincare_step(u[i-1],s)
		return u


	def solve_poincare_unstable_direction(u, v_init, n_steps, s, ds):
		v = empty((n_steps, v_init.size))
		v[0] = v_init
		for i in range(1,n_steps):
			v[i] = tangent_poincare_step(v[i-1],u[i-1],s,ds)
			v[i] /= linalg.norm(v[i])
		return v


	def solve_poincare_unstable_adjoint_direction(u, w_init, n_steps, s, dJ):
		w = empty((n_steps, w_init.size))
		w[-1] = w_init
		for i in range(n_steps-1,0,-1):
			w[i-1] = adjoint_poincare_step(w[i],u[i-1],s,dJ)
			w[i-1] /= norm(w[i-1])
		return w



	def compute_poincare_source_tangent(u, n_steps, s0):
		param_dim = s0.size
		dFds = zeros((n_steps,param_dim,state_dim))
		for i in range(n_steps):
			dFds[i] = DFDs_poincare(u[i],s0)
		return dFds




	def compute_fourier_transform(u,J,xi):
		pass
	




