from pylab import *
from numpy import *
from numba import jitclass
from numba import float64, int64
spec = [
    ('s0', float64[:]),
    ('state_dim',int64),
    ('u_init',float64[:]),
    ('param_dim',int64),
    ('m',float64)
]

@jitclass(spec)
class Solver:
    def __init__(self):
        self.state_dim = 2
        self.param_dim = 1
        self.u_init = rand(self.state_dim)
        self.m = 10.0
        self.s0 = zeros(self.param_dim)
        self.s0[0] = 0.


    def primal_step(self,u0,s,n=1):
        u1 = copy(u0)
        for i in range(n):
            u = copy(u1)
            u1[0] = (2.0*u[0] + s[0]/(2**self.m)\
                    *sin((2**self.m)*pi*u[0]) + u[1])%1
            u1[1] = (u[0] + u[1])%1 
        return u1


    def get_lyapunov_exponents_vectors(self):
        DF = array([2.0,1.0,1.0,1.0]).reshape(2,2)
        lyapunov_exponents, v_basis = eig(DF)
        return lyapunov_exponents, v_basis

    def gradFs(self,u,s):
        return array([2.0,1.0,1.0,1.0]).reshape(2,2)
