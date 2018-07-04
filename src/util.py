from pylab import *
from numpy import *
from numba import *
@jit(nopython=True)
def decompose_tangent(v,v0,w0):
    vu = dot(v,w0)/dot(v0,w0)*v0
    vs = v - vu
    return vs,vu

@jit(nopython=True)
def decompose_adjoint(w,v0,w0):
    wu = dot(w,v0)/dot(w0,v0)*w0
    ws = w - wu
    return wu,ws
