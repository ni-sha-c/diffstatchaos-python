from pylab import *
from numpy import *
from numba import jit

dt = 1.e-2

s0 = array([1.0,1.0])
T = 6.0
boundaries = array([[-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [0, T]]).T
state_dim = boundaries.shape[1]
n_poincare = int(ceil(T/dt))

@jit(nopython=True)
def primal_step(u0,s,n=1):
    state_dim= u0.size
    param_dim= s.size
    u = copy(u0)
    for i in arange(n):
        x = u[0]
        y = u[1]
        z = u[2]
        r2 = x**2.0 + y**2.0 + z**2.0
        r = sqrt(r2)
        sigma = diff_rot_freq(u[3])
        a = rot_freq(u[3])

        coeff1 = sigma*pi*0.5*(z*sqrt(2) + 1)
        coeff2 = s[0]*(1. - sigma*sigma - a*a)
        coeff3 = s[1]*a*a*(1.0 - r)

        u[0] += dt*(-coeff1*y -
                        coeff2*x*y*y +
                        0.5*a*pi*z + coeff3*x)

        u[1] += dt*(coeff1*x +
                        coeff2*y*x*x +
                        coeff3*y)

        u[2] += dt*(-0.5*a*pi*x + coeff3*z)

        u[3] = (u[3] + dt)%T
    return u

def objective(u,s,theta0,dtheta,phi0,dphi):
    r = sqrt(u[0]**2.0 + u[1]**2.0 + u[2]**2.0)
    theta = 0.0
    if(r > 0):
        theta = arccos(u[2]/r)
    phi = arctan2(u[1],u[0])
    phi += pi
    phi0 += pi
    if(phi0 < dphi):
        phi = (phi + dphi)%(2*pi) - dphi
    if(phi0 > 2*pi - dphi):
        phi = (phi + 2*pi - dphi)%(2*pi) + dphi
    phifrac = (phi-phi0)/dphi
    if(theta0 < dtheta):
        theta = (theta + dtheta)%(pi) - dtheta
    if(theta0 > pi - dtheta):
        theta = (theta + pi - dtheta)%(pi) + dtheta
    pfrac = (phi-phi0)/dphi
    tfrac = (theta-theta0)/dtheta
    obj1 = (max(0.0, min(1.0+pfrac,1.0-pfrac))*
            max(0.0, min(1.0+tfrac,1.0-tfrac)))
    return obj1

def Dobjective(u,s,theta0,dtheta,phi0,dphi):

    res = zeros(state_dim)
    x = u[0]
    y = u[1]
    z = u[2]
    t = u[3]
    r2 = x**2.0 + y**2.0 + z**2.0
    r = sqrt(r2)
    phi = arctan2(y,x)
    phi += pi
    phi0 += pi
    theta = 0.0
    if(r > 0.0):
        theta = arccos(z/r)
    if(phi0 < dphi):
        phi = (phi + dphi)%(2*pi) - dphi
    if(phi0 > 2*pi - dphi):
        phi = (phi + 2*pi - dphi)%(2*pi) + dphi
    if(theta0 < dtheta):
        theta = (theta + dtheta)%(pi) - dtheta
    if(theta0 > pi - dtheta):
        theta = (theta + pi - dtheta)%pi + dtheta

    pfrac = (phi - phi0)/dphi
    tfrac = (theta - theta0)/dtheta

    hattheta = max(0.0, min(tfrac + 1, -tfrac + 1))
    hatphi = max(0.0, min(pfrac + 1, -pfrac + 1))
    ddtheta = 0.0
    ddphi = 0.0
    if (hattheta > 0.0) and (theta > theta0):
        ddtheta = -1.0/dtheta
    if (hattheta > 0.0) and (theta < theta0):
        ddtheta = 1.0/dtheta
    if (hatphi > 0.0) and  (phi > phi0):
        ddphi = -1.0/dphi
    if (hattheta > 0.0) and (theta < theta0):
        ddphi = 1.0/dphi
    sphi = 0.0
    cphi = 1.0
    if(x**2.0 + y**2.0 > 0.0):
        sphi = y/sqrt(x**2.0 + y**2.0)
        cphi = x/sqrt(x**2.0 + y**2.0)
    ct = 1.0
    if(r > 0.0):
        ct = z/r
    st = sqrt(1 - ct*ct)
    dthetadx = dthetady = dthetadz = 0.0
    dphidx = dphidy = dphidz = 0.0
    if (r > 0.0) and (ct != 0.0):
        dthetadx = cphi*ct/r
        dthetady = sphi*ct/r
        dthetadz = -st/r

    if (r > 0.0) and (st != 0.0):
        dphidx = -sphi/r/st
        dphidy = cphi/r/st
        dphidz = 0.0

    res[0] = hatphi*ddtheta*dthetadx + hattheta*ddphi*dphidx
    res[1] = hatphi*ddtheta*dthetady + hattheta*ddphi*dphidy
    res[2] = hatphi*ddtheta*dthetadz + hattheta*ddphi*dphidz
    return res

def convert_to_spherical(u):
    x = u[0]
    y = u[1]
    z = u[2]
    r = sqrt(x**2 + y**2 + z**2)
    theta = arccos(z/r)
    phi = arctan2(y,x)
    return r,theta,phi


def stereographic_projection(u):
    x = u[0]
    y = u[1]
    z = u[2]
    deno = x + z + sqrt(2.0)

    re_part = (x - z)/deno
    im_part = y*sqrt(2.0)/deno

    return re_part,im_part

@jit(nopython=True)
def tangent_source(v0, u, s, ds):
    v = copy(v0)
    x = u[0]
    y = u[1]
    z = u[2]
    t = u[3]
    r2 = x**2 + y**2 + z**2	
    r = sqrt(r2)
    t = t%T
    sigma = diff_rot_freq(t)
    a = rot_freq(t)
    coeff2 = s[0]*(1. - sigma*sigma - a*a)
    coeff3 = s[1]*a*a*(1.0 - r)		
    dcoeff2_ds1 = coeff2/s[0]
    dcoeff3_ds2 = coeff3/s[1]


    v[0] += (-1.0*dcoeff2_ds1*ds[0]*x*y*y + 
                            dcoeff3_ds2*ds[1]*x)
    v[1] += (dcoeff2_ds1*ds[0]*y*x*x + 
                            dcoeff3_ds2*ds[1]*y)
    v[2] += dcoeff3_ds2*ds[1]*z
    
    return v

@jit(nopython=True)
def DfDs(u,s):
    param_dim = s.size
    dfds = zeros((param_dim,state_dim))
    ds1 = array([1.0, 0.0])
    ds2 = array([0.0, 1.0])
    dfds[0] = tangent_source(zeros(state_dim),u,s,ds1)
    dfds[1] = tangent_source(zeros(state_dim),u,s,ds2)
    return dfds



def gradfs(u,s):

	x = u[0]
	y = u[1]
	z = u[2]
	t = u[3]	
	r2 = x**2 + y**2 + z**2	
	r = sqrt(r2)
	
	t = t%T

	sigma = diff_rot_freq(t)
	a = rot_freq(t)
	dsigma_dt = ddiff_rot_freq_dt(t)
	da_dt = drot_freq_dt(t)

	coeff1 = sigma*pi*0.5*(z*sqrt(2) + 1)
	coeff2 = s[0]*(1. - sigma*sigma - a*a)
	coeff3 = s[1]*a*a*(1.0 - r)		

	dcoeff1_dt = pi*0.5*(z*sqrt(2) + 1)*dsigma_dt
	dcoeff2_dt = s[0]*(-2.0)*(sigma*dsigma_dt + a*da_dt)
	dcoeff3_dt = s[1]*(1.0 - r)*2.0*a*da_dt


	dcoeff1_dz = sigma*pi*0.5*sqrt(2)
	dcoeff2_ds1 = coeff2/s[0]
	dcoeff3_ds2 = coeff3/s[1]
	dcoeff3_dx = s[1]*a*a*(-x)/r
	dcoeff3_dy = s[1]*a*a*(-y)/r
	dcoeff3_dz = s[1]*a*a*(-z)/r
			
	dFdu = zeros((state_dim,state_dim))

	dFdu[0,0] = (-coeff2*y*y +
				 coeff3 + dcoeff3_dx*x)

	dFdu[0,1] = (-1.0*coeff1 - 
				coeff2*2.0*y*x + 
				dcoeff3_dy*x)

	dFdu[0,2] = (-1.0*dcoeff1_dz*y + 
				0.5*a*pi + dcoeff3_dz*x)


	dFdu[0,3] = (-1.0*dcoeff1_dt*y - 
				 dcoeff2_dt*x*y*y + 
				 0.5*pi*z*da_dt + 
				 dcoeff3_dt*x)

	dFdu[1,0] = (coeff1*0.5 + 	
				coeff2*y*2.0*x + 
				dcoeff3_dx*y)	 
							 
	dFdu[1,1] = (coeff2*x*x + 
				coeff3 + dcoeff3_dy*y)	

	dFdu[1,2] = (dcoeff1_dz*0.5*x + 
				dcoeff3_dz*y)

	dFdu[1,3] = (dcoeff1_dt*0.5*x + 
				 dcoeff2_dt*y*x*x + 
				 dcoeff3_dt*y)	

	dFdu[2,0] = (-0.5*a*pi + dcoeff3_dx*z)

	dFdu[2,1] = dcoeff3_dy*z

	dFdu[2,2] = coeff3 + dcoeff3_dz*z

	dFdu[2,3] = (-0.5*pi*x*da_dt + dcoeff3_dt*z)

	return dFdu


def divGradfs(u,s):

	epsi = 1.e-8			
	dgf = zeros(d)
	v = zeros(d)
	tmp_matrix = zeros((d,d))
	for i in range(d):
		v = zeros(d)
		v[i] = 1.0
		tmp_matrix = (gradfs(u + epsi*v,s) - 
		 gradfs(u - epsi*v,s))/(2*epsi)
		dgf += tmp_matrix[i,:]

	
	return dgf.T
	

def divDfDs(u,s):

    epsi = 1.e-8
    dpfps = zeros(p)
    v = zeros(d)
    for i in range(d):
        v = zeros(d)
        v[i] = 1.0
        dpfps += ((DfDs(u + epsi*v,s) - 
            DfDs(u - epsi*v,s))[i,:])/(2.0*epsi)
    return dpfps


@jit(nopython=True)
def tangent_step(v0,u,s,ds):

	x = u[0]
	y = u[1]
	z = u[2]
	t = u[3]
	dx = v0[0]
	dy = v0[1]
	dz = v0[2]
	dtime = v0[3]
	v = copy(v0)

	
	r2 = x**2 + y**2 + z**2	
	r = sqrt(r2)
	t = t%T
	sigma = diff_rot_freq(t)
	a = rot_freq(t)
	dsigma_dt = ddiff_rot_freq_dt(t)
	da_dt = drot_freq_dt(t)
	coeff1 = sigma*pi*0.5*(z*sqrt(2) + 1)
	coeff2 = s[0]*(1. - sigma*sigma - a*a)
	coeff3 = s[1]*a*a*(1.0 - r)		

	dcoeff1_dt = pi*0.5*(z*sqrt(2) + 1)*dsigma_dt
	dcoeff2_dt = s[0]*(-2.0)*(sigma*dsigma_dt + a*da_dt)
	dcoeff3_dt = s[1]*(1.0 - r)*2.0*a*da_dt

	dcoeff1_dz = sigma*pi*0.5*sqrt(2)
	dcoeff3_dx = s[1]*a*a*(-x)/r
	dcoeff3_dy = s[1]*a*a*(-y)/r
	dcoeff3_dz = s[1]*a*a*(-z)/r

	dcoeff2_ds1 = coeff2/s[0]
	dcoeff3_ds2 = coeff3/s[1]

	v[0] += dt*(-1.0*dcoeff1_dz*y*dz - 1.0*
				coeff1*dy - dcoeff2_ds1*ds[0]*x*y*y - 
				coeff2*y*y*dx +
                                - coeff2*x*2.0*y*dy + 
				0.5*a*pi*dz + dcoeff3_ds2*ds[1]*x + 
				dcoeff3_dx*x*dx + 
				dcoeff3_dy*x*dy + 
				dcoeff3_dz*x*dz +
				coeff3*dx - 1.0*dcoeff1_dt*y*dtime +
                                - dcoeff2_dt*x*y*y*dtime + 
				0.5*da_dt*pi*z*dtime + 
				dcoeff3_dt*x*dtime)

	v[1] += dt*(coeff1*dx + 
				dcoeff1_dz*x*dz + 
				dcoeff2_ds1*y*x*x*ds[0] + 
				coeff2*dy*x*x + 
				coeff2*2.0*x*dx*y + 
				dcoeff3_ds2*ds[1]*y + 
				dcoeff3_dx*y*dx + 
				dcoeff3_dy*y*dy +
		 		dcoeff3_dz*y*dz +
			 	coeff3*dy +
                                dcoeff1_dt*x*dtime +
				dcoeff2_dt*y*x*x*dtime + 
				dcoeff3_dt*y*dtime) 

	v[2] += dt*(-0.5*a*pi*dx + 
				dcoeff3_ds2*z*ds[1] + 
				dcoeff3_dx*z*dx + 
				dcoeff3_dy*z*dy + 
				dcoeff3_dz*z*dz + 
				coeff3*dz - 
				0.5*pi*x*da_dt*dtime + 
				dcoeff3_dt*z*dtime)
	 

	return v





@jit(nopython=True)
def rot_freq(t): 
    a0 = -1.0
    a1 = 0.0
    a2 = 1.0

    c0 = 2.0
    c1 = 3.0
    c2 = 5.0
    c3 = 6.0
    c4 = 0.0

    '''
    if t > c0 and t < c1:
        return -1
    elif t > c2 and t < c3:
        return 1
    else:
        return 0
    '''
    slope = 20.0
    est = exp(slope*t)
    esc0 = exp(slope*c0)
    esc1 = exp(slope*c1)
    esc2 = exp(slope*c2)
    esc3 = exp(slope*c3)
    esc4 = exp(slope*c4)

    fn0 = (a1*esc0 + a0*est)/(esc0 + est)	
    fn1 = (a0*esc1 + a1*est)/(esc1 + est)
    fn2 = (a1*esc2 + a2*est)/(esc2 + est)
    fn3 = (a2*esc3 + a1*est)/(esc3 + est)
    fn4 = (a2*esc4 + a1*est)/(esc4 + est)

    return fn0 + fn1 + fn2 + fn3 + fn4



@jit(nopython=True)
def diff_rot_freq(t):
    a0 = -1.0
    a1 = 0.0
    a2 = 1.0
    c0 = 1.0
    c1 = 2.0
    c2 = 4.0
    c3 = 5.0 

    '''
    if t > c0 and t < c1:
        return -1
    elif t > c2 and t < c3:
        return 1
    else:
        return 0
    '''

    slope = 20.0
    est = exp(slope*t)
    esc0 = exp(slope*c0)
    esc1 = exp(slope*c1)
    esc2 = exp(slope*c2)
    esc3 = exp(slope*c3)
	
    fn0 = (a1*esc0 + a0*est)/(esc0 + est)	
    fn1 = (a0*esc1 + a1*est)/(esc1 + est)
    fn2 = (a1*esc2 + a2*est)/(esc2 + est)
    fn3 = (a2*esc3 + a1*est)/(esc3 + est)

    return fn0 + fn1 + fn2 + fn3


@jit(nopython=True)
def ddiff_rot_freq_dt(t):

    a0 = -1.0
    a1 = 0.0
    a2 = 1.0

    c0 = 1.0
    c1 = 2.0
    c2 = 4.0
    c3 = 5.0 

    slope = 20.0
    est = exp(slope*t)
    esc0 = exp(slope*c0)
    esc1 = exp(slope*c1)
    esc2 = exp(slope*c2)
    esc3 = exp(slope*c3)


    dfn0 = esc0*est*slope*(a0-a1)/(esc0 + est)/(esc0 + est)
    dfn1 = esc1*est*slope*(a1-a0)/(esc1 + est)/(esc1 + est)
    dfn2 = esc2*est*slope*(a2-a1)/(esc2 + est)/(esc2 + est)
    dfn3 = esc3*est*slope*(a1-a2)/(esc3 + est)/(esc3 + est)

    return dfn0 + dfn1 + dfn2 + dfn3 




@jit(nopython=True)
def drot_freq_dt(t):
    
    a0 = -1.0
    a1 = 0.0
    a2 = 1.0

    c0 = 2.0
    c1 = 3.0
    c2 = 5.0
    c3 = 6.0 
    c4 = 0.0

    slope = 20.0
    est = exp(slope*t)
    esc0 = exp(slope*c0)
    esc1 = exp(slope*c1)
    esc2 = exp(slope*c2)
    esc3 = exp(slope*c3)
    esc4 = exp(slope*c4)

    fn0 = (a1*esc0 + a0*est)/(esc0 + est)	
    fn1 = (a0*esc1 + a1*est)/(esc1 + est)
    fn2 = (a1*esc2 + a2*est)/(esc2 + est)
    fn3 = (a2*esc3 + a1*est)/(esc3 + est)
    fn4 = (a2*esc4 + a1*est)/(esc4 + est)

    dfn0 = esc0*est*slope*(a0-a1)/(esc0 + est)/(esc0 + est)
    dfn1 = esc1*est*slope*(a1-a0)/(esc1 + est)/(esc1 + est)
    dfn2 = esc2*est*slope*(a2-a1)/(esc2 + est)/(esc2 + est)
    dfn3 = esc3*est*slope*(a1-a2)/(esc3 + est)/(esc3 + est)
    dfn4 = esc4*est*slope*(a1-a2)/(esc4 + est)/(esc4 + est)

    return dfn0 + dfn1 + dfn2 + dfn3 + dfn4



@jit(nopython=True)
def adjoint_step(y1,u,s,dJ):


	y0 = copy(y1)

	x = u[0]
	y = u[1]
	z = u[2]
	t = u[3]

	r2 = x**2 + y**2 + z**2
	r = sqrt(r2)
	sigma = diff_rot_freq(t)
	a = rot_freq(t)
	dsigmadt = ddiff_rot_freq_dt(t)
	dadt = drot_freq_dt(t)
	coeff1 = sigma*pi*0.5*(z*sqrt(2) + 1)
	coeff2 = s[0]*(1. - sigma*sigma - a*a)
	coeff3 = s[1]*a*a*(1.0 - r)

	dcoeff1dt = pi*0.5*(z*sqrt(2) + 1)*dsigmadt
	dcoeff2dt = s[0]*(-2.0)*(sigma*dsigmadt + a*dadt)
	dcoeff3dt = s[1]*(1.0 - r)*2.0*a*dadt

	dcoeff1dz = sigma*pi*0.5*sqrt(2)
	dcoeff3dx = s[1]*a*a*(-x)/r 
	dcoeff3dy = s[1]*a*a*(-y)/r
	dcoeff3dz = s[1]*a*a*(-z)/r 

	y0[0] += dt * (y1[0]*(-1.0*coeff2*y*y) + 
			y1[0]*coeff3 + 
			y1[0]*x*dcoeff3dx + 
			y1[1]*coeff1 + 
			y1[1]*coeff2*y*2.0*x + 
			y1[1]*dcoeff3dx*y + 
			y1[2]*(-0.5)*a*pi + 
			y1[2]*z*dcoeff3dx) 	

	y0[1] += (y1[0]*dt*(-1.0)*coeff1 +
			y1[0]*dt*(-1.0)*coeff2*x*2.0*y + 
			y1[0]*dt*dcoeff3dy*x + 
			y1[1]*dt*coeff2*x*x + 
			y1[1]*dt*coeff3 + 
			y1[1]*dt*dcoeff3dy*y + 
			y1[2]*dt*dcoeff3dy*z)

	
	y0[2] += (y1[0]*dt*(-1.0)*dcoeff1dz*y + 
			y1[0]*dt*0.5*a*pi + 
			y1[0]*dt*dcoeff3dz*x + 
			y1[1]*dt*dcoeff1dz*x + 
			y1[1]*dt*y*dcoeff3dz + 
			y1[2]*dt*z*dcoeff3dz + 
			y1[2]*dt*coeff3)


	y0[3] += (-1.0*y1[0]*dt*dcoeff1dt*y + 
			  -y1[0]*dt*x*y*y*dcoeff2dt + 
			 y1[0]*dt*x*dcoeff3dt +
			 y1[0]*dt*0.5*pi*z*dadt +  
			 y1[1]*dt*x*dcoeff1dt + 
			 y1[1]*dt*y*x*x*dcoeff2dt +
			 y1[1]*dt*y*dcoeff3dt + 
			 y1[2]*dt*dcoeff3dt*z + 
			 y1[2]*dt*(-0.5)*pi*x*dadt) 
			 

	y0 += dJ		

	return y0


