from pylab import *
from numpy import *
from numba import jitclass
from numba import int64, float64
spec = [
    ('T', float64),
    ('dt', float64),
    ('s0', float64[:]),
    ('boundaries', float64[:,:]),
    ('state_dim', int64),
    ('n_poincare',int64),
    ('param_dim', int64),
    ('u_init',float64[:])
]
@jitclass(spec)
class Solver:

    def __init__(self):
        self.dt = 2.e-3
        self.s0 = array([1.0,1.0])
        self.T = 6.0
        self.state_dim = 4
        self.boundaries = ones((2,self.state_dim))        
        self.boundaries[0] = -1.
        self.boundaries[0,-1] = 0.
        self.boundaries[1,-1] = self.T
        self.n_poincare = int(ceil(self.T/self.dt))
        self.u_init = rand(self.state_dim)*(self.boundaries[1]- \
                     self.boundaries[0]) + self.boundaries[0]
        self.u_init[-1] = 0.0
        self.param_dim = self.s0.size


    
    def primal_step(self,u0,s,n=1):
        u = copy(u0)
        for i in range(n):
            u = self.primal_halfstep(u,s,-1.,-1.)
            u = self.primal_halfstep(u,s,1.,1.)
        u[3] = u0[3]
        return u
    
    
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

    
    def objective(self,u,s,theta0,dtheta,phi0,dphi):
        r = sqrt(u[0]**2.0 + u[1]**2.0 + u[2]**2.0)
        theta = 0.0
        if(r > 0):
            theta = arccos(u[2]/r)
        phi = arctan2(u[1],u[0])
        phi += pi
        phi0 += pi
        #if(phi0 < dphi):
        #    phi = (phi + dphi)%(2*pi) - dphi
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
    
    
    def Dobjective(self,u,s,theta0,dtheta,phi0,dphi):
    
        res = zeros(state_dim)
        epsi = 1.e-5

        '''
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
        '''

        for l in range(state_dim):
            v0 = zeros(state_dim)
            v0[l] = 1.0
            res[l] = (self.objective(u+epsi*v0,\
                    s,theta0,dtheta,phi0,dphi)- \
                    self.objective(u-epsi*v0,s,theta0,dtheta,\
                    phi0,dphi))/(2.0*epsi)
    
    
        return res
    
    def convert_to_spherical(self,u):
        x = u[0]
        y = u[1]
        z = u[2]
        r = sqrt(x**2 + y**2 + z**2)
        theta = arccos(z/r)
        phi = arctan2(y,x)
        return r,theta,phi
    
    
    def stereographic_projection(self,u):
        x = u[0]
        y = u[1]
        z = u[2]
        deno = x + z + sqrt(2.0)
    
        re_part = (x - z)/deno
        im_part = y*sqrt(2.0)/deno
    
        return re_part,im_part
    
    def tangent_source_half(self,v,u,s0,ds,sigma,a):
        emmu = exp(-s0[1])
        x = u[0]
        y = u[1]
        z = u[2]
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
    
        dem2erxy2_ds1 = em2erxy2*(-2.0*rxy2)
        demerxy2_ds1 = emerxy2*(-rxy2)
        dcoeff1_ds2 = -coeff1*coeff1*(r*emmu - emmu)
        dcoeff2_ds1 = -0.5*rxy/(sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)))**3.0*((x**2.0)*dem2erxy2_ds1)
    
    
        v1 = copy(v)
        v1[0] += a*z*dcoeff1_ds2*ds[1]
        v1[1] += (dcoeff1_ds2*ds[1]*coeff2 + \
                dcoeff2_ds1*ds[0]*coeff1)*(sigma*x*emerxy2*sterm + \
                y*cterm) + coeff1*coeff2*(sigma*x*demerxy2_ds1\
                *ds[0]*sterm)
    
        v1[2] += (dcoeff1_ds2*ds[1]*coeff2 + \
                dcoeff2_ds1*ds[0]*coeff1)*(-a*x*emerxy2*cterm + \
                a*sigma*y*sterm) + coeff1*coeff2* \
            (-a*x*cterm*demerxy2_ds1*ds[0])
            
    
        return v1
    
    
    
    def tangent_source(self,v,u,s,ds):
        uhalf = self.primal_halfstep(u,s,-1.,-1)
        vhalf = self.tangent_source_half(v,uhalf,s,ds,1.,1.)
        vfull = vhalf + dot(self.gradFs_halfstep(uhalf,s,1.,1.),\
                self.tangent_source_half(zeros(state_dim),\
                u,s,ds,-1,-1))
        return vfull
    
    def DFDs(self,u,s):
        param_dim = self.param_dim
        state_dim = self.state_dim
        dfds = zeros((param_dim,state_dim))
        ds1 = array([1.0, 0.0])
        ds2 = array([0.0, 1.0])
        dfds[0] = self.tangent_source(zeros(state_dim),u,s,ds1)
        dfds[1] = self.tangent_source(zeros(state_dim),u,s,ds2)
        return dfds
    
    def gradFs_halfstep(self,u,s,sigma,a):
        state_dim = self.state_dim
        emmu = exp(-s0[1])
        x = u[0]
        y = u[1]
        z = u[2]
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
    
        dem2erxy2_dx = exp(-2.0*s0[0]*rxy2)*(-2.0*s0[0])*(2.0*x)
        dem2erxy2_dy = exp(-2.0*s0[0]*rxy2)*(-2.0*s0[0])*(2.0*y)
        demerxy2_dx = exp(-s0[0]*rxy2)*(-s0[0])*(2.0*x)
        demerxy2_dy = exp(-s0[0]*rxy2)*(-s0[0])*(2.0*y)
        dterm_dz = pi*0.5*sqrt(2)
        dsterm_dz = cos(term)*dterm_dz
        dcterm_dz = -sin(term)*dterm_dz
        
        dcoeff1_dx = -coeff1*coeff1*(1.0 - emmu)*x/r
        dcoeff1_dy = -coeff1*coeff1*(1.0 - emmu)*y/r
        dcoeff1_dz = -coeff1*coeff1*(1.0 - emmu)*z/r
    
        dcoeff2_dx = x/rxy/sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)) - 0.5*rxy/(sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)))**3.0*(2.0*x*em2erxy2 + x*x*dem2erxy2_dx)
    
        dcoeff2_dy = y/rxy/sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)) - 0.5*rxy/(sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)))**3.0*(x*x*dem2erxy2_dy + 2.0*y)
    
        
        dcoeff1_ds2 = -coeff1*coeff1*(-emmu + r*emmu)
        dcoeff2_ds1 = -0.5*rxy/(sqrt((x**2.0)*em2erxy2 + \
                (y**2.0)))**3.0*(x*x*em2erxy2*(-2.0*rxy2))
    
        state_dim = u.shape[0]
        dFdu = zeros((state_dim,state_dim))
        dFdu[0,0] = dcoeff1_dx*a*z
        dFdu[0,1] = dcoeff1_dy*a*z
        dFdu[0,2] = dcoeff1_dz*a*z + coeff1*a
        
        dFdu[1,0] = (sigma*x*emerxy2*sterm + \
                y*cterm)*(dcoeff1_dx*coeff2 + \
                coeff1*dcoeff2_dx) + coeff1*coeff2*\
                (sigma*emerxy2*sterm + sigma*x*demerxy2_dx*sterm)
    
    
        dFdu[1,1] = (sigma*x*emerxy2*sterm + \
                y*cterm)*(dcoeff1_dy*coeff2 + coeff1*dcoeff2_dy) + \
                coeff1*coeff2*(sigma*x*sterm*demerxy2_dy + \
                cterm)
    
        dFdu[1,2] =  (sigma*x*emerxy2*sterm + \
                y*cterm)*(dcoeff1_dz*coeff2) + \
                coeff1*coeff2*(sigma*x*emerxy2*dsterm_dz + \
                y*dcterm_dz)
    
    
        dFdu[2,0] = (-a*x*emerxy2*cterm + \
                a*sigma*y*sterm)*(coeff1*dcoeff2_dx + \
                coeff2*dcoeff1_dx) + coeff1*coeff2* \
                (-a*emerxy2*cterm - a*x*demerxy2_dx*cterm)
    
    
        dFdu[2,1] = (-a*x*emerxy2*cterm + \
                a*sigma*y*sterm)*(coeff1*dcoeff2_dy + \
                dcoeff1_dy*coeff2) + coeff1*coeff2* \
                (-a*x*demerxy2_dy*cterm + a*sigma*sterm)
    
    
        dFdu[2,2] = dcoeff1_dz*coeff2*(-a*x*emerxy2*cterm + \
                a*sigma*y*sterm) + coeff1*coeff2*( \
                -a*x*emerxy2*dcterm_dz + a*sigma*y*dsterm_dz)
    
    
        dFdu[3,3] = 1.0
    
        return dFdu
    
    
    
    def gradFs(self,u,s):
    
        u_nphalf = self.primal_halfstep(u,s,-1,-1)
        gradFs_half = self.gradFs_halfstep(u,s,-1,-1)
        gradFs_full = self.gradFs_halfstep(u_nphalf,s,1,1)
        return dot(gradFs_full,gradFs_half)
    
    def divGradFsinv(self,u,s):
        epsi = 1.e-4
        state_dim = self.state_dim
        div_DFDu_inv = zeros(state_dim)
        #I have no better choice here.
        for i in range(state_dim):
            uplus = copy(u)
            uminus = copy(u)
            uplus[i] += epsi
            uminus[i] -= epsi
            DFDu_inv_plus = inv(self.gradFs(uplus,s))[i]
            DFDu_inv_minus = inv(self.gradFs(uminus,s))[i]
            div_DFDu_inv += (DFDu_inv_plus - DFDu_inv_minus)/ \
                    (2*epsi)
        return div_DFDu_inv
    
    
    def trace_gradDFDs_gradFsinv(self,u,s):
        epsi = 1.e-4
        DFDuinv = inv(self.gradFs(u,s))
        param_dim = self.param_dim
        state_dim = self.state_dim
        res = zeros(param_dim)
        for i in range(state_dim):
            uplus = copy(u)
            uminus = copy(u)
            uplus[i] += epsi
            uminus[i] -= epsi
            diDFDs = (DFDs(uplus,s) - DFDs(uminus,s))/(2*epsi)
            res += dot(diDFDs,DFDuinv[i])
        return res
    
    def tangent_step(self,v0,u,s,ds):
    
        v1 = dot(self.gradFs(u,s),v0) 
        v1 = self.tangent_source(v1,u,s,ds)
        return v1
    	
    
    def adjoint_step(self,w1,u,s,dJ):
    
        w0 = dot(self.gradFs(u,s).T,w1) 
        w0 += dJ
        return w0
    
    
