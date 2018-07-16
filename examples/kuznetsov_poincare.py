from pylab import *
from numpy import *
from numba import jit


class Solver:

    dt = 2.e-3
    s0 = array([1.0,1.0])
    T = 6.0
    boundaries = array([[-1, 1],
                        [-1, 1],
                        [-1, 1],
                        [0, T]]).T
    state_dim = boundaries.shape[1]
    n = int(ceil(T/dt))
    
    @jit(nopython=True)
    def primal_step(u0,s,n=1):
        state_dim= u0.size
        param_dim= s.size
        u = copy(u0)
        for i in range(n):
            u = primal_halfstep(u,s,-1.,-1.)
            u = primal_halfstep(u,s,1.,1.)
        u[3] = u0[3]
        return u
    
    @jit(nopython=True)
    def primal_halfstep(u,s0,sigma,a):
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
    
        u1 = copy(u)
        u1[0] = coeff1*a*z
        u1[1] = coeff1*coeff2*(sigma*x*emerxy2*sterm + \
                y*cterm)
        u1[2] = coeff1*coeff2*(-a*x*emerxy2*cterm + \
                a*sigma*y*sterm)
        u1[3] = (u[3] + T/2.0)%T
    
        return u1

    @jit(nopython=True)
    def objective(u,s,theta0,dtheta,phi0,dphi):
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
    
    @jit(nopython=True)
    def Dobjective(u,s,theta0,dtheta,phi0,dphi):
    
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
            res[l] = (objective(u+epsi*v0,\
                    s,theta0,dtheta,phi0,dphi)- \
                    objective(u-epsi*v0,s,theta0,dtheta,\
                    phi0,dphi))/(2.0*epsi)
    
    
        return res
    
    def convert_to_spherical(u):
        x = u[0]
        y = u[1]
        z = u[2]
        r = sqrt(x**2 + y**2 + z**2)
        theta = arccos(z/r)
        phi = arctan2(y,x)
        return r,theta,phi
    
    @jit(nopython=True)
    def stereographic_projection(u):
        x = u[0]
        y = u[1]
        z = u[2]
        deno = x + z + sqrt(2.0)
    
        re_part = (x - z)/deno
        im_part = y*sqrt(2.0)/deno
    
        return re_part,im_part
    @jit(nopython=True)
    def tangent_source_half(v,u,s0,ds,sigma,a):
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
    
    
    @jit(nopython=True)
    def tangent_source(v,u,s,ds):
        uhalf = primal_halfstep(u,s,-1.,-1)
        vhalf = tangent_source_half(v,uhalf,s,ds,1.,1.)
        vfull = vhalf + dot(gradFs_halfstep(uhalf,s,1.,1.),\
                tangent_source_half(zeros(state_dim),\
                u,s,ds,-1,-1))
        return vfull
    @jit(nopython=True)
    def DFDs(u,s):
        param_dim = s.size
        dfds = zeros((param_dim,state_dim))
        ds1 = array([1.0, 0.0])
        ds2 = array([0.0, 1.0])
        dfds[0] = tangent_source(zeros(state_dim),u,s,ds1)
        dfds[1] = tangent_source(zeros(state_dim),u,s,ds2)
        return dfds
    @jit(nopython=True)
    def gradFs_halfstep(u,s,sigma,a):
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
    
    
    @jit(nopython=True)
    def gradFs(u,s):
    
        u_nphalf = primal_halfstep(u,s,-1,-1)
        gradFs_half = gradFs_halfstep(u,s,-1,-1)
        gradFs_full = gradFs_halfstep(u_nphalf,s,1,1)
        return dot(gradFs_full,gradFs_half)
    @jit(nopython=True)
    def divGradFsinv(u,s):
        epsi = 1.e-4
        div_DFDu_inv = zeros(state_dim)
        #I have no better choice here.
        for i in range(state_dim):
            uplus = copy(u)
            uminus = copy(u)
            uplus[i] += epsi
            uminus[i] -= epsi
            DFDu_inv_plus = inv(gradFs(uplus,s))[i]
            DFDu_inv_minus = inv(gradFs(uminus,s))[i]
            div_DFDu_inv += (DFDu_inv_plus - DFDu_inv_minus)/ \
                    (2*epsi)
        return div_DFDu_inv
    
    @jit(nopython=True)
    def trace_gradDFDs_gradFsinv(u,s):
        epsi = 1.e-4
        DFDuinv = inv(gradFs(u,s))
        param_dim = s.size
        res = zeros(param_dim)
        for i in range(state_dim):
            uplus = copy(u)
            uminus = copy(u)
            uplus[i] += epsi
            uminus[i] -= epsi
            diDFDs = (DFDs(uplus,s) - DFDs(uminus,s))/(2*epsi)
            res += dot(diDFDs,DFDuinv[i])
        return res
    @jit(nopython=True)
    def tangent_step(v0,u,s,ds):
    
        v1 = dot(gradFs(u,s),v0) 
        v1 = tangent_source(v1,u,s,ds)
        return v1
    	
    @jit(nopython=True)
    def adjoint_step(w1,u,s,dJ):
    
        w0 = dot(gradFs(u,s).T,w1) 
        w0 += dJ
        return w0
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
        '''
    
    
    @jit(nopython=True)
    def diff_rot_freq(t):
        a0 = -1.0
        a1 = 0.0
        a2 = 1.0
        c0 = 1.0
        c1 = 2.0
        c2 = 4.0
        c3 = 5.0 
    
        
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
        '''
    
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
 
