testName="kuznetsov"
import sys
sys.path.insert(0, '../')
from kuznetsov import *
sys.path.insert(0, '../../src')
from flow_sens import *
from matplotlib.pyplot import *
from pylab import *
from numpy import *


def plot_attractor():

        n_testpoints = 5000
        n_times = 6
        n_steps = n_poincare*100
        u0 = rand(state_dim,n_testpoints)
        u0 = (u0.T*(boundaries[state_dim:2*state_dim]-boundaries[0:state_dim])).T
        u0 = (u0.T + boundaries[0:state_dim]).T
        u = zeros((state_dim,n_testpoints,n_times))
        for j in arange(n_times):
                u[:,:,j] = copy(u0)
        subplot(331)
        plot(u0[0,:],u0[1,:],"o")
        for i in arange(n_testpoints):
                u[:,i,1] = primal_step(u[:,i,1],s0,n_steps)
                for n in arange(1,n_times):
                        u[:,i,n] = primal_step(u[:,i,n-1],s0,n_steps)
        
        subplot(332)
        plot(u[0,:,1],u[1,:,1],"o")
                
        subplot(333)
        plot(u[0,:,2],u[1,:,2],"o")

        subplot(334)
        plot(u[0,:,3],u[1,:,3],"o")

        subplot(335)
        plot(u[0,:,4],u[1,:,4],"o")
                
        subplot(336)
        plot(u[0,:,5],u[1,:,5],"o")


        r = zeros(n_testpoints)
        theta = zeros(n_testpoints)
        phi = zeros(n_testpoints)
        x = zeros(n_testpoints)
        y = zeros(n_testpoints)
        for i in range(n_testpoints):
                r[i],theta[i],phi[i] = convert_to_spherical(u[:,i,5]) 
                x[i],y[i] = stereographic_projection(u[:,i,5])
        figure()                
        subplot(121)                
        plot(phi,theta,"ko")
        xlim([-pi,pi])
        ylim([0.,pi])
        subplot(122)
        plot(x,y,"ro")


def visualize_primal():
    u_init = rand(state_dim)
    u_init[3] = 0.0
    u_init = primal_step(u_init,s0,n_runup)
    n_steps = int(T/dt)*1000
    u = solve_primal(u_init,n_steps,s0)
    u = u[::int(T/dt)]
    stereo_real, stereo_imag = stereographic_projection(u.T)
    figure()
    plot(stereo_real, stereo_imag, '.', ms=1)
    savefig('st_proj_attractor.png')

def extrapolate(a0, a1, multiplier):
    return a0 + (a1 - a0) * multiplier

def visualize_tangent(u, v):
    EPS = 1E-8
    u_plus, u_minus = u + v * EPS, u - v * EPS
    stereo_real, stereo_imag = stereographic_projection(u.T)
    stereo_real_plus, stereo_imag_plus = stereographic_projection(u_plus.T)
    stereo_real_minus, stereo_imag_minus = stereographic_projection(u_minus.T)
    stereo_real_plus = extrapolate(stereo_real, stereo_real_plus, 1E6)
    stereo_real_minus = extrapolate(stereo_real, stereo_real_minus, 1E6)
    stereo_imag_plus = extrapolate(stereo_imag, stereo_imag_plus, 1E6)
    stereo_imag_minus = extrapolate(stereo_imag, stereo_imag_minus, 1E6)
    plot([stereo_real_plus, stereo_real_minus],
         [stereo_imag_plus, stereo_imag_minus], '-k', ms=1)

visualize_tangent(u[::int(T/dt)], v0[::int(T/dt)])



def test_tangent():

        n_testpoints = 100
        n_epsi = 8
        
        u0 = rand(n_testpoints,state_dim)
        epsi = logspace(-n_epsi,-1.0,n_epsi)
        vu_fd = zeros((n_epsi,n_testpoints,state_dim))
        vs_fd = zeros((n_epsi,n_testpoints,state_dim))
        vu_ana = zeros((n_testpoints,state_dim))
        vs_ana = zeros((n_testpoints,state_dim))
        u0next = zeros(state_dim)
        v0 = rand(4)
        ds0 = array([1.,1.])
        for i in arange(n_testpoints):
                u0[i] = primal_step(u0[i],s0,n_poincare)        
                for k in arange(n_epsi):                
                        vu_fd[k,i] = (primal_step(u0[i] + epsi[k]*v0,
                            s0,1)-\
                            primal_step(u0[i] - epsi[k]*v0,s0,1)\
                            )/(2.0*epsi[k])

                        vs_fd[k,i] = (primal_step(u0[i],s0 + epsi[k]*ds0,1) - 
                            primal_step(u0[i],s0 - epsi[k]*ds0,1)) \
                                    /(2.0*epsi[k])


                vu_ana[i] = tangent_step(v0,u0[i],s0,zeros(param_dim))
                vs_ana[i] = tangent_step(zeros(state_dim),u0[i],s0,ds0)

        erru = zeros(n_epsi)
        errs = zeros(n_epsi)

        for k in arange(n_epsi):
                erru[k] = norm(vu_ana-vu_fd[k])
                errs[k] = norm(vs_ana-vs_fd[k])

        figure()
        loglog(epsi,erru, 'o-')
        savefig('erru')
        figure()
        loglog(epsi,errs, 'o-')
        savefig('errs')





def test_jacobian():
    u0 = rand(state_dim)
    u0[3] *= T
    epsi = 1.e-8
    Jacu = zeros((state_dim,state_dim))
    Jacu[:,0] = ((primal_step(u0 + epsi*array([1.0,0.0,0.0,0.0]),s0,1) - 
                            primal_step(u0 - epsi*array([1.0,0.0,0.0,0.0]),s0,1))/
                            (2.0*epsi))

    Jacu[:,1] = ((primal_step(u0 + epsi*array([0.0,1.0,0.0,0.0]),s0,1) - 
                            primal_step(u0 - epsi*array([0.0,1.0,0.0,0.0]),s0,1))/
                            (2.0*epsi))

    Jacu[:,2] = ((primal_step(u0 + epsi*array([0.0,0.0,1.0,0.0]),s0,1) - 
                            primal_step(u0 - epsi*array([0.0,0.0,1.0,0.0]),s0,1))/
                            (2.0*epsi))

    Jacu[:,3] = ((primal_step(u0 + epsi*array([0.0,0.0,0.0,1.0]),s0,1) - 
                            primal_step(u0 - epsi*array([0.0,0.0,0.0,1.0]),s0,1))/
                            (2.0*epsi))

    dFds1 = (primal_step(u0,s0 + epsi*array([1.0,0.0]),1)-primal_step(u0,s0
                    - epsi*array([1.0,0.0]),1))/(2.0*epsi)        


    dFds2 = (primal_step(u0,s0 + epsi*array([0.0,1.0]),1)-primal_step(u0,s0
                    - epsi*array([0.0,1.0]),1))/(2.0*epsi)        

    Jacana = dt*gradfs(u0,s0) + eye(state_dim,state_dim)
    print(norm(Jacu-Jacana))
    print(Jacu)
        
    v0 = rand(4)
    v0_fd = dot(Jacu,v0) 
    print(v0_fd)
    v0_hand = tangent_step(v0,u0,s0,zeros(2))
    print(norm(v0_fd - v0_hand))


    v1_fd = v0_fd + dFds1 
    v1_hand = tangent_step(v0,u0,s0,[1.0,0.0])
    print(norm(v1_fd - v1_hand))

    v2_fd = v0_fd + dFds2 
    v2_hand = tangent_step(v0,u0,s0,[0.0,1.0])
    print(norm(v2_fd - v2_hand))



def test_adjoint():

        u0 = rand(4)
        u0[3] *= T
        u1 = primal_step(u0,s0,1)
        epsi = 1.e-8

        y1 = [0.0,1.,0.,0.]
        y0_ana = adjoint_step(y1,u0,s0,y1)

        y0_fd = zeros(state_dim)
        v0 = zeros(state_dim)
        for i in arange(state_dim):
                v0 = zeros(state_dim)
                v0[i] = 1.0
                u0pert = u0 + epsi*v0
                u1pert =  primal_step(u0pert,s0,1)
                obj2 = u0pert[1] + u1pert[1]

                u0pert = u0 - epsi*v0
                u1pert =  primal_step(u0pert,s0,1)
                obj1 = u0pert[1] + u1pert[1]

                y0_fd[i] = (obj2 - obj1)/(2.0*epsi) 
        print(norm(y0_fd-y0_ana))



def test_tangentadjoint():
        u = rand(4)
        u[3] *= T
        y1 = rand(4)
        v0 = rand(4)
        v1 = tangent_step(v0,u,s0,zeros(2))
        y0 = adjoint_step(y1,u,s0,zeros(4))
        print(dot(v1,y1))
        print(dot(v0,y0))


def test_objective():
#if __name__=="__main__":
    nsamples = 10000
    u = rand(state_dim,nsamples)*2.0 + -1.0
    u[3,:] = rand(nsamples)*T
    ntheta = 20
    nphi = 20
    dtheta = pi/(ntheta-1)
    dphi = 2*pi/(nphi-1)
    theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,ntheta)
    phi_bin_centers = linspace(-pi+dphi/2,pi-dphi/2.,nphi)
    J = zeros((ntheta,nphi))
    #for k in arange(nsamples):
       # for i in arange(ntheta):
        #    for j in arange(nphi):
         #       J[i,j] += objective(u[:,k],s0,theta_bin_centers[i],dtheta,
          #                  phi_bin_centers[j],dphi)/nsamples

  #  contourf(phi_bin_centers,theta_bin_centers,J)
   # colorbar()
    N = 500
    theta_grid = linspace(0.,pi,N)
    phi_grid = linspace(-pi,pi,N)
    r_grid = linspace(0.0,1.0,N)
    st_grid = sin(theta_grid)
    ct_grid = cos(theta_grid)
    sp_grid = sin(phi_grid)
    cp_grid = cos(phi_grid)

    x_grid = r_grid*st_grid*cp_grid
    y_grid = r_grid*st_grid*sp_grid
    z_grid = r_grid*ct_grid
    J1 = zeros(N)
    J2 = zeros(N)
    dJ1 = zeros((state_dim,N))
    dJ2 = zeros((state_dim,N))
    dJ1dphi = zeros(N)
    for i in arange(N):
        u_grid = array([x_grid[i],y_grid[i],z_grid[i],2.0])
        dxdphi = r_grid[i]*st_grid[i]*(-sp_grid[i])
        dydphi = r_grid[i]*st_grid[i]*cp_grid[i]
        J1[i] = objective(u_grid,s0,dtheta/2.0,dtheta,-pi+dphi/2.0,dphi)
        J2[i] = objective(u_grid,s0,pi-dtheta/2.0,dtheta,pi-dphi/2.0,dphi)
        dJ1[:,i] = Dobjective(u_grid,s0,dtheta/2.0,dtheta,-pi+dphi/2.0,dphi)
        dJ1dphi[i] = dJ1[0,i]*dxdphi + dJ1[1,i]*dydphi
        dJ2[:,i] = Dobjective(u_grid,s0,pi-dtheta/2.0,dtheta,pi-dphi/2.0,dphi)

    plot(theta_grid,J1,"o-",theta_grid,J2,"o-")
    figure()
    plot(theta_grid,norm(dJ1,axis=0),"o-",theta_grid,norm(dJ2,axis=0),"o-")
    figure()
    plot(phi_grid,dJ1dphi,"o-")
    #show()
