testName="kuznetsov"
import sys
sys.path.insert(0, '../')
from kuznetsov import *
sys.path.insert(0, '../../src')
from flow_sens import *
from matplotlib.pyplot import *
from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
n_runup = 100*int(T/dt)
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

def visualize_poincare_primal():
    u_init = rand(state_dim)
    u_init = poincare_step(u_init,s0,n_runup)
    n_steps = 10000
    u = solve_poincare_primal(u_init,n_steps,s0)
    stereo_real, stereo_imag = stereographic_projection(u.T)
    figure()
    plot(stereo_real, stereo_imag, '.', ms=4)
    savefig('st_proj_poincare_attractor.png')


@jit(nopython=True)
#if __name__ == "__main__":
def test_poincare_primal():
    u_init = rand(state_dim)
    u_init = poincare_step(u_init,s0,n_runup)
    u_init[-1] = 0.
    n_steps = 1000
    u = solve_poincare_primal(u_init,n_steps,s0)
    u_ode = solve_primal(u_init,n_steps*int(T/dt),s0)
    print(norm(u_ode[::int(T/dt)]-u))


@jit(nopython=True)
#if __name__ == "__main__":
def test_step_primal():
    holes = array([1./sqrt(2.),0,1./sqrt(2.0),\
                   -1./sqrt(2.),0,1./sqrt(2.0), \
                    1./sqrt(2.),0,-1./sqrt(2.0), \
                    -1./sqrt(2.),0,-1./sqrt(2.0)]) 
    holes = holes.reshape(4,3)

    flag = 1
    u_init = rand(state_dim)
    u_init[3] = 0.0
    u_init = primal_step(u_init,s0,n_runup)
    fac = 1000
    n_steps = int(T/dt)*fac
    u = solve_primal(u_init,n_steps,s0)
    for i in range(1,n_steps):
        if(dot((u[i,:3]-u[i-1,:3])/dt,u[i-1,:3])>1.e-8):
            break
    assert(i==n_steps-1)
            
    u = u[::int(T/dt)]
    
    epsi = 1.e-1
    for i in range(1,fac):
        if(not(flag)):
            break
        for j in range(4):
            if(norm(u[i,:3]-holes[j])<epsi):
                flag = 0
            if(norm(u[i,:3])>1.0 + epsi):
                flag = 0
    assert(i==fac-1)
    
def extrapolate(a0, a1, multiplier):
    return a0 + (a1 - a0) * multiplier

def visualize_tangent_stereographic(u, v):
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

def visualize_tangent_2D(u, v):
    EPS = 1E-8
    u = u.T
    v = v.T
    u_plus, u_minus = u + v * EPS, u - v * EPS
    phi = arctan2(u[1],u[0])
    theta = arccos(u[2],norm(u,axis=0))
    phi_plus = arctan2(u_plus[1],u_plus[0])
    phi_minus = arctan2(u_minus[1],u_minus[0])
    theta_plus = arccos(u_plus[2],norm(u_plus,axis=0))
    theta_minus = arccos(u_minus[2],norm(u_minus,axis=0))

    phi_plus = extrapolate(phi, phi_plus, 1E6)
    phi_minus = extrapolate(phi, phi_minus, 1E6)
    theta_plus = extrapolate(theta, theta_plus, 1E6)
    theta_minus = extrapolate(theta, theta_minus, 1E6)
    plot([phi_plus, phi_minus],
         [theta_plus, theta_minus], '-k', ms=1)

#visualize_tangent(u[::int(T/dt)], v0[::int(T/dt)])
def visualize_tangent_2D(u, v):
    EPS = 1E-8
    u = u.T
    v = v.T
    u_plus, u_minus = u + v * EPS, u - v * EPS
    phi = arctan2(u[1],u[0])
    theta = arccos(u[2],norm(u,axis=0))
    phi_plus = arctan2(u_plus[1],u_plus[0])
    phi_minus = arctan2(u_minus[1],u_minus[0])
    theta_plus = arccos(u_plus[2],norm(u_plus,axis=0))
    theta_minus = arccos(u_minus[2],norm(u_minus,axis=0))

    phi_plus = extrapolate(phi, phi_plus, 1E6)
    phi_minus = extrapolate(phi, phi_minus, 1E6)
    theta_plus = extrapolate(theta, theta_plus, 1E6)
    theta_minus = extrapolate(theta, theta_minus, 1E6)
    plot([phi_plus, phi_minus],
         [theta_plus, theta_minus], '-k', ms=1)


def visualize_tangent_3D(u, v):
    EPS = 1E-8
    u0 = copy(u).T
    u = u[::500]
    v = v[::500]

    u = u.T
    v = v.T
    r = 1
    phi, theta = mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.1, linewidth=0)
    ax.quiver(u[0],u[1],u[2],v[0], \
            v[1],v[2],color='k',length=0.025)
    ax.scatter(u0[0],u0[1],u0[2],color='gray')
    pointA = array([-1/sqrt(2.0), -1/sqrt(2.0), 0])
    pointB = array([-1/sqrt(2.0), 1/sqrt(2.0), 0])
    pointC = array([1/sqrt(2.0), -1/sqrt(2.0), 0])
    pointD = array([1/sqrt(2.0), 1/sqrt(2.0), 0])
    ax.scatter([pointA[0],pointB[0],pointC[0],pointD[0]],\
            [pointA[1],pointB[1],pointC[1],pointD[1]],\
            [pointA[2],pointB[2],pointC[2],pointD[2]],\
                color='red',s=50.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    savefig("plykin_inverse_adjoint_3d",dpi=500)


    #plot([u_plus[0], u_minus[0]], \
     #    [u_plus[1], u_minus[1]],[u_plus[2],u_minus[2]],'-k', ms=1)

def visualize_field_density_3D(u, v):
    EPS = 1E-8
    u0 = copy(u)
    rhou0, = histogramdd(u0,bins = (100,100,100))
    


    u = u[::500]
    v = v[::500]

    u = u.T
    v = v.T
    u_plus, u_minus = u + v * EPS, u - v * EPS
    
    u_plus = extrapolate(u, u_plus, 1E6)
    u_minus = extrapolate(u, u_minus, 1E6)
    r = 1
    phi, theta = mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.1, linewidth=0)
    ax.quiver(u[0],u[1],u[2],u_plus[0], \
            u_plus[1],u_plus[2],color='k',length=0.25)
    ax.scatter(u0[0],u0[1],u0[2],color='gray')
    pointA = array([-1/sqrt(2.0), -1/sqrt(2.0), 0])
    pointB = array([-1/sqrt(2.0), 1/sqrt(2.0), 0])
    pointC = array([1/sqrt(2.0), -1/sqrt(2.0), 0])
    pointD = array([1/sqrt(2.0), 1/sqrt(2.0), 0])
    ax.scatter([pointA[0],pointB[0],pointC[0],pointD[0]],\
            [pointA[1],pointB[1],pointC[1],pointD[1]],\
            [pointA[2],pointB[2],pointC[2],pointD[2]],\
                color='red',s=50.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    savefig("plykin_inverse_adjoint_3d",dpi=1000)

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
        assert(min(erru) < 1.e-5)
        assert(min(errs) < 1.e-5)




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


#if __name__ == "__main__":
def test_poincare_halfstep_Jacobian():
    n_test = 100
    state_dim = 4
    u = rand(n_test,state_dim)
    u[:,3] = 0.0
    n_epsi = 10
    epsi = logspace(-n_epsi,-1.0,n_epsi)
    dFdu_fd = zeros((n_epsi,n_test,state_dim,state_dim))
    dFdu = zeros((n_test,state_dim,state_dim))
    for i in range(n_test):
        dFdu[i] = gradFs_poincare_halfstep(u[i],s0,-1,-1)
        for j in range(n_epsi):
            for k in range(state_dim):
                v = zeros(state_dim)
                v[k] = 1.0
                dFdu_fd[j,i,:,k] = (poincare_half_step(u[i] + epsi[j]*v,s0,-1,-1) \
                        - poincare_half_step(u[i] - epsi[j]*v,s0,-1,-1))/(2*epsi[j])
                
    err_poincare_jacobian = zeros(n_epsi)
    for j in range(n_epsi):
        err_poincare_jacobian[j] = norm(dFdu_fd[j]-dFdu)
    figure()
    loglog(epsi,err_poincare_jacobian)
    xlabel(r"$\epsilon$")
    ylabel("Error in Jacobian of Poincare half step")
    savefig("errJacobianHalfPoincare")
    assert(min(err_poincare_jacobian) < 1.e-8)


#if __name__ == "__main__":
def test_poincare_Jacobian():
    n_test = 100
    state_dim = 4
    u = rand(n_test,state_dim)
    u[:,3] = 0.0
    n_epsi = 10
    epsi = logspace(-n_epsi,-1.0,n_epsi)
    dFdu_fd = zeros((n_epsi,n_test,state_dim,state_dim))
    dFdu = zeros((n_test,state_dim,state_dim))
    for i in range(n_test):
        dFdu[i] = gradFs_poincare(u[i],s0)
        for j in range(n_epsi):
            for k in range(state_dim):
                v = zeros(state_dim)
                v[k] = 1.0
                dFdu_fd[j,i,:,k] = (poincare_step(u[i] + epsi[j]*v,s0) \
                        - poincare_step(u[i] - epsi[j]*v,s0))/(2*epsi[j])
                
    err_poincare_jacobian = zeros(n_epsi)
    for j in range(n_epsi):
        err_poincare_jacobian[j] = norm(dFdu_fd[j]-dFdu)
    figure()
    loglog(epsi,err_poincare_jacobian)
    xlabel(r"$\epsilon$")
    ylabel("Error in Jacobian of Poincare step")
    savefig("errJacobianPoincare")
    assert(min(err_poincare_jacobian) < 1.e-8)

        

def test_DfDs():
    u0 = rand(4)
    u0[3] *= T
    n_epsi = 10
    param_dim = s0.size
    epsi = logspace(-n_epsi,-1.0,n_epsi)
    dfds_fd = zeros((n_epsi,param_dim,state_dim))
    dfds_ana = zeros((param_dim,state_dim))
    for i in range(param_dim):
        for k in range(n_epsi):
            splus = copy(s0)
            splus[i] += epsi[k]
            sminus = copy(s0)
            sminus[i] -= epsi[k]

            dfds_fd[k,i] = (primal_step(u0,splus) - 
                            primal_step(u0,sminus))/(2.0*epsi[k])/dt
    err_dfds = zeros(n_epsi)
    dfds_ana = DfDs(u0,s0)
    for k in range(n_epsi):
        err_dfds[k] = norm(dfds_fd[k]-dfds_ana)
    figure()
    loglog(epsi,err_dfds, 'o-')
    savefig('err_dfds')

    assert(min(err_dfds) < 1.e-8)

#if __name__ == "__main__":
def test_poincare_DFDs():
    u0 = rand(4)
    u0[3] *= 0.
    n_epsi = 10
    param_dim = s0.size
    epsi = logspace(-n_epsi,-1.0,n_epsi)
    dfds_fd = zeros((n_epsi,param_dim,state_dim))
    dfds_ana = zeros((param_dim,state_dim))
    for i in range(param_dim):
        for k in range(n_epsi):
            splus = copy(s0)
            splus[i] += epsi[k]
            sminus = copy(s0)
            sminus[i] -= epsi[k]

            dfds_fd[k,i] = (poincare_step(u0,splus) - 
                            poincare_step(u0,sminus))/(2.0*epsi[k])
    err_dfds = zeros(n_epsi)
    dfds_ana = DFDs_poincare(u0,s0)
    for k in range(n_epsi):
        err_dfds[k] = norm(dfds_fd[k]-dfds_ana)
    figure()
    loglog(epsi,err_dfds, 'o-')
    savefig('err_dfds_poincare')

    assert(min(err_dfds) < 1.e-8)



#if __name__=="__main__":
def test_divGradfs():

    n_samples = 100
    n_epsi = 10
    epsi = logspace(-n_epsi,-1,n_epsi)
    u0 = rand(n_samples,state_dim)
    u0 *= (boundaries[1]-boundaries[0])
    u0 += boundaries[0]
    dgf_hand = zeros((n_samples,state_dim))
    dgf_fd = zeros((n_epsi,n_samples,state_dim))
    tmp_matrix = zeros((state_dim,state_dim))
    for i in range(n_samples):
        dgf_hand[i] = divGradfs(u0[i],s0) 
        for k in range(n_epsi):
            for j in range(state_dim):
                v0 = zeros(state_dim)
                v0[j] = 1.0
                tmp_matrix = (gradfs(u0[i]+epsi[k]*v0,s0) -
                              gradfs(u0[i]-epsi[k]*v0,s0))/(2.0*epsi[k])
                dgf_fd[k,i] += tmp_matrix[j]
        
    err = zeros(n_epsi)
    for k in range(n_epsi):
        err[k] = norm(dgf_hand-dgf_fd[k]) 
    figure()
    loglog(epsi,err,'o-')
    ylabel(r'$|| \nabla\cdot(\nabla f_s) -\nabla\cdot(\nabla f_s)_\epsilon^{\rm FD}||$')
    xlabel(r'$\epsilon$')
    savefig('err_divGradfs')

    assert(min(err)<1.e-8)


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
        print(u[3])
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

def test_gradient_objective():
    n_test = 100
    n_epsi = 10
    n_theta = 20
    n_phi = 20
    dtheta = pi/(n_theta -1)
    dphi = 2*pi/(n_phi -1)
    theta0 = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
    phi0 = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)
    u0 = rand(n_test,state_dim)
    epsi = logspace(-n_epsi,-1,n_epsi)
    dJdu_ana = zeros((n_test,n_theta,n_phi,state_dim))
    dJdu_fd = zeros((n_epsi,n_test,n_theta,n_phi,state_dim))
    
    for p in range(n_test):
        for i in range(n_theta):
            for j in range(n_phi):
                dJdu_ana[p,i,j] = Dobjective(u0[p],\
                        s0,theta0[i],dtheta,phi0[j],dphi)
                for k in range(n_epsi):
                    for l in range(state_dim):
                        v0 = zeros(state_dim)
                        v0[l] = 1.0
                        dJdu_fd[k,p,i,j,l] = (objective(u0[p]+epsi[k]*v0,\
                            s0,theta0[i],dtheta,phi0[j],dphi)- \
                            objective(u0[p]-epsi[k]*v0,s0,theta0[i],dtheta,\
                            phi0[j],dphi))/(2.0*epsi[k])


    err_dJdu = zeros(n_epsi)
    for k in range(n_epsi):
        err_dJdu[k] = norm(dJdu_fd[k]-dJdu_ana)
    loglog(epsi,err_dJdu,"o")
    xlabel(r"$\epsilon$")
    ylabel("Error in the gradient of objective")
    savefig("err_dJdu")
    assert(max(err_dJdu) < 1.e-1)
    assert(min(err_dJdu) < 1.e-6)



