#Statistical sensitivity analysis algorithm for maps.
from __future__ import division
from __future__ import print_function
import sys
import pdb

sys.path.insert(0, '../examples/')
from kuznetsov import *
sys.path.insert(0, '../examples/tests')
from test_kuznetsov import *

from pylab import *
from numpy import *
from time import clock
from util import *

@jit(nopython=True)
def solve_poincare_primal(u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[0] = u_init
    for i in range(1,n_steps):
        u[i] = poincare_step(u[i-1],s)
    return u

@jit(nopython=True)
def solve_poincare_unstable_direction(u, v_init, n_steps, s, ds):
    v = empty((n_steps, v_init.size))
    v[0] = v_init
    for i in range(1,n_steps):
        v[i] = tangent_poincare_step(v[i-1],u[i-1],s,ds)
        v[i] /= linalg.norm(v[i])
    return v

@jit(nopython=True)
def solve_poincare_unstable_adjoint_direction(u, w_init, n_steps, s, dJ):
    w = empty((n_steps, w_init.size))
    w[-1] = w_init
    for i in range(n_steps-1,0,-1):
        w[i-1] = adjoint_poincare_step(w[i],u[i-1],s,dJ)
        w[i-1] /= norm(w[i-1])
    return w


@jit(nopython=True)
def compute_poincare_source_tangent(u, n_steps, s0):
    param_dim = s0.size
    dFds = zeros((n_steps,param_dim,state_dim))
    for i in range(n_steps):
        dFds[i] = DFDs_poincare(u[i],s0)
    return dFds

@jit(nopython=True)
def compute_poincare_source_inverse_adjoint(u, n_steps, s0):
    dgf = zeros((n_steps,state_dim))
    for i in range(n_steps):
        dgf[i] = divGradFsinv_poincare(u[i],s0)
    return dgf

@jit(nopython=True)
def compute_poincare_source_sensitivity(u, n_steps, s0):
    param_dim = s0.size
    t_ddFds_dFinv = zeros((n_steps,param_dim))
    for i in range(n_steps):
        t_ddFds_dFinv[i] = trace_gradDFDs_gradFsinv(u[i],s0)
    return t_ddFds_dFinv


@jit(nopython=True)
def compute_objective(u,s0,n_steps,n_theta=25,n_phi=25):
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
    phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)
    J_theta_phi = zeros((n_steps,n_theta,n_phi))
    for i in arange(n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                J_theta_phi[i,t_ind,p_ind] += objective(u[i],
                        s0,theta0,dtheta,phi0,dphi)
    return J_theta_phi 

@jit(nopython=True)
def preprocess_objective(J_theta_phi):
    n_steps = J_theta_phi.shape[0]
    integral_J = zeros(J_theta_phi.shape)
    integral_J[-1] = copy(J_theta_phi[-1])
    for i in range(n_steps-1,0,-1):
        integral_J[i-1] = integral_J[i] + J_theta_phi[i-1]
    return integral_J


@jit(nopython=True)
def compute_gradient_objective(u,s0,n_steps,n_theta=25,n_phi=25):
    dtheta = pi/(n_theta-1)
    dphi = 2*pi/(n_phi-1)
    theta_bin_centers = linspace(dtheta/2.0,pi-dtheta/2.0,n_theta)
    phi_bin_centers = linspace(-pi+dphi/2.0,pi-dphi/2.0,n_phi)

    DJ_theta_phi = zeros((n_steps,n_theta,n_phi,state_dim))
    for i in arange(n_steps):
        for t_ind, theta0 in enumerate(theta_bin_centers):
            for p_ind, phi0 in enumerate(phi_bin_centers):
                DJ_theta_phi[i,t_ind,p_ind] += Dobjective(u[i],
                        s0,theta0,dtheta,phi0,dphi)
    return DJ_theta_phi 


@jit(nopython=True)
def compute_sensitivity(u,s,v0,w0,J,dJ,dFds,dJds_0,N,Ninf):
    g = zeros(Ninf)
    w_inv = zeros((n_steps,state_dim))
    v = zeros(state_dim)
    n_points_theta = J.shape[1]
    n_points_phi = J.shape[2]
    gsum_mean = 0.0
    gsum_history = zeros(n_steps)
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dFds_unstable = zeros(state_dim)
    n_samples = N - 2*Ninf
    for n in range(n_steps-1):
        b = dJds_0[n]
        q = source_inverse_adjoint[n]
        nablaFs = gradFs_poincare(u[n],s)   
        w_inv[n+1] = -1.0*q + solve(nablaFs.T, w_inv[n])
        w_inv[n+1] = dot(w_inv[n+1], v0[n+1]) * v0[n+1]
        g[n % Ninf] = dot(w_inv[n+1],dFds_unstable) - b   
        gsum_mean += sum(g)
        v = dot(nablaFs,v) + dFds[n]
        gsum_history[n] = sum(g)
        v, dFds_unstable = decompose_tangent(v, v0[n+1], w0[n+1])
        if(n>=2*Ninf):
            for binno_t in range(n_points_theta):
                for binno_p in range(n_points_phi):
                    dJds_unstable[binno_t,binno_p] += \
                            J[n+1,binno_t,binno_p]*(sum(g))/n_samples
                    dJds_stable[binno_t,binno_p] += \
                            dot(dJ[n+1,binno_t,binno_p], v)/n_samples
    return dJds_stable, dJds_unstable, w_inv	

if __name__ == "__main__":
#def compute_sensitivity()

    Ninf = 10
    n_adjoint_converge = 10
    n_samples = 100000
    n_runup = 100
    n_steps = n_samples + Ninf +\
            n_adjoint_converge + 1 
    n_points_theta = 20
    n_points_phi = 20
    dtheta = pi/(n_points_theta-1)
    dphi = 2*pi/(n_points_phi-1)
    u = zeros((n_steps,state_dim))
    random.seed(0)
    u_init = rand(state_dim)
    u_init[3] = 0
    u_init = poincare_step(u_init,s0,n_runup)
    param_dim = s0.size
    ds0 = zeros(param_dim)
    ds1 = copy(ds0)
    ds1[0] = 1.0
    dJ0 = zeros(state_dim)
    
    w0 = zeros((n_steps,state_dim))
    w0_init = rand(state_dim)
    w0_init /= linalg.norm(w0_init)
    
    v0_init = rand(state_dim)
    v0_init /= linalg.norm(v0_init)
    
    J_theta_phi = zeros((n_steps,n_points_theta,n_points_phi))
    DJ_theta_phi = zeros((n_steps,n_points_theta,n_points_phi,state_dim))  
    theta_bin_centers = linspace(dtheta/2.0,pi - dtheta/2.0, n_points_theta)
    phi_bin_centers = linspace(-pi-dphi/2.0,pi - dphi/2.0, n_points_phi)
    v = zeros(state_dim)
    w_inv = zeros(state_dim)
    dfds = zeros((n_steps,param_dim,state_dim))
    source_inverse_adjoint = zeros((n_steps,state_dim))
    source_tangent = zeros((n_steps,param_dim,state_dim))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    divdfds = zeros(n_steps)

    t0 = clock()
    u = solve_poincare_primal(u_init, n_steps, s0)
    t1 = clock()
    v0 = solve_poincare_unstable_direction(u, v0_init, n_steps, s0, ds0)
    t2 = clock()
    source_tangent = compute_poincare_source_tangent(u,n_steps,s0)[:,0,:] 
    t3 = clock()
    J_theta_phi = compute_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t4 = clock()
    DJ_theta_phi = compute_gradient_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    t5 = clock()
    source_inverse_adjoint = compute_poincare_source_inverse_adjoint(u,n_steps,s0)
    t6 = clock()
    unstable_sensitivity_source = (compute_poincare_source_sensitivity(u,n_steps,s0))
    t7 = clock()
    w0 = solve_poincare_unstable_adjoint_direction(u, w0_init, n_steps, s0, dJ0)
    t8 = clock()
    
    print('='*50)
    print("Pre-computation times for {:>10d} steps".format(n_samples))
    print('='*50)
    print('{:<35s}{:>16.10f}'.format("primal", t1-t0))
    print('{:<35s}{:>16.10f}'.format("tangent",t2 - t1)) 
    print('{:<35s}{:>16.10f}'.format("tangent source", t3 - t2))
    print('{:<35s}{:>16.10f}'.format("inverse adjoint source", t6 - t5))
    print('{:<35s}{:>16.10f}'.format("objective ", t4 - t3)) 
    print('{:<35s}{:>16.10f}'.format("gradient objective ", t5 - t4))
    print('{:<35s}{:>16.10f}'.format("sensitivity source ", t7 - t6))
    print('{:<35s}{:>16.10f}'.format("adjoint ", t8 - t7))
    print('*'*50)
    print("End of pre-computation")
    print('*'*50)

    ds1 = copy(ds0)
    ds1[0] = 1.0
    print('Starting stable-(adjoint-unstable) split evolution...')
    t13 = clock()
    unstable_sensitivity_source = unstable_sensitivity_source[:,0]
    dJds_stable, dJds_unstable, w0_inv = compute_sensitivity(u,s0,v0,w0,J_theta_phi,\
            DJ_theta_phi,\
            source_tangent,\
            unstable_sensitivity_source, \
            n_steps,Ninf)
    t14 = clock()
    print('{:<35s}{:>16.10f}'.format("time taken",t14-t13))
    print('End of computation...')

    dJds = dJds_stable + dJds_unstable
    theta = linspace(0,pi,n_points_theta)
    phi = linspace(-pi,pi,n_points_phi)
    #figure()
    #contourf(phi,theta,dJds_stable,100)
    #xlabel(r"$\phi$")
    #ylabel(r"$\theta$")
    #colorbar()
    figure()
    visualize_tangent_3D(u,w0_inv)
    #savefig("../examples/plots/plykin_poincare_main_stable")
    #figure()
    #contourf(phi,theta,dJds_unstable,100)
    #xlabel(r"$\phi$")
    #ylabel(r"$\theta$")
    #colorbar()
    #savefig("../examples/plots/plykin_poincare_main_unstable")
