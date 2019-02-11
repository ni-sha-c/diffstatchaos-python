import sys
sys.path.insert(0, '../examples/')
import kuznetsov_poincare as kp
from objective import *
import scipy
from scipy.interpolate import griddata
import map_sens as map_sens
from time import clock
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    solver = kp.Solver()
    n_steps = 5000
    s3_map = map_sens.Sensitivity(solver,n_steps)
    n_runup = s3_map.n_runup

    u_init = solver.u_init
    s0 = solver.s0
    state_dim = solver.state_dim
    param_dim = s0.size
    u_init = solver.primal_step(u_init, s0, n_runup)
    u_trj = s3_map.solve_primal(solver,\
            solver.u_init, n_steps, solver.s0)
    ds0 = zeros(param_dim)
    ds1 = copy(ds0)
    ds1[0] = 1.0
    dJ0 = zeros(state_dim)
    
    w0 = zeros((n_steps,state_dim))
    w0_init = rand(state_dim)
    w0_init /= linalg.norm(w0_init)
    
    v0_init = rand(state_dim)
    v0_init /= linalg.norm(v0_init)

    n_points_theta = 20
    n_points_phi = 20
    dtheta = pi/n_points_theta
    dphi = 2.*pi/n_points_phi
    J_theta_phi = zeros((n_steps,n_points_theta,n_points_phi))
    DJ_theta_phi = zeros((n_steps,n_points_theta,n_points_phi,state_dim))  
    theta_bin_centers = linspace(dtheta/2.0, pi - dtheta/2.0, n_points_theta)
    phi_bin_centers = linspace(-pi-dphi/2.0, pi - dphi/2.0, n_points_phi)
    v = zeros(state_dim)
    w_inv = zeros(state_dim)
    dfds = zeros((n_steps,param_dim,state_dim))
    source_forward_adjoint = zeros((n_steps,state_dim))
    source_tangent = zeros((n_steps,param_dim,state_dim))
    dJds_stable = zeros((n_points_theta,n_points_phi))
    dJds_unstable = zeros((n_points_theta,n_points_phi))
    divdfds = zeros(n_steps)

    source_tangent = s3_map.compute_source_tangent(solver, u_trj, n_steps, s0)[:,0,:] 
#    J_theta_phi = compute_objective(u,s0,n_steps,n_points_theta,n_points_phi)
 #   DJ_theta_phi = compute_gradient_objective(u,s0,n_steps,n_points_theta,n_points_phi)
    source_forward_adjoint = s3_map.compute_source_forward_adjoint(solver, u_trj, n_steps, s0)

    unstable_sensitivity_source = s3_map.compute_source_sensitivity(solver,u_trj,n_steps,s0)


    v0 = s3_map.solve_unstable_direction(solver, u_trj, v0_init, n_steps, s0)

    w0 = s3_map.solve_unstable_adjoint_direction(solver, u_trj, w0_init, n_steps, s0)


def plot_tangent_vectors_3D(u_trj, v):
    n_points = u_trj.shape[0]
    state_dim = u_trj.shape[1]
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n_grid = 50
    theta_grid = linspace(0.,pi,n_grid)
    phi_grid = linspace(-pi,pi,n_grid)
    theta_grid, phi_grid = meshgrid(theta_grid, phi_grid)

    ct_grid = cos(theta_grid)
    st_grid = sin(theta_grid)
    cp_grid = cos(phi_grid)
    sp_grid = sin(phi_grid)

    x_grid = st_grid*cp_grid
    y_grid = st_grid*sp_grid
    z_grid = ct_grid
    
    ax.plot_surface(x_grid, y_grid,\
            z_grid,color="k",alpha=0.2)
    u_trj = u_trj.T
    v = v[::10]
    v = v.T
    v /= norm(v,axis=0)
    ax.plot(u_trj[0],u_trj[1],\
            u_trj[2],'k.',alpha=0.5,ms=10)
   
    u_start = u_trj[:,::10]
    u_end = u_start + 2.e-1*v
    n_points = u_start.shape[1]
    u_to_upv_x = array([u_start[0],u_end[0]]).\
            reshape(2, n_points).T
    u_to_upv_y = array([u_start[1],u_end[1]]).\
            reshape(2, n_points).T
    u_to_upv_z = array([u_start[2],u_end[2]]).\
            reshape(2, n_points).T
    for i in range(n_points):
        ax.plot(u_to_upv_x[i],\
            u_to_upv_y[i],\
            u_to_upv_z[i],linewidth=2.5,color='b')
    ax.set_xlabel("x",fontsize=24)
    ax.set_ylabel("y",fontsize=24)
    ax.set_zlabel("z",fontsize=24)
    ax.tick_params(axis='both',labelsize=24)
    return fig, ax


def plot_tangent_vectors_2D(solver, u_trj, v):
    
    n_points = u_trj.shape[0]
    state_dim = u_trj.shape[1]
    u_trj = u_trj.T
    v = v.T
    re_trj, im_trj = solver.stereographic_projection(u_trj)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(re_trj, im_trj, 'k.', ms=10, alpha=0.5)


    ax.set_xlabel(r"$x_1$",fontsize=24)
    ax.set_ylabel(r"$x_2$",fontsize=24)
    ax.tick_params(axis='both',labelsize=24)

    return fig, ax

def coordinate_function(u, index=0):
    n_points = u.shape[0]
    state_dim = u.shape[1]
    assert(index < state_dim)
    u = u.T
    return u[index]


def derivative_coordinate_function(u, index=0):
    n_points = u.shape[0]
    state_dim = u.shape[1]
    assert(index < state_dim)
    res = zeros_like(u.T)
    res[index] = 1.
    return res


def plot_scalar_function(u_trj, scalar_fn_handle, solver):
    x1_trj, x2_trj = solver.stereographic_projection(u_trj.T)
    u_trj = array([x1_trj, x2_trj]).reshape(2,-1).T
    J_trj = scalar_fn_handle(u_trj,index=0)
    x1_min, x1_max = min(x1_trj), max(x1_trj)
    x2_min, x2_max = min(x2_trj), max(x2_trj)
    x1_grid = linspace(x1_min, x1_max, 100)
    x2_grid = linspace(x2_min, x2_max, 100)
    x1_grid, x2_grid = meshgrid(x1_grid, x2_grid)
    J_grid = scipy.interpolate.griddata(u_trj, J_trj, (x1_grid, \
            x2_grid), method='linear')
    fig = figure()
    ax = fig.add_subplot(111)
    ax.contourf(x1_grid, x2_grid, J_grid)


    ax.set_xlabel(r"$x_1$",fontsize=24)
    ax.set_ylabel(r"$x_2$",fontsize=24)
    ax.tick_params(axis='both',labelsize=24)

    return fig, ax


def plot_directional_derivative(u_trj, derivative_fn_handle, v_trj, solver):

    DJ_trj = derivative_fn_handle(u_trj)
    x1_trj, x2_trj = solver.stereographic_projection(u_trj.T)
    u_trj = array([x1_trj, x2_trj]).reshape(2,-1)
    v_trj = v_trj.T
    v1, v2 = solver.convert_tangent_to_stereo(u_trj,\
            v_trj)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x1_trj, x2_trj, "k.", ms=10, alpha=0.5)
    x1_trj_pert = x1_trj + 1.e-1*v1
    x2_trj_pert = x2_trj + 1.e-2*v2
    ax.plot([x1_trj, x1_trj_pert],\
            [x2_trj_pert, x2_trj],\
            "b", linewidth=2.5)
    return fig, ax

    '''    
    ds1 = copy(ds0)
    ds1[0] = 1.0
    print('Starting stable-(adjoint-unstable) split evolution...')
    t13 = clock()
    unstable_sensitivity_source = unstable_sensitivity_source[:,0]
    dJds_stable, dJds_unstable, w0_inv, gsum_history = compute_sensitivity(u,s0,v0,w0,J_theta_phi,\
            DJ_theta_phi,\
            source_tangent,\
            unstable_sensitivity_source, \
            source_forward_adjoint,\
            n_steps,n_runup_forward_adjoint)
    t14 = clock()
    print('{:<35s}{:>16.10f}'.format("time taken",t14-t13))
    print('End of computation...')

    dJds = dJds_stable + dJds_unstable
    theta = linspace(0,pi,n_points_theta)
    phi = linspace(-pi,pi,n_points_phi)
    figure()
    contourf(phi,theta,dJds_stable,100)
    xlabel(r"$\phi$")
    ylabel(r"$\theta$")
    colorbar()
    figure()
    #visualize_tangent_3D(u,w0_inv)
    #savefig("../examples/plots/plykin_poincare_main_stable")
    #figure()
    contourf(phi,theta,dJds_unstable,100)
    xlabel(r"$\phi$")
    ylabel(r"$\theta$")
    colorbar()
    #savefig("../examples/plots/plykin_poincare_main_unstable")
    figure()
    contourf(phi,theta,dJds_unstable+dJds_stable,100)
    xlabel(r"$\phi$")
    ylabel(r"$\theta$")
    colorbar()
    #savefig("../examples/plots/plykin_poincare_main_total")
    '''
