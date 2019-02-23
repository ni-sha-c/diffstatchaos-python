import sys
sys.path.insert(0, '../../examples')
import kuznetsov_poincare as kp
from numpy import *
if __name__=="__main__":
#def test_tangent_conversions():
    solver = kp.Solver()
    u_init = solver.u_init
    n_runup = 500
    s0 = solver.s0
    u_init = solver.primal_step(u_init, s0, n_runup)
    v_xyz = random.rand(u_init.size)
    v_x1, v_x2 = solver.convert_tangent_to_stereo(u_init, v_xyz) 
    v_x1x2 = array([v_x1, v_x2])
    x1, x2 = solver.stereographic_projection(u_init)
    u = array([x1,x2])
    v_x, v_y, v_z = solver.convert_tangent_to_spherical(u, v_x1x2)
    v_xyz_computed = array([v_x, v_y, v_z])
    print(v_xyz)
    print(v_xyz_computed)
    assert(max(linalg.norm(v_xyz_computed - v_xyz[:-1])) < 1.e-5)
