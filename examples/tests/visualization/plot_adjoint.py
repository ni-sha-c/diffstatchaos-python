from numpy import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
style.use('presentation')
import sys
import pdb

sys.path.insert(0, '../../')
from kuznetsov import *
matplotlib.interactive(False)
def rotate(u, uRef, i0, i1):
    r = sqrt(uRef[i0] * uRef[i0] + uRef[i1] * uRef[i1])
    a0 = uRef[i0] / r
    a1 = uRef[i1] / r
    u0 = -a0 * u[i1] + a1 * u[i0]
    u1 =  a0 * u[i0] + a1 * u[i1]
    u[i0] = u0
    u[i1] = u1 
   


def project2D_bottom(state, u, v):
    rotate(state, u, 0, 2)                         
    rotate(v, u, 0, 2)
    rotate(u, u, 0, 2)                         
    rotate(state, u, 1, 2)
    rotate(v, u, 1, 2)                         
    rotate(state, v, 1, 0)                         
    twoDCoord = zeros((2,state.shape[1]))
    twoDCoord[0] = state[0] 
    twoDCoord[1] = state[1]
    return twoDCoord
               

def project2D_top(state, u, v):
    rotate(state, u, 1, 2)
    rotate(v, u, 1, 2)
    rotate(u, u, 1, 2)
    rotate(state, u, 0, 2)
    rotate(v, u, 0, 2)
    rotate(state, v, 1, 0)
    twoDCoord = zeros((2,state.shape[1]))
    twoDCoord[0] = state[0] 
    twoDCoord[1] = state[1]
    return twoDCoord
   
       
        



@jit(nopython=True)
def solve_primal(u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[0] = u_init
    for i in range(1,n_steps):
        u[i] = primal_step(u[i-1],s)
    return u


@jit(nopython=True)
def solve_unstable_adjoint_direction(u, w_init, n_steps, s, dJ):
    w = empty((n_steps, w_init.size))
    w[-1] = w_init
    for i in range(n_steps-1,0,-1):
        w[i-1] = adjoint_step(w[i],u[i-1],s,dJ)
        w[i-1] /= norm(w[i-1])
    return w



A = loadtxt("trajectory.txt").T
frame = A[0]
u = A[1:5]
v0 = A[5:]

r = 1
phi, theta = mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)
fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.1, linewidth=0)
ax.scatter(u[0],u[1],u[2],s=30)
n_timesteps = u.shape[1]
u_padded = zeros((n_timesteps + 1,state_dim))
u_padded[:-1] = copy(u.T)


w0 = zeros((n_timesteps + 1, state_dim))
u = u.T
dJ0 = zeros(state_dim)
w0[-1] = rand(state_dim)
w0[-1] /= norm(w0[-1])
for i in range(n_timesteps,0,-1):
    u_mine = solve_primal(u_padded[i-1],51,s0) 
    w_mine = solve_unstable_adjoint_direction(u_mine, \
            w0[i], 51, s0, dJ0)
    w0[i-1] = w_mine[0]

w0 = w0[:-1]
w0 = w0.T
u = u.T
ax.quiver(u[0],u[1],u[2],w0[0], \
            w0[1],w0[2],color='gray',length=0.25)


w = copy(w0)
wtop = w0[:-1,abs(u[1])>abs(u[0])]
wbottom = w0[:-1,abs(u[0])>abs(u[1])]
utop = u[:-1,abs(u[1])>abs(u[0])]
ubottom = u[:-1,abs(u[0])>abs(u[1])]
vtop = v0[:-1,abs(u[1])>abs(u[0])]
vbottom = v0[:-1,abs(u[0])>abs(u[1])]

w2Dtop = project2D_top(wtop,utop,vtop)
w2Dbottom = project2D_bottom(wbottom,ubottom,vbottom)
w2D = zeros((2,n_timesteps))
w2D[:,abs(u[1])>abs(u[0])] = w2Dtop
w2D[:,abs(u[0])>abs(u[1])] = w2Dbottom
w2D /= norm(w2D,axis=0)
my_dpi = 80
for i in range(n_timesteps):
    figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    plot( [w2D[1,i]/3,-w2D[1,i]/3], [-w2D[0,i]/3,w2D[0,i]/3], \
           color='xkcd:azure',lw=3.0)

    plot([-1.0/3,1.0/3], \
            [0,0],color='xkcd:green',lw=3.0)

    axis('off')
    axis('scaled')
    xlim([-16./9,16./9])
    ylim([-1.0,1.0])
    filename = str("adjoint2D" + "_" + str(i) + ".png")
    savefig(filename, transparent=True, dpi=my_dpi)



