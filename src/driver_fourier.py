import sys
from pylab import *
from numpy import *
from time import clock
from numba import jit
from matplotlib import *
from fourierAnalysis import *
sys.path.insert(0,'../examples/')
from kuznetsov import *

foa = FourierAnalysis()
solver = Solver()
u = foa.solve_primal(solver, \
        solver.u_init, \
        solver.n_poincare*1000, solver.s0)

u = u.T
'''
n_max = 500
corr_uu = foa.compute_correlation_function(u[0],u[0],n_max) - \
        mean(u[0])*mean(u[0])
figure()
plot(range(1,n_max+1),corr_uu ,".")
title(r'$\rho_{u,u}}$')


dft_corr_uu = (2.0/n_max)*\
        abs(fft.fft(corr_uu))[:n_max//2]
figure()
plot(range(n_max//2),dft_corr_uu)
title(r'$\hat{\rho}_{u,u}}$')


'''
fig, ax = subplots(nrows=3,ncols=3)
fig_fft, ax_fft = subplots(nrows=3,ncols=3)
n_max = solver.n_poincare*50
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[1,-1].axis('off')

ax_fft[0,1].axis('off')
ax_fft[0,2].axis('off')
ax_fft[1,-1].axis('off')

corr_uu = zeros((3,3,n_max))
dft_corr_uu = zeros((3,3,n_max//2))

for i in range(3):
    f = u[i]
    for j in range(i+1):
        g = u[j]
        corr_uu[i,j] = abs(foa.compute_correlation_function(f,g,n_max) \
                - mean(f)*mean(g))
        ax[i,j].plot(range(1,n_max+1),corr_uu[i,j],'--')
        ax[i,j].set_title(r'$\rho_{%d,%d}}$' %(i,j) )
        
        dft_corr_uu[i,j] = (2.0/n_max)*\
                abs(fft.fft(corr_uu[i,j]))[:n_max//2]
        ax_fft[i,j].plot(range(n_max//2),dft_corr_uu[i,j],'--')
        ax_fft[i,j].set_title(r'$\hat{\rho}_{%d,%d}}$' %(i,j) )

r = sqrt(u[0]**2.0 + u[1]**2.0 + \
        u[2]**2.0)
theta = arccos(u[2]/r)
phi = arctan2(u[1],u[0])
corr_theta_phi = foa.compute_correlation_function(theta,phi,n_max)
figure()
plot(range(1,n_max+1),corr_theta_phi)
title(r'$\rho_{\theta,\phi}}$')


dft_corr_theta_phi = (2.0/n_max)*\
        abs(fft.fft(corr_theta_phi))[:n_max//2]
figure()
plot(range(n_max//2),dft_corr_theta_phi)
title(r'$\hat{\rho}_{\theta,\phi}}$')


