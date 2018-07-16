from numpy import *
from pylab import *
style.use('presentation')
n = 50
Jp = loadtxt("Jplus.txt").T
Jm = loadtxt("Jminus.txt").T
#dobjds1 = loadtxt("dJds1_fd.txt").T
#dobjds2 = loadtxt("dJds2_fd.txt").T
phi = linspace(-pi,pi,n)
theta = linspace(0.,pi,n)
f = figure()
subplot(3,1,1)
contourf(phi, theta, Jp, 100)
axis('scaled')
subplot(3,1,2)
contourf(phi, theta, Jm, 100)
axis('scaled')
subplot(3,1,3)
contourf(phi, theta, Jp-Jm, 100)
axis('scaled')
#colorbar()
#xlabel(r"$\phi$")
#ylabel(r"$\theta$")
'''
savefig("../../plots/plykin_dJds1")

g = figure(2)
contourf(phi, theta, dobjds2, 100)
xlabel(r"$\phi$")
ylabel(r"$\theta$")
axis('scaled')
colorbar()
savefig("../../plots/plykin_dJds2")
f = figure(3)
contourf(phi, theta, J, 100)
colorbar()
xlabel(r"$\phi$")
ylabel(r"$\theta$")
'''
