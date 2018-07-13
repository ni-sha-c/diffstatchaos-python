from numpy import *
from pylab import *
style.use('presentation')
n = 50
J = loadtxt("J.txt").T
dobjds1 = loadtxt("dJds1_fd.txt").T
dobjds2 = loadtxt("dJds2_fd.txt").T
phi = linspace(-pi,pi,n)
theta = linspace(0.,pi,n)
f = figure(1)
contourf(phi, theta, dobjds1, 100)
colorbar()
xlabel(r"$\phi$")
ylabel(r"$\theta$")
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

