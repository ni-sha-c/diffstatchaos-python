import numpy as np

from adFVM import config
from adFVM.density import RCF 
from adFVM.mesh import Mesh
from adpy import tensor

def objectiveDrag(U, T, p, *mesh, **options):
    solver = options['solver']
    mesh = Mesh.container(mesh)
    U0 = U.extract(mesh.neighbour)[0]
    U0i = U.extract(mesh.owner)[0]
    p0 = p.extract(mesh.neighbour)
    T0 = T.extract(mesh.neighbour)
    nx = mesh.normals[0]
    mungUx = solver.mu(T0)*(U0-U0i)/mesh.deltas
    drag = (p0*nx-mungUx)*mesh.areas
    return drag.sum()

def objectiveLift(U, T, p, *mesh, **options):
    solver = options['solver']
    mesh = Mesh.container(mesh)
    U0 = U.extract(mesh.neighbour)[0]
    U0i = U.extract(mesh.owner)[0]
    p0 = p.extract(mesh.neighbour)
    T0 = T.extract(mesh.neighbour)
    nx = mesh.normals[0]
    ny = mesh.normals[1]
    muDUDy = solver.mu(T0)*(U0-U0i)/mesh.deltas 
    drag = (p0*nx-mungUx)*mesh.areas
    return drag.sum()


def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    patch = mesh.boundary['slab']
    startFace, nFaces = patch['startFace'], patch['nFaces']
    meshArgs = _meshArgs(startFace)
    #drag = tensor.Zeros((1,1))
    #drag = tensor.Kernel(objectiveDrag)(nFaces, (drag,))(U, T, p, *meshArgs, solver=solver)
    drag = tensor.Zeros((1,1))
    drag = tensor.Kernel(objectiveDrag)(nFaces, (drag,))(U, T, p, *meshArgs, solver=solver)
    inputs = (drag,)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    (drag,) = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs
    return drag
   
#primal = RCF('cases/cylinder_chaos_test/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)), boundaryRiemannSolver='eulerLaxFriedrichs')
primal = RCF('/master/home/nishac/adFVM/cases/slab_60_fds/temp/adjoint',
#primal = RCF('/home/talnikar/adFVM/cases/cylinder/Re_500/',
#primal = RCF('/home/talnikar/adFVM/cases/cylinder/chaotic/testing/', 
             #mu=lambda T: 2.5e-5*T/T,
             mu=lambda T: 3.4e-5,
             boundaryRiemannSolver='eulerLaxFriedrichs',
             objective = objective,
             fixedTimeStep = True,
             readConservative = False
)

#def makePerturb(param, eps=1e-4):
#    def perturbMesh(fields, mesh, t):
#        if not hasattr(perturbMesh, 'perturbation'):
#            ## do the perturbation based on param and eps
#            #perturbMesh.perturbation = mesh.getPerturbation()
#            points = np.zeros_like(mesh.points)
#            #points[param] = eps
#            points[:] = eps*mesh.points
#            perturbMesh.perturbation = mesh.getPointsPerturbation(points)
#        return perturbMesh.perturbation
#    return perturbMesh
##perturb = [makePerturb(1), makePerturb(2)]
#perturb = [makePerturb(1)]
#
#parameters = 'mesh'

def makePerturb(scale):
    def perturb(fields, mesh, t):
        #mid = np.array([-0.012, 0.0, 0.])
        #G = 100*np.exp(-3e4*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        mid = np.array([-0.0005, 0.0, 0.], config.precision)
        #G = scale*np.exp(-2.5e9*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        G = scale*np.exp(-np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        rho = G
        rhoU = np.zeros((mesh.nInternalCells, 3), config.precision)
        rhoU[:, 0] += G.flatten()*200
        rhoE = G*4e5
        return rho, rhoU, rhoE
    return perturb
 
perturb = [makePerturb(1e-3)]
parameters = 'source'

#patchID = 'inlet'
#def makePerturb(pt_per):
#    def perturb(fields, mesh, t):
#        nFaces = mesh.boundary[patchID]['nFaces']
#        return pt_per*np.ones((nFaces, 1), config.precision)
#    return perturb

#def makePerturb(scale):
#    def perturb(fields, mesh, t):
#        return scale*mesh.cellCentres[:

##perturb = [makePerturb(0.1), makePerturb(0.2), makePerturb(0.4)]
#perturb = [makePerturb(1.)]
#parameters = ('BCs', 'p', patchID, 'pt')

#nSteps = 200000
#writeInterval = 10000
#reportInterval = 100
#sampleInterval = 50
nSteps = 1
reportInterval = 1
sampleInterval = 1
startTime = 30.
writeInterval = 1
dt = 1e-4
