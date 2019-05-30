#!/usr/bin/python2
import numpy as np
import os
import shutil
import sys
import h5py
#sys.path.insert(0,'/master/home/nishac/.local/lib/python2.7/site-packages')
sys.path.append("/master/home/nishac/fds/")
sys.path.append("/master/home/nishac/S3/")
#import map_sens
import fds
from adFVM.interface import SerialRunner
from adFVM.mesh import Mesh
class S3(SerialRunner):
    def __init__(self, *args, **kwargs):
        super(S3, self).__init__(*args, **kwargs)

    def __call__(self, initFields, parameter, nSteps, run_id):
        case = self.base + 'temp/' + run_id + "/"
        print(case)
        self.copyCase(case)
        data = self.runPrimal(initFields, (parameter, nSteps), case, args='--hdf5')
        if run_id=='0':
	    shutil.rmtree(case)
        return data

    def adjoint(self, initPrimalFields, parameter, nSteps, initAdjointFields, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runAdjoint(initPrimalFields, (parameter, nSteps), initAdjointFields, case, homogeneous=True)
        return data[0] 

def solve_unstable_tangent(runner, tanField, nSteps, time, trjdir):
    dt = runner.dt
    eps = 1.e-4
    parameter = 0.0
    primalFieldOrig = runner.readFields(trjdir, time)
    tanDir = runner.base + 'temp/' + 'tangent/'
    lyap_exp = 0.
    if not os.path.exists(tanDir):
        os.makedirs(tanDir)
    for i in range(nSteps):
        primalFieldPert = primalFieldOrig + eps*tanField
        primalFieldPert, _ = runner(primalFieldPert, parameter,\
                            1, '0')
        tanField = (primalFieldPert - primalFieldOrig)/eps
        norm_tan = np.linalg.norm(tanField)
        tanField /= norm_tan
        lyap_exp += np.log(norm_tan)/nSteps
        print("nsteps, lyap_exp", i, lyap_exp)
        runner.writeFields(tanField, tanDir, time) 
        time += dt
        primalFieldOrig = runner.readFields(trjdir, time) 
    return lyap_exp 

def solve_unstable_adjoint(runner, adjField, nSteps, initTime,\
                           finalTime, parameter, case):
    dt = runner.dt
    primalField = runner.readFields(case, finalTime)
    adjDir = runner.base + 'temp/' + 'adjoint'
    lyap_exp = 0.
    if not os.path.exists(adjDir):
        os.makedirs(adjDir)
    for i in range(nSteps):
        adjField = runner.adjoint(primalField, parameter,\
                1, adjField, 'adjoint')
        stop
        #norm_adj = np.linalg.norm(adjField)
        #adjField /= norm_adj
        #lyap_exp += np.log(norm_tan)/nSteps
        #runner.writeFields(tanField, tanDir, time) 
        #time += dt
        #primalFieldOrig = runner.readFields(case, time) 
    return lyap_exp 


     
def main():
    base = '/master/home/nishac/adFVM/cases/slab_60_fds/'
    time = 30.0
    dt = 1e-4
    template = '/master/home/nishac/adFVM/templates/slab_60_fds.py'
    nProcs = 1

    runner = S3(base, time, dt, template, nProcs=nProcs, flags=['-g', '--gpu_double'])
    #s3sens = map_sens.Sensitivity
    nSteps = 20000
    nExponents = 2
    runUpSteps = 0
    parameter = 0.0
    checkpointPath = base + 'checkpoint/'

    time = 30.6762
    trjdir = base + 'temp/' + 'primal/'
    initField = runner.readFields(trjdir, time)

    #outField = runner(initField, parameter, nSteps, runId)
    tanInit = np.random.rand(initField.shape[0])
    tanInit /= np.linalg.norm(tanInit)


    le = solve_unstable_tangent(runner, tanInit, nSteps, time, trjdir)
    print(le)
    #initTime = 30.0
    #finalTime = initTime + dt 
    #adjField = np.random.rand(initField.shape[0])    
    #le = solve_unstable_adjoint(runner, adjField, 1, initTime,\
    #                      finalTime, parameter, case)
if __name__ == '__main__':
    main()
