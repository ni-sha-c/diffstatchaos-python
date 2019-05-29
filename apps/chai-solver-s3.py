#!/usr/bin/python2
import numpy as np
import os
import shutil
import sys
import h5py
sys.path.insert(0,'/master/home/nishac/.local/lib/python2.7/site-packages')
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
        self.copyCase(case)
        data = self.runPrimal(initFields, (parameter, nSteps), case, args='--hdf5')
        if run_id=='0':
	    shutil.rmtree(case)
        return data

    def adjoint(self, initPrimalFields, parameter, nSteps, initAdjointFields, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runAdjoint(initPrimalFields, (parameter, nSteps), initAdjointFields, case)
        return 
def solve_unstable_tangent(runner, tanField, nSteps, time, case):
    dt = runner.dt
    eps = 1.e-4
    parameter = 0.0
    primalFieldOrig = runner.readFields(case, time)
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
        runner.writeFields(tanField, tanDir, time) 
        time += dt
        primalFieldOrig = runner.readFields(case, time) 
    return lyap_exp 

def solve_unstable_adjoint(runner, adjField, nSteps, time, case):
    dt = runner.dt
    eps = 1.e-4
    parameter = 0.0
    primalFieldOrig = runner.readFields(case, time)
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
        runner.writeFields(tanField, tanDir, time) 
        time += dt
        primalFieldOrig = runner.readFields(case, time) 
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
    initFields = runner.readFields(base, time)
    runId = 'primal'

    #outFields = runner(initFields, parameter, nSteps, runId)
    tanInit = np.random.rand(initFields.shape[0])
    tanInit /= np.linalg.norm(tanInit)
    case = base + 'temp/' + runId + '/'   
    le = solve_unstable_tangent(runner, tanInit, nSteps, time, case)
    print(le)
    '''if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)

    fields = runner.readFields(base, time)
    J, dJds_tan = fds.shadowing(runner, fields, parameter, nExponents, nSegments, nSteps, runUpSteps, epsilon=1.e-2,checkpoint_path=checkpointPath)
    #dJds_adj = fds.adjoint_shadowing(runner.solve, runner.adjoint, parameter, nExponents, checkpointPath)
    '''
if __name__ == '__main__':
    main()
