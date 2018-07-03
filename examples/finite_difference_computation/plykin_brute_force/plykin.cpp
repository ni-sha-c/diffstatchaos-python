#include<cmath>
#include<cstdio>
#include<cassert>
#include<cstdlib>
#include<algorithm>

using namespace std;

template<typename ftype>
__device__ __forceinline__ void step(ftype u[4], ftype s[2], int n)
{
	const double PI = atan2(1.0, 0.0) * 2;
	const double dt = 1.e-2;
	const double T = 6.0;
    for (int i = 0; i < n; ++i) {
        ftype x = u[0];
        ftype y = u[1];
        ftype z = u[2];
    	ftype t = u[3];

        ftype r2 = x * x + y * y + z * z;
        ftype r = sqrt(r2);
		ftype sigma = diff_rot_freq(t);
		ftype a = rot_freq(t);

		ftype coeff1 = sigma*PI*0.5*(z*sqrt(2.0) + 1.0);
		ftype coeff2 = s[0]*(1.0 - sigma*sigma - a*a);
		ftype coeff3 = s[1]*a*a*(1.0 - r);
        	
		u[0] += dt*(-coeff1*y -
                        coeff2*x*y*y +
                        0.5*a*PI*z + coeff3*x);

        u[1] += dt*(coeff1*x +
                        coeff2*y*x*x +
                        coeff3*y);

        u[2] += dt*(-0.5*a*PI*x + coeff3*z);

        u[3] = fmod((u[3] + dt),T);

    
	
	}
}

template<typename ftype>
__device__ __forceinline__ ftype rot_freq(ftype t)
{
    ftype a0 = -1.0;
    ftype a1 = 0.0;
    ftype a2 = 1.0;

    ftype c0 = 2.0;
    ftype c1 = 3.0;
    ftype c2 = 5.0;
    ftype c3 = 6.0;
    ftype c4 = 0.0;

    ftype slope = 10.0;
    ftype est = exp(slope*t);
    ftype esc0 = exp(slope*c0);
    ftype esc1 = exp(slope*c1);
    ftype esc2 = exp(slope*c2);
    ftype esc3 = exp(slope*c3);
    ftype esc4 = exp(slope*c4);

    ftype fn0 = (a1*esc0 + a0*est)/(esc0 + est);	
    ftype fn1 = (a0*esc1 + a1*est)/(esc1 + est);
    ftype fn2 = (a1*esc2 + a2*est)/(esc2 + est);
    ftype fn3 = (a2*esc3 + a1*est)/(esc3 + est);
    ftype fn4 = (a2*esc4 + a1*est)/(esc4 + est);

    return fn0 + fn1 + fn2 + fn3 + fn4;

}

template<typename ftype>
__device__ __forceinline__ ftype diff_rot_freq(ftype t)
{
    ftype a0 = -1.0;
    ftype a1 = 0.0;
    ftype a2 = 1.0;

    ftype c0 = 1.0;
    ftype c1 = 2.0;
    ftype c2 = 4.0;
    ftype c3 = 5.0;

    ftype slope = 10.0;
    ftype est = exp(slope*t);
    ftype esc0 = exp(slope*c0);
    ftype esc1 = exp(slope*c1);
    ftype esc2 = exp(slope*c2);
    ftype esc3 = exp(slope*c3);

    ftype fn0 = (a1*esc0 + a0*est)/(esc0 + est);	
    ftype fn1 = (a0*esc1 + a1*est)/(esc1 + est);
    ftype fn2 = (a1*esc2 + a2*est)/(esc2 + est);
    ftype fn3 = (a2*esc3 + a1*est)/(esc3 + est);

    return fn0 + fn1 + fn2 + fn3;

}



template<typename ftype>
__device__ __forceinline__ ftype objective(ftype u[4], ftype s[2], int itheta, ftype dtheta, int iphi, ftype dphi)
{
    const double PI = atan2(1.0, 0.0) * 2;
    ftype x = u[0];
    ftype y = u[1];
	ftype z = u[2];
    ftype r2 = x * x + y * y + z * z;
    ftype r = sqrt(r2);
    ftype theta = acos(z/r);
	ftype phi = atan2(y,x);

    phi += PI;
    if (iphi == 0) theta = fmod(theta + dtheta, (ftype)(2*PI)) - dtheta;

    ftype pFrac = phi / dphi - iphi;
    ftype tFrac = theta / dtheta - itheta;
    return max((ftype)0, min(1 - pFrac, 1 + pFrac)) *
           max((ftype)0, min(1 - tFrac, 1 + tFrac));
}

template<typename ftype, typename objType>
__global__ void accumulate(ftype (*u)[4], ftype s[2], objType *obj,
                           int ntheta, int nphi, int steps)
{
    const double PI = atan2(1.0, 0.0) * 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	step(u[i], s, steps);
	ftype dtheta = PI / ntheta;
    ftype dphi = (2*PI) / nphi;
    for (int itheta = 0; itheta < ntheta; ++itheta) {
        for (int iphi = 0; iphi < nphi; ++iphi) {
            int ibin = itheta * nphi + iphi;
            ftype obji = objective(u[i], s, itheta, dtheta, iphi, dphi);
            if (obji) 
				atomicAdd(obj + ibin, obji);
			
        }
    }
	
}

typedef float ftype;

void init(ftype (**u)[4], ftype ** s, ftype s1, ftype s2, int nBlocks, int threadsPerBlock)
{
    const int nSamples = nBlocks * threadsPerBlock;
    ftype (*uCPU)[4] = new ftype[nSamples][4];
    for (int i = 0; i < nSamples; ++i) {
        uCPU[i][0] = 2.0*rand() / (ftype)RAND_MAX - 1.0;
        uCPU[i][1] = 2.0*rand() / (ftype)RAND_MAX - 1.0;
        uCPU[i][2] = 2.0*rand() / (ftype)RAND_MAX - 1.0;
		uCPU[i][3] = 6.0*rand() / (ftype)RAND_MAX ;
    }
    cudaMalloc(u, sizeof(ftype) * nSamples * 4);
    cudaMemcpy(*u, uCPU, sizeof(ftype) * nSamples * 4, cudaMemcpyHostToDevice);
	
    delete[] uCPU;

    ftype sCPU[2] = {s1, s2};
    cudaMalloc(s, sizeof(ftype) * 2);
    cudaMemcpy(*s, sCPU, sizeof(ftype) * 2, cudaMemcpyHostToDevice);

    accumulate<<<nBlocks, threadsPerBlock>>>(*u, *s, (ftype*)0, 0, 0, 5000);


}

int main(int argc, char * argv[])
{
    const int nBlocks = 32;
    const int threadsPerBlock = 256;

    assert (argc == 6);
    int iDevice = atoi(argv[1]);
    if (cudaSetDevice(iDevice)) {
        fprintf(stderr, "error selecting %d\n", iDevice);
        exit(-1);
    }
    int ntheta0 = atoi(argv[2]);
    int nphi0 = atoi(argv[3]);
	ftype s1 = atof(argv[4]);
	ftype s2 = atof(argv[5]); 
	

    ftype (*u)[4], *s;
    init(&u, &s, s1, s2, nBlocks, threadsPerBlock);

    ftype * objCPU = new ftype[ntheta0 * nphi0];
    double * objFinal = new double[ntheta0 * nphi0];
    memset(objFinal, 0, sizeof(double) * ntheta0 * nphi0);

    ftype * objective;
    cudaMalloc(&objective, sizeof(ftype) * nphi0 * ntheta0);

    const int nRepeat = 2048;
    for (int iRepeat = 0; iRepeat < nRepeat; ++iRepeat) {
        cudaMemset(objective, 0, sizeof(ftype) * nphi0 * ntheta0);
        accumulate<<<nBlocks, threadsPerBlock>>>(u, s, objective, ntheta0, nphi0, 10);
        cudaMemcpy(objCPU, objective, sizeof(ftype) * nphi0 * ntheta0, cudaMemcpyDeviceToHost);

    
        for (int i = 0; i < ntheta0 * nphi0; ++i) {
            objFinal[i] += objCPU[i] / nRepeat / nBlocks / threadsPerBlock;
        }
    }
	
    for (int it0 = 0; it0 < ntheta0; ++it0) {
        for (int ir0 = 0; ir0 < nphi0; ++ir0) {
            int i = it0 * nphi0 + ir0;
            printf("%d %d %40.30f\n", it0, ir0, objFinal[i]);
        }
    }
}
