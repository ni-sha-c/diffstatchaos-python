from numba import cuda, float64
import numpy as np
TPB = 16

@cuda.jit
def matmul(A,B,C):
	sA = cuda.shared.array(shape=(TPB,TPB), dtype=float64)
	sB = cuda.shared.array(shape=(TPB,TPB), dtype=float64)
	x, y = cuda.grid(2)
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bpg = cuda.gridDim.x

	if x >= C.shape[0] and y >= C.shape[1]:
		return
	tmp = 0.
	for i in range(bpg):
		sA[tx,ty] = A[x, ty + i*TPB]
		sB[tx,ty] = B[tx + i*TPB, y]
		
		cuda.syncthreads()
		for j in range(TPB):
			tmp += sA[tx,j]*sB[j,ty]
		cuda.syncthreads()
	
	C[x,y] = tmp
 
A = np.random.rand(256,256)
B = np.random.rand(256,256)
C = np.empty_like(A)
threadsperblock = (16, 16)
blockspergrid_x = np.math.ceil(A.shape[0] / threadsperblock[0])
blockspergrid_y = np.math.ceil(A.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
matmul[blockspergrid, threadsperblock](A,B,C)
