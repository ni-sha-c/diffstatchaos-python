ARCH=52
CUDA=-Wno-deprecated-gpu-targets -m64 -x cu -arch=sm_${ARCH} -lineinfo -Xptxas --warn-on-local-memory-usage -Xptxas --warn-on-spills --use_fast_math

default:	plykin.exe

%.exe:	%.o
	CXX=g++-6 CC=gcc-6 nvcc -arch=sm_${ARCH} $< -o $@

%.o:	%.cpp
	CXX=g++-6 CC=gcc-6 nvcc -D_FORCE_INLINES -std=c++11 ${CUDA} ${INC} -Xcompiler -fPIC -I. -O3 -dc $< -o $@

plot:
	rsync -av nishac@endeavour.mit.edu:/home/nishac/plykin_brute_force/*.txt .
	ipython -i plot_sensitivities.py
