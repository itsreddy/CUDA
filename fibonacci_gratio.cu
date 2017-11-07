#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
//#include "UnifiedMathCUDA.cuh"
#include "helper_functions.h"
#include <Windows.h>

cudaError_t addWithCuda(int *c, unsigned int size);

__global__ void addKernel(int *c)
{
	int i = threadIdx.x;
	double phi = 1.618033988749895;  // Golden Ratio
		c[i] = round((pow(phi, i) - pow(-phi, -i)) / sqrt((double)5));
}

int main()
{
	__int64 ctr1 = 0, ctr2 = 0, freq = 0;
	const int arraySize = 1000;
	int NoThreads = 8;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	QueryPerformanceCounter((LARGE_INTEGER *)&ctr1);    //profiling

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	QueryPerformanceCounter((LARGE_INTEGER *)&ctr2);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	printf("Time taken (In seconds) : %f\n", ((ctr2 - ctr1) * 1.0 / freq));


	printf("Sequence size: %d\nFibonacci sequence: ", arraySize);
	/*	for (int k = 0; k < arraySize; k++)
		printf("%d ", c[k]);
	
	printf("\n");
	*/
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);

	return cudaStatus;
}
