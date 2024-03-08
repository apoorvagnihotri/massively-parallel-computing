// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <device_launch_parameters.h>

//typedef long long   __int64_t;
using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_BLOCKS 256
#define MAX_THREADS 256

inline __int64_t continuousTimeNs()
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    __int64_t result = (__int64_t)now.tv_sec * 1000000000 + (__int64_t)now.tv_nsec;

    return result;
}

__global__ void dotProdKernel(float* dst, const float* a1, const float* a2, int dim)
{
    // Number of the current thread
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    // Number of all threads
    unsigned int threadSize = gridDim.x * blockDim.x;

    // Sum up every (threadSize)th element starting at the threads index and ending before dim
    float result = 0.0f;
    for (unsigned int t = threadNo; t < dim; t += threadSize)
        result += a1[t] * a2[t];

    // Write the result to dst[threadIdx] if it can contain something
    dst[threadNo] = result;
}

// !!! missing !!!
// Kernel for reducing gridDim.x*blockDim.x elements to gridDim.x elements

/* This program sets up two large arrays of size dim and computes the
 dot product of both arrays.

 Most of the code of previous exercises is reused.
 Mode 0 of the program computes the final dot product as before.

 Mode 1: After computing the dot product and storing the result for all
 MAX_BLOCKS * MAX_THREAD threads, this time, the reduction of the sum
 is to be computed on the GPU.

 Write a reduction sum kernel which is called log(n) times.
 The number of total threads will be divided by nThreads(iter-1)
 in each iteration.

 Inside the kernel, the problem will be reduced by a factor of 2 in
 each step.

 */
__global__ void reductionKernel(float* dst, const float* a, int dim)
{

    extern __shared__ float partialSum[];
    
    // Number of the current thread
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int blockThreadNo = gridDim.x * threadIdx.x + blockIdx.x;

    // Number of all threads
    unsigned int threadSize = gridDim.x * blockDim.x;

    // load a into shared memory
    for (int i = 0; i* threadSize + blockThreadNo < dim; i++) {
        partialSum[i*blockDim.x + threadIdx.x] = a[i* threadSize + blockThreadNo];
    }

    // make sure all threads have written their parts
    __syncthreads();

    
    unsigned int firstStride = 1;
    int miniDim = dim/gridDim.x + 1;
    while (firstStride * 2 <= miniDim)
    {
    	firstStride *= 2;
    }
    for (unsigned int stride = firstStride; stride >= 1; stride >>= 1)
    {
    	__syncthreads();
    	for (int t = threadIdx.x; t < stride; t += blockDim.x)
    	{
    		if ( t + stride < miniDim && t + stride < dim)
        	{
        		partialSum[t] += partialSum[t+stride];
        	}
    	}


    }


    //int firstStride = blockDim.x;
    //while (firstStride * 2 <= dim)
    //{
    //    firstStride *= 2;
    //}
    //
    //for (unsigned int stride = firstStride; stride >= blockDim.x; stride >>= 1)
    //{
    //    __syncthreads();
    //    int startPos = threadNo;
    //
    //  for (int t = threadNo; t < stride; t += threadSize)
    //  {
    //  	printf("threadNo: %d, t: %d, stride: %d\n", threadNo, t, stride);
    //      if (t < dim && t+ stride < dim)
    //      {
    //
    //          printf("block: %d, thread: %d, stride: %d, startPos: %d, t: %d, %f + %f\n", blockIdx.x, threadIdx.x, stride, startPos, t, partialSum[t], partialSum[t + stride]);
    //
    //          partialSum[t] += partialSum[t + stride];
    //      }
    //
    //  }
    //
    //}

    __syncthreads();

    if (threadIdx.x == 0)
    {
        dst[blockIdx.x] = partialSum[0];
        //printf("%f ", dst[blockIdx.x]);
    }
    

}

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProductStreams <dim> <reduction mode [gold:0, CPU:1, GPU:2]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    int mode = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    // Allocate only pagelocked memory for simplicity
    float* cpuArray1;
    float* cpuArray2;
    float* cpuResult;
    cudaMallocHost((void**)&cpuArray1, dim * sizeof(float));
    cudaMallocHost((void**)&cpuArray2, dim * sizeof(float));
    cudaMallocHost((void**)&cpuResult, MAX_THREADS * MAX_BLOCKS * sizeof(float));

    // initialize the two arrays
    for (int i = 0; i < dim; ++i)
    {
#ifdef RTEST
        cpuArray1[i] = drand48();
        cpuArray2[i] = drand48();
#else
        cpuArray1[i] = 1.0;
        cpuArray2[i] = 1.; // i % 10;
#endif
    }

    // Allocate GPU memory
    float* gpuArray1;
    float* gpuArray2;
    float* gpuResult1; // Two result arrays to be able to move data from one to the other during
                       // reduction
    float* gpuResult2;
    cudaMalloc((void**)&gpuArray1, dim * sizeof(float));
    cudaMalloc((void**)&gpuArray2, dim * sizeof(float));

    cudaMalloc((void**)&gpuResult1, MAX_BLOCKS * MAX_THREADS * sizeof(float));

    cudaMalloc((void**)&gpuResult2,
               MAX_BLOCKS * MAX_THREADS
                   * sizeof(float)); // MAX_BLOCKS elements would be sufficient here...

    // Upload input data

    cudaMemcpy(gpuArray1, cpuArray1, dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpuArray2, cpuArray2, dim * sizeof(float), cudaMemcpyHostToDevice);

    // Variable for output
    double finalDotProduct = 0.;

    __int64_t startTime = continuousTimeNs();

    // Iterations for benchmarking only the kernel call
    for (int iter = 0; iter < 1; ++iter)
    {
        // a simplistic way of splitting the problem into threads
        dim3 blockGrid(MAX_BLOCKS);
        dim3 threadBlock(MAX_THREADS);

        unsigned int expectedResultSize;

        switch (mode)
        {
        case 0:
            finalDotProduct = 0.0;

            for (unsigned int i = 0; i < dim; i++)
                finalDotProduct += cpuArray1[i] * cpuArray2[i];

            break;

        case 1:
            // call the dot kernel
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);

            // If dim < launchedThreads, only the first dim elements will contain data
            expectedResultSize = min(dim, MAX_THREADS * MAX_BLOCKS);

            // download and combine the results of multiple threads

            cudaMemcpy(cpuResult, gpuResult1, expectedResultSize * sizeof(float),
                       cudaMemcpyDeviceToHost);

            finalDotProduct = 0.;

            // accumulate the final result on the host
            for (int i = 0; i < expectedResultSize; ++i)
                finalDotProduct += cpuResult[i];

            break;

        case 2:
            // call the dot kernel, store result in gpuResult1
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);
            expectedResultSize = min(dim, MAX_THREADS * MAX_BLOCKS);
            //cudaMemcpy(cpuResult, gpuResult1, expectedResultSize * sizeof(float),  cudaMemcpyDeviceToHost);



			// accumulate the final result on the host
			//for (int i = 0; i < expectedResultSize; ++i)
			//	printf("%f ",cpuResult[i]);
			//printf("\n");
            // !!! missing !!!
            // Reduce all the dot product summands to one single value,
            // download it to a float and use it to set finalDotProduct.
            cudaDeviceSynchronize();
            reductionKernel << <blockGrid, threadBlock, ((expectedResultSize/MAX_BLOCKS) + 1) * sizeof(float) >> > (gpuResult2, gpuResult1, expectedResultSize);
            {
                cudaError_t cudaerr = cudaDeviceSynchronize();
                if (cudaerr != cudaSuccess)
                    printf("kernel launch failed with error \"%s\".\n",
                           cudaGetErrorString(cudaerr));
            }
            reductionKernel << <dim3(1), threadBlock, MAX_BLOCKS * sizeof(float) >> > (gpuResult1, gpuResult2, MAX_BLOCKS);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuResult, gpuResult1, 1 * sizeof(float), cudaMemcpyDeviceToHost);
            finalDotProduct = (double)*cpuResult;
            break;

        } // end switch
    }

    __int64_t endTime = continuousTimeNs();
    __int64_t runTime = endTime - startTime;

    // Print results and timing
    printf("Result: %f\n", finalDotProduct);
    printf("Time: %f\n", (float)runTime / 1000000000.0f);

    // cleanup GPU memory
    cudaFree(gpuResult1);
    cudaFree(gpuResult2);
    cudaFree(gpuArray2);
    cudaFree(gpuArray1);

    // free page locked memory
    cudaFreeHost(cpuArray1);
    cudaFreeHost(cpuArray2);
    cudaFreeHost(cpuResult);

    checkCUDAError("end of program");

    printf("done\n");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}