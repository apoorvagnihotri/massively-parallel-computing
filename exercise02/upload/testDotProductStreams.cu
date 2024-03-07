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

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_BLOCKS 256
#define MAX_THREADS 128

__global__ void dotProdKernel(float* _dst, const float* _a1, const float* _a2, int _dim)
{

    // calculate how many elements each thread needs to calculate
    const unsigned int iter = _dim / (blockDim.x * gridDim.x);
    int pos = blockIdx.x * MAX_THREADS + threadIdx.x;

    // clear the output
    _dst[blockIdx.x * MAX_THREADS + threadIdx.x] = 0;

    for (int i = 0; i < iter; ++i)
    {
        _dst[blockIdx.x * MAX_THREADS + threadIdx.x] += _a1[pos] * _a2[pos];
        pos += blockDim.x * gridDim.x;
    }

    // for the last iteration, check if the elements are still available
    if (pos < _dim)
    {
        _dst[blockIdx.x * MAX_THREADS + threadIdx.x] += _a1[pos] * _a2[pos];
    }
}

/* This program sets up two large arrays of size dim and computes the
 dot product of both arrays.

 The kernel from the last exercise can be reused.
 In contrast to the last exercise the second array will be uploaded in each iteration.

 In this exercise you should implement the upload of the second array in three different ways:

 1. using simple memcopy (as in exercise 1)
 2. using memcopy from non-pageable memory (should be faster)
 3. using asynchronous memory copy with two streams

 */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProductStreams <dim> <copy mode [0,1,2]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    // mode of the memory upload
    int mode = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    // Set up CPU arrays
    float* cpuOperator1[2];
    float* cpuOperator2[2];
    float* cpuResult[2];
    for (unsigned int pass = 0; pass < 2; pass++)
    {
        if (mode == 0) // simple memcpy
        {
            cpuOperator1[pass] = new float[dim];
            cpuOperator2[pass] = new float[dim];
            cpuResult[pass] = new float[MAX_THREADS * MAX_BLOCKS];
        }
        else // non-pageable memory
        {
            // !!! missing !!!
            // Allocate non-pageable memory
        }
    }

    // initialize the cpu arrays
    for (unsigned int pass = 0; pass < 2; pass++)
    {
        for (int i = 0; i < dim; ++i)
        {
#ifdef RTEST // With random numbers or constants...
            cpuOperator1[pass][i] = drand48();
            cpuOperator2[pass][i] = drand48();
#else
            cpuOperator1[pass][i] = 1.0f;
            cpuOperator2[pass][i] = 2.0f;
#endif
        }
    }

    // Set up the gpu arrays
    float* gpuOperator1[2];
    float* gpuOperator2[2];
    float* gpuResult[2];
    for (unsigned int pass = 0; pass < 2; pass++)
    {
        // !!! missing !!!
        // Allocate GPU memory
    }

    // create two streams for the last mode
    cudaStream_t stream[2];
    // !!! missing !!!
    // Create two streams

    // copy array 1 once to the device (will be static during all iterations)
    cudaMemcpy(gpuOperator1[0], cpuOperator1[0], dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuOperator1[1], cpuOperator1[1], dim * sizeof(float), cudaMemcpyHostToDevice);

    // 100 Iterations for better benchmarking
    for (int iter = 0; iter < 100; ++iter)
    {
        // Two calculations of dot products per iteration to see the advantage of streams.

        // a simplistic way of splitting the problem into threads
        dim3 blockGrid(MAX_BLOCKS);
        dim3 threadBlock(MAX_THREADS);

        switch (mode)
        {

        case 0:
            // copy a simple array
            printf("simple memcpy: \n");

            // Two passes per iteration (to be comparable to streamed version)
            for (unsigned int pass = 0; pass < 2; pass++)
            {
                cudaMemcpy(gpuOperator2[pass], cpuOperator2[pass], dim * sizeof(float),
                           cudaMemcpyHostToDevice);

                // call the kernel
                dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult[pass], gpuOperator1[pass],
                                                          gpuOperator2[pass], dim);

                // download and combine the results of multiple threads
                cudaMemcpy(cpuResult[pass], gpuResult[pass],
                           MAX_BLOCKS * MAX_THREADS * sizeof(float), cudaMemcpyDeviceToHost);

                // Calculate the result
                float finalDotProduct = 0.0f;
                for (int i = 0; i < MAX_BLOCKS * MAX_THREADS; ++i)
                    finalDotProduct += cpuResult[pass][i];
                printf("Iteration %d, pass %d: %f\n", iter, pass, finalDotProduct);
            }

            break;

        case 1:

            // copy Array2 from pagelocked memory
            printf("pagelocked memory:\n");

            // Two passes per iteration (to be comparable to streamed version)
            for (unsigned int pass = 0; pass < 2; pass++)
            {
                // !!! missing !!!
                // Calculate the dot product with non-pageable memory.
            }

            break;

        case 2:

            // use two streams with interleaved processing
            // use asynchronous up and download
            printf("2 streams:\n");

            // !!! missing !!!
            // Calculate the dot product with streams.

            break;

        } // end switch
    }

    // !!! missing !!!
    // Destroy streams

    // !!! missing !!!
    // cleanup GPU memory

    // cleanup host memory
    for (unsigned int pass = 0; pass < 2; pass++)
    {
        if (mode == 0) // simple memcpy
        {
            delete[] cpuOperator1[pass];
            delete[] cpuOperator2[pass];
            delete[] cpuResult[pass];
        }
        else // non-pageable memory
        {
            // !!! missing !!!
        }
    }

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
