// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
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

#define RTEST // use random initialization of array

/* compute the dot product between a1 and a2. a1 and a2 are of size
 dim. The result of each thread should be stored in _dst[blockIdx.x *
 blockDim.x + threadIdx.x]. Each thread should accumulate the dot
 product of a subset of elements.
 */
__global__ void dotProdKernel(float* _dst, const float* _a1, const float* _a2, int _dim)
{

    // program your kernel here
    //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
}

/* This program sets up two large arrays of size dim and computes the
dot product of both arrays.

The arrays are uploaded only once and the dot product is computed
multiple times. While this does not make too much sense it
demonstrated the possible speedup.  */
int main(int argc, char* argv[])
{
    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProduct <dim> <GPU-flag [0,1]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    // flag indicating weather the CPU or the GPU version should be executed
    bool gpuVersion = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    float* cpuArray1 = new float[dim];
    float* cpuArray2 = new float[dim];

    // initialize the two arrays (either random or deterministic)
    for (int i = 0; i < dim; ++i)
    {
#ifdef RTEST
        cpuArray1[i] = drand48();
        cpuArray2[i] = drand48();
#else
        cpuArray1[i] = 2.0;
        cpuArray2[i] = i % 10;
#endif
    }

    // now the gpu stuff
    float* gpuArray1;
    float* gpuArray2;
    float* gpuResult;

    float* h;

    if (gpuVersion)
    {
        // allocate two gpuArray 1 and gpuArray 2 and gpuResult array on GPU

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!

        // copy the array once to the device

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!

        // allocate an array to download the results of all threads
        h = new float[MAX_BLOCKS * MAX_THREADS];
    }

    const int num_iters = 100;
    double finalDotProduct;

    if (!gpuVersion)
    {
        printf("cpu: ");
        for (int iter = 0; iter < num_iters; ++iter)
        {
            finalDotProduct = 0.0;
            for (int i = 0; i < dim; ++i)
            {
                finalDotProduct += cpuArray1[i] * cpuArray2[i];
            }
        }
    }
    else
    {

        // CUDA version here
        printf("gpu: ");

        // a simplistic way of splitting the problem into threads
        dim3 blockGrid(MAX_BLOCKS);
        dim3 threadBlock(MAX_THREADS);

        for (int iter = 0; iter < num_iters; ++iter)
        {
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult, gpuArray1, gpuArray2, dim);
        }

        // download and combine the results of multiple threads on the CPU

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
    }

    printf("Result: %f\n", finalDotProduct);

    if (gpuVersion)
    {

        // cleanup GPU memory

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!

        delete[] h;
    }

    delete[] cpuArray2;
    delete[] cpuArray1;

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
