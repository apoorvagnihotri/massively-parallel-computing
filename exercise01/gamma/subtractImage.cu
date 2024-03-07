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
#include <device_launch_parameters.h>


#include <fstream>
#include <iostream>

#include "PPM.hh"

using namespace ppm;


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_THREADS 128

//-------------------------------------------------------------------------------

/* compute the difference between the two provided images and store the
 results in img3[blockIdx.x * blockDim.x + threadIdx.x].
 Each thread computes on pixel element.
 */
__global__ void subtractKernel(const float* img1, const float* img2,  float* img3, int _w)
{
    int x = blockIdx.x * MAX_THREADS + threadIdx.x;
    int y = blockIdx.y;
    int pos = y * _w + x;

    if (x < _w)
    {
        img3[pos] = abs(img1[pos] - img2[pos]);
    }
}

//-------------------------------------------------------------------------------
//#define DEBUG 1

int main(int argc, char* argv[])
{
    int acount = 1; // parse command line

//#if DEBUG
//    argv = { "subtractImage.exe", "vase.ppm", "vase_blur.ppm", "new.ppm" };
//    argc = 4;
//#endif
    if (argc < 4)
    {
        printf("usage: %s <inImg> <inImg2> <outImg>\n", argv[0]);
        exit(1);
    }

    float* img1;
    float* img2;

    bool success = true;
    int w, h;
    success &= readPPM(argv[acount++], w, h, &img1);
    if (!success) {
        exit(1);
    }
    success &= readPPM(argv[acount++], w, h, &img2);
    if (!success) {
        exit(1);
    }

    int nPix = w * h;

    float* gpuImg1;
    float* gpuImg2;
    float* gpuImg3;

    //-------------------------------------------------------------------------------
    printf("Executing the GPU Version\n");
    // copy the image to the device
    cudaMalloc((void**)&gpuImg1, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImg2, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImg3, nPix * 3 * sizeof(float));
    cudaMemcpy(gpuImg1, img1, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuImg2, img2, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // calculate the block dimensions
    dim3 threadBlock(MAX_THREADS);
    // select the number of blocks vertically (*3 because of RGB)
    dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);
    printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);

    subtractKernel << <blockGrid, threadBlock >> > (gpuImg1, gpuImg2, gpuImg3, w * 3);

    // Initialize img3 with a dynamic memory allocation
    float* img3 = new float[nPix * 3];

    // download result
    cudaMemcpy(img3, gpuImg3, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuImg1);
    cudaFree(gpuImg2);
    cudaFree(gpuImg3);
       

    writePPM(argv[acount++], w, h, (float*)img3);

    // Free the memory allocated for img3
    delete[] img3;

    checkCUDAError("end of program");

    printf("  done\n");
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
