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
#include <cstring>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <device_launch_parameters.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

__device__ __constant__ float3 gpuClusterCol[2048];

#define THREADS 256 // this is also the size of the image (width, and h conveniently)
#define LOG_IMG_SIZE 8
#define IMG_SIZE 256
#define WINDOW 6

/* The function measures for every pixel the distance to all
 clusters, and determines the clusterID of the nearest cluster
 center. It then colors the pixel in the cluster's color.

 The cluster centers are given as an array of linear indices into
 the vector image, i.e.    _clusterInfo[0] = (x_0 + y_0 * _w).

 */
__global__ void voronoiKernel(float3 *_dst, int _w, int _h, int _nClusters, const int *_clusterInfo)
{
    // get the shared memory
    extern __shared__ int shm[];

    int nIter = _nClusters / THREADS + 1;
    // load cluster data
    for (int i = 0; i < nIter; ++i)
    {
        int pos = i * THREADS + threadIdx.x;
        if (pos < _nClusters)
        {
            shm[pos] = _clusterInfo[pos];
        }
    }

    __syncthreads();

    // compute the position within the image
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y;

    int pos = x + y * _w;

    // determine which is the closest cluster
    float minDist = 1000000.;
    int minIdx = 0;
    for (int i = 0; i < _nClusters; ++i)
    {

        float yy = shm[i] >> LOG_IMG_SIZE;
        float xx = shm[i] % IMG_SIZE;

        float dist = (x - xx) * (x - xx) + (y - yy) * (y - yy);
        if (dist < minDist)
        {
            minDist = dist;
            minIdx = i;
        }
    }

    _dst[pos].x = gpuClusterCol[minIdx].x; // globally accessed, symbol
    _dst[pos].y = gpuClusterCol[minIdx].y;
    _dst[pos].z = gpuClusterCol[minIdx].z;

    // mark the center of each cluster
    if (minDist <= 2.)
    {
        _dst[pos].x = 255;
        _dst[pos].y = 0.;
        _dst[pos].z = 0.;
    }
}

__device__ float luminance(const float4 &_col)
{
    return 0.299 * _col.x + 0.587 * _col.y + 0.114 * _col.z;
}

/** stores a 1 in _dst if the pixel's luminance is a maximum in the
WINDOW x WINDOW neighborhood
 */
__global__ void featureKernel(int *_dst, cudaTextureObject_t texImg, int _w, int _h)
{
    // compute the position within the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    float lum = luminance(tex2D<float4>(texImg, x, y));

    bool maximum = false;

    if (lum > 20)
    {
        maximum = true;
        for (int v = y - WINDOW; v < y + WINDOW; ++v)
        {
            for (int u = x - WINDOW; u < x + WINDOW; ++u)
            {

                if (lum < luminance(tex2D<float4>(texImg, u, v)))
                {
                    maximum = false;
                }
            }
        }
    }

    if (maximum)
    {
        _dst[x + y * _w] = 1;
    }
    else
    {
        _dst[x + y * _w] = 0;
    }
}

// !!! missing !!!
// Kernels for Prefix Sum calculation (compaction, spreading, possibly shifting)
// and for generating the gpuFeatureList from the prefix sum.

__global__ void reductionKernel(int *_dst, int *_src, int _nPix) // called <<((1, 256), (256), THREADS)>>
{
    // get the thread index and the global index
    int tid = threadIdx.x;
    int index = blockIdx.y * blockDim.x + threadIdx.x;

    // store the scanline corresponding to the block in shared memory.
    __shared__ int scanline[THREADS];
    scanline[tid] = _src[index];
    __syncthreads();

    // do the reduction step
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        // s is the stride, 1, 2, 4, 8, 16, ...
        if ((tid + 1) % s == 0 && tid + s < blockDim.x)
        { // the sums happen between s-1, 2s-1, 3s-1, 4s-1, ...
            scanline[tid] += scanline[tid + s];
        }
        __syncthreads();
    }

    // store the result back to global memory
    if (index < _nPix)
        _dst[index + 1] = scanline[tid];
}

__global__ void reductionKernelLast(int *_dst, int *_src, int _nPix) // called <<((1), (256), THREADS)>>
{
    // get the thread index and the global index
    int tid = threadIdx.x;
    // the last element of every row forms a scanline
    int index = threadIdx.x * blockDim.x + blockDim.x - 1;

    // store the scanline corresponding to the block in shared memory.
    __shared__ int scanline[THREADS];
    scanline[tid] = _src[index];
    __syncthreads();

    // do the reduction step
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        // s is the stride, 1, 2, 4, 8, 16, ...
        if ((tid + 1) % s == 0 && tid + s < blockDim.x)
        { // the sums happen between s-1, 2s-1, 3s-1, 4s-1, ...
            scanline[tid] += scanline[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    { // since shiftedPrefixSum by definition has 0 as first element
        _dst[0] = 0;
    }

    // store the result back to global memory,
    if (index < _nPix)
        _dst[index + 1] = scanline[tid];
}

__global__ void spreadingKernel(int *_dst, int _nPix) // called <<((1, 256), (128), THREADS)>>
{
    // get the thread index and the global index
    int tid = threadIdx.x;
    int index1 = blockIdx.y * blockDim.x + threadIdx.x;
    int index2 = blockIdx.y * blockDim.x + threadIdx.x + blockDim.x;

    // store the scanline corresponding to the block in shared memory.
    __shared__ int scanline[THREADS];
    scanline[tid] = _dst[index1];
    scanline[tid + blockDim.x] = _dst[index2];
    __syncthreads();

    // do the spreading step
    // 0, 4
    // 0, 2 & 4, 6
    // 0, 1 & 2, 3 & 4, 5 & 6, 7
    // scanline is 256 everytime, that means, stride starts with 256, then 128, 64, 32, 16, 8, 4, 2, 1
    // 0, 128,
    // 0, 64 & 128, 192,
    // 0, 32 & 64, 96 & 128, 160 & 192, 224,
    for (int s = blockDim.x * 2; s > 1; s /= 2)
    {
        // s is the stride, 256, 128, 64, 32, 16, 8, 4, 2, 1
        if (tid % s == 0 && tid + s / 2 < blockDim.x * 2)
        {
            scanline[tid + s / 2] += scanline[tid];
        }
        __syncthreads();
    }

    // store the result back to global memory
    _dst[index1] = scanline[tid];
    _dst[index2] = scanline[tid + blockDim.x];
}

__global__ void spreadingKernelLast(int *_dst, int _nPix) // called <<((1), (128), THREADS)>>
{
    // get the thread index and the global index
    int tid = threadIdx.x;
    int scanLineLen = blockDim.x * 2;
    // 255 (0th row, 1st row, 2nd row, ... 127th row)
    int index1 = threadIdx.x * scanLineLen + scanLineLen - 1;
    // ... (255 + 128*256) (128th row, 129th row, 130th row, ... 255th row)
    int index2 = (threadIdx.x + blockDim.x) * scanLineLen + scanLineLen - 1;

    // store the scanline corresponding to the block in shared memory.
    __shared__ int scanline[THREADS];
    scanline[tid] = _dst[index1];
    scanline[tid + blockDim.x] = _dst[index2];
    __syncthreads();

    // do the spreading step
    for (int s = blockDim.x * 2; s > 1; s /= 2)
    {
        // s is the stride, 256, 128, 64, 32, 16, 8, 4, 2, 1
        if (tid % s == 0 && tid + s / 2 < blockDim.x * 2)
        {
            scanline[tid + s / 2] += scanline[tid];
        }
        __syncthreads();
    }

    // store the result back to global memory
    _dst[index1] = scanline[tid];
    _dst[index2] = scanline[tid + blockDim.x];
}

/* This program detects the local maxima in an image, writes their
location into a vector and then computes the Voronoi diagram of the
image given the detected local maxima as cluster centers.

A Voronoi diagram simply colors every pixel with the color of the
nearest cluster center. */

int main(int argc, char *argv[])
{

    // parse command line
    int acount = 1;
    if (argc < 4)
    {
        printf("usage: testPrefix <inImg> <outImg> <mode>\n");
        exit(1);
    }
    string inName(argv[acount++]);
    string outName(argv[acount++]);
    int mode = atoi(argv[acount++]);

    // Load the input image
    float *cpuImage;
    int w, h;
    readPPM(inName.c_str(), w, h, &cpuImage);
    int nPix = w * h;

    // Allocate GPU memory
    int *gpuFeatureImg; // Contains 1 for a feature, 0 else
    // Can be used to do the reduction step of prefix sum calculation in place
    int *gpuPrefixSumShifted; // Output buffer containing the prefix sum
    // Shifted by 1 since it contains 0 as first element by definition
    int *gpuFeatureList;   // List of pixel indices where features can be found.
    float3 *gpuVoronoiImg; // Final rgb output image
    cudaMalloc((void **)&gpuFeatureImg, (nPix) * sizeof(int));

    cudaMalloc((void **)&gpuPrefixSumShifted, (nPix + 1) * sizeof(int));
    cudaMalloc((void **)&gpuFeatureList, 10000 * sizeof(int));

    cudaMalloc((void **)&gpuVoronoiImg, nPix * 3 * sizeof(float));

    // color map for the cluster
    float clusterCol[2048 * 3];
    float *ci = clusterCol;
    for (int i = 0; i < 2048; ++i, ci += 3)
    {
        ci[0] = 32 * i % 256;
        ci[1] = (10 * i + 128) % 256;
        ci[2] = (40 * i + 255) % 256;
    }

    cudaMemcpyToSymbol(gpuClusterCol, clusterCol, 2048 * 3 * sizeof(float));

    cudaArray *gpuTex;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&gpuTex, &floatTex, w, h);

    // pad to float4 for faster access
    float *img4 = new float[w * h * 4];

    for (int i = 0; i < w * h; ++i)
    {
        img4[4 * i] = cpuImage[3 * i];
        img4[4 * i + 1] = cpuImage[3 * i + 1];
        img4[4 * i + 2] = cpuImage[3 * i + 2];
        img4[4 * i + 3] = 0.;
    }

    // upload to array

    cudaMemcpy2DToArray(gpuTex, 0, 0, img4, w * 4 * sizeof(float), w * 4 * sizeof(float), h,
                        cudaMemcpyHostToDevice);

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpuTex;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    cout << "setup texture" << endl;
    cout.flush();

    // calculate the block dimensions
    dim3 threadBlock(THREADS);
    dim3 blockGrid(w / THREADS, h, 1);

    printf("blockDim: %d  %d \n", threadBlock.x, threadBlock.y);
    printf("gridDim: %d  %d \n", blockGrid.x, blockGrid.y);

    featureKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg, tex, w, h);

    // variable to store the number of detected features = the number of clusters
    int nFeatures;

    if (mode == 0)
    {
        ////////////////////////////////////////////////////////////
        // CPU compaction:
        ////////////////////////////////////////////////////////////

        // download result

        cudaMemcpy(cpuImage, gpuFeatureImg, nPix * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<int> features;

        float *ii = cpuImage;
        for (int i = 0; i < nPix; ++i, ++ii)
        {
            if (*ii > 0)
            {
                features.push_back(i);
            }
        }

        cout << "nFeatures: " << features.size() << endl;

        nFeatures = features.size();
        // upload feature vector

        cudaMemcpy(gpuFeatureList, &(features[0]), nFeatures * sizeof(int), cudaMemcpyHostToDevice);
    }
    else
    {
        ////////////////////////////////////////////////////////////
        // GPU compaction:
        ////////////////////////////////////////////////////////////

        // the image size is 256x256

        // !!! missing !!!
        // implement the prefixSum algorithm
        // 1. Do the reduction step for all scanlines, one scanline per block.
        dim3 gridSize(w / THREADS, h); // 1, 256
        dim3 blockSize(THREADS);       // 256
        reductionKernel<<<gridSize, blockSize, THREADS * sizeof(int)>>>(
            gpuFeatureList, gpuFeatureImg, nPix
        );

        // 2. Do the reduction step for the last elements of all scanlines, all in one block.
        gridSize = dim3(w / THREADS, 1); // 1, 1
        reductionKernelLast<<<gridSize, blockSize, THREADS * sizeof(int)>>>(
            gpuFeatureList, gpuFeatureImg, nPix
        );

        // 3. Do the spreading step for the last elements of all scanlines, all in one block.
        //    -> The last elements / elements before the scanlines have the right values now.
        blockSize = dim3(THREADS / 2); // 128
        gridSize = dim3(1, h);         // 1, 256
        spreadingKernelLast<<<gridSize, blockSize, THREADS * sizeof(int)>>>(
            gpuFeatureList, nPix
        );

        // 4. Do the spreading step for all scanlines, one scanline per block.
        gridSize = dim3(1, h / 2); // 1, 128
        spreadingKernel<<<gridSize, blockSize, THREADS * sizeof(int)>>>(
            gpuFeatureList, nPix
        );

        // !!! missing !!!
        // Make sure that gpuFeatureList is filled according to the CPU implementation
        // and that nFeatures has the correct value!
        
        // extracting the last element from the prefix sum
        cudaMemcpy(&nFeatures, gpuFeatureList + nPix, sizeof(int), cudaMemcpyDeviceToHost); // Replace index with the index of the value you want to extract
        cout << "nFeatures: " << nFeatures << endl;
        // upload feature vector
    }

    // now compute the Voronoi Diagram around the detected features.
    voronoiKernel<<<blockGrid, threadBlock, nFeatures * sizeof(int)>>>(gpuVoronoiImg, w, h,
                                                                       nFeatures, gpuFeatureList);

    // download final voronoi image.

    cudaMemcpy(cpuImage, gpuVoronoiImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // Write to disk
    writePPM(outName.c_str(), w, h, (float *)cpuImage);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(gpuTex);
    cudaFree(gpuFeatureList);
    cudaFree(gpuFeatureImg);
    cudaFree(gpuPrefixSumShifted);
    cudaFree(gpuVoronoiImg);

    delete[] cpuImage;
    delete[] img4;

    checkCUDAError("end of program");

    printf("done\n");
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
