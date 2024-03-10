#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// No-operation sorting kernel
//
// This takes in an unordered set, and builds a dummy bucket representation around it
// It does not perform any actual sorting!
//
// This kernel must be launched with a 1,1 configuration (1 grid block, 1 thread).

//static __global__ void noSortKernel(const Position* inputPositions, int numInputPositions,
//                                    Position* outputPositions, Bucket* outputBuckets)
//{
//  int numBuckets = BucketsPerAxis * BucketsPerAxis;
//
//  // Copy contents of input positions into output positions
//
//  for (int i = 0; i < numInputPositions; ++i)
//      outputPositions[i] = inputPositions[i];
//
//  // Set up the set of buckets to cover the output positions evenly
//
//  for (int i = 0; i < numBuckets; i++)
//  {
//      Bucket& bucket = outputBuckets[i];
//
//      bucket.startIndex = numInputPositions * i / numBuckets;
//      bucket.numElements = (numInputPositions * (i + 1) / numBuckets) - bucket.startIndex;
//  }
//}

// !!! missing !!!
// Kernels needed for sortPositionsIntoBuckets(...)

static __global__ void histogramKernel(const Position* inputPositions, int numInputPositions, int* noOfElementsInBucket, int* placeInRespectiveBucket, int* belongsToBucket)
{
    __shared__ int bucketSize;
    if (threadIdx.x == 0)
    {
    	bucketSize = 0;
    }
    __syncthreads();

    float lowY = (float)blockIdx.y / gridDim.y;
    float highY = (float)(blockIdx.y + 1) / gridDim.y;

    float lowX = (float)blockIdx.x / gridDim.x;
    float highX = (float)(blockIdx.x + 1) / gridDim.x;


    for (int i = threadIdx.x; i < numInputPositions; i += blockDim.x)
    {
        if (lowX <= inputPositions[i].x && inputPositions[i].x < highX && lowY <= inputPositions[i].y && inputPositions[i].y < highY) // [a,b) -> one can get into multiple buckets if at boundary
        {
            placeInRespectiveBucket[i] = atomicAdd(&bucketSize, 1);
            belongsToBucket[i] = gridDim.x * blockIdx.y + blockIdx.x;
        }
    }
    
    __syncthreads();

    if (threadIdx.x == 0)
    {
        noOfElementsInBucket[gridDim.x * blockIdx.y + blockIdx.x] = bucketSize;
    }

}

//must be called with 1 block
static __global__ void prefixSumKernel(int* values, int numValues)
{
    extern __shared__ float sharedValues[];

    if (threadIdx.x == 0)
    {
        sharedValues[0] = 0;
    }

    // load a into shared memory
    for (int i = threadIdx.x; i < numValues; i += blockDim.x)
    {
        sharedValues[i + 1] = values[i];
    }


    int stride;
    for (stride = 1; stride * 2 <= numValues + 1; stride *= 2)
    {
        __syncthreads();

        for (int i = numValues - (threadIdx.x * stride * 2) - stride; i > 0; i -= blockDim.x * stride * 2)
        {
            sharedValues[i + stride] += sharedValues[i];
        }
    }

    for (int b_stride = stride; b_stride  >= 1; b_stride /= 2)
        {
            __syncthreads();

            for (int i = numValues - (threadIdx.x * b_stride * 2) - b_stride; i >= b_stride; i -= blockDim.x * b_stride * 2)
            {
                sharedValues[i] += sharedValues[i - b_stride];
            }
        }

    __syncthreads();
    // write out memory
	for (int i = threadIdx.x; i < numValues + 1; i += blockDim.x)
	{
		values[i] = sharedValues[i];
	}

}

static __global__ void sortKernel(const Position* inputPositions, int numInputPositions, int* bucketHistogram, int* placeInRespectiveBucket, int* belongsToBucket, Position* outputPositions, Bucket* outputBuckets)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int threadSize = gridDim.x * blockDim.x * blockDim.y;

    for (int i = threadNo; i < numInputPositions; i += threadSize)
    {
        outputPositions[bucketHistogram[belongsToBucket[i]] + placeInRespectiveBucket[i]] = inputPositions[i];
    }

    for (int i = threadNo; i < numBuckets; i += threadSize)
    {
        Bucket& bucket = outputBuckets[i];

        bucket.startIndex = bucketHistogram[i];
        bucket.numElements = bucketHistogram[i + 1] - bucketHistogram[i];
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    //
    // 2. Given the histogram, compute where in the output array that each bucket begins, and how
    // large it is
    //    (perform prefix summation over the histogram)
    //
    // 3. Given the start of each bucket within the output array, scatter elements from the input
    //    array into the output array
    //
    // Your new sort implementation should be able to handle at least 10 million entries, and
    //  run in reasonable time (the reference implementations does the job in 200 milliseconds).

    //=================  Your code here =====================================
    // !!! missing !!!
    


    // Instead of sorting, we will now run a dummy kernel that just duplicates the
    //  output positions, and constructs a set of dummy buckets. This is just so that
    //  the test program will not crash when you try to run it.
    //
    // This kernel is run single-threaded because it is throw-away code where performance
    //  does not matter; after all, the purpose of the lab is to replace it with a
    //  proper sort algorithm instead!

    //========== Remove this code when you begin to implement your own sorting algorithm ==========
	CudaArray<int> cudaNoOfElementsInBucket(BucketsPerAxis * BucketsPerAxis + 1);
	CudaArray<int> cudaPlaceInRespectiveBucket(cudaInputPositions.size());
	CudaArray<int> cudaBelongsToBucket(cudaInputPositions.size());




    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);
    histogramKernel<<<grid, numThreads >>>(cudaInputPositions.cudaArray(), cudaInputPositions.size(), cudaNoOfElementsInBucket.cudaArray(), cudaPlaceInRespectiveBucket.cudaArray(), cudaBelongsToBucket.cudaArray());
    cudaDeviceSynchronize();
    prefixSumKernel<<<1,numThreads, (BucketsPerAxis * BucketsPerAxis + 1) * sizeof(int)>>>(cudaNoOfElementsInBucket.cudaArray(), BucketsPerAxis*BucketsPerAxis);
    cudaDeviceSynchronize();
    sortKernel << <grid, numThreads  >> > (cudaInputPositions.cudaArray(), cudaInputPositions.size(), cudaNoOfElementsInBucket.cudaArray(), cudaPlaceInRespectiveBucket.cudaArray(), cudaBelongsToBucket.cudaArray(), cudaOutputPositions.cudaArray(), cudaOutputPositionBuckets.cudaArray());

    cudaDeviceSynchronize();


    // Copy results back to host memory
    //std::vector<int> cpuNoOfElementsInBucket;
    //cpuNoOfElementsInBucket.resize(cudaNoOfElementsInBucket.size());
    //cudaNoOfElementsInBucket.copyFromCuda(&(*cpuNoOfElementsInBucket.begin()));
    //printf("a");
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);

        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
