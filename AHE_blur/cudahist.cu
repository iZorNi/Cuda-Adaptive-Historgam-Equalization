#include "device_launch_parameters.h"

#include "stdafx.h"

#define TAG_MASK 0xFFFFFFFFU
inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag)
{
    atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag)
{
    addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogramKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM_THREADBLOCK_MEMORY / HISTOGRAM_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();

    for (uint bin = threadIdx.x; bin < HISTOGRAM_BIN_COUNT; bin += HISTOGRAM_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM_BIN_COUNT] & TAG_MASK;
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM_BIN_COUNT + bin] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogramKernel(
    uint *d_Histogram,
    uint *d_PartialHistograms,
    uint histogramCount
)
{
    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
    {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM_BIN_COUNT];
    }

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        d_Histogram[blockIdx.x] = data[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
//histogramkernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM_COUNT = 240;
static uint *d_PartialHistograms;

//Internal memory allocation
extern "C" void initHistogram(void)
{
    Utilities::getError(cudaMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM_COUNT * HISTOGRAM_BIN_COUNT * sizeof(uint)));
}

//Internal memory deallocation
extern "C" void closeHistogram(void)
{
    Utilities::getError(cudaFree(d_PartialHistograms));
}

extern "C" void histogram(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
    assert(byteCount % sizeof(uint) == 0);
    histogramKernel<<<PARTIAL_HISTOGRAM_COUNT, HISTOGRAM_THREADBLOCK_SIZE>>>(
        d_PartialHistograms,
        (uint *)d_Data,
        byteCount / sizeof(uint)
    );
    //getLastCudaError("histogramKernel() execution failed\n");

    mergeHistogramKernel<<<HISTOGRAM_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
        d_Histogram,
        d_PartialHistograms,
        PARTIAL_HISTOGRAM_COUNT
    );
    //getLastCudaError("mergeHistogramKernel() execution failed\n");
}