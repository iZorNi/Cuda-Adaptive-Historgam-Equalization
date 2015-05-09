#include "device_launch_parameters.h"

#include "stdafx.h"
#include "math_functions.h"

__global__
void histogram_equalization(
const uchar* const		inputChannel,
uchar* const			outputChannel,
int								rows,
int								cols
);

__global__
void gaussian_blur(
	uchar* const		blurredChannel,						// return value: blurred channel
	const uchar* const	inputChannel,						// channel from the original image
	int							rows,
	int							cols,
	const float* const			filterWeight,						// gaussian filter weights. The weights look like a bell shape.
	int							filterWidth							// number of pixels in x and y directions for calculating average blurring
	);

__global__
void create_mask(
	const uchar* const			inputChannel,
	const uchar* const			blurredChannel,
	uchar* const						mask,
	int									rows,
	int									cols
	);

__global__
void mask_overlay(
	uchar* const				inputChannel,
	const uchar* const			contrastChannel,
	const uchar* const			mask,
	int									rows,
	int									cols
	);

__global__
void convert_to_hsv(
const uchar*		image,
int*				hue,
float*				saturation,
uchar*				value,
int					rows,
int					cols);

__global__
void convert_to_rgb(
uchar*				image,
int*				hue,
float*				saturation,
uchar*				value,
int					rows,
int					cols);

__global__
void convert_to_uint(const uchar* const input, uint* const output, int rows, int cols);

__global__
void generate_LUT(uint* const lut, const uint* const cdf, int levels, int nPixels);

__global__
void equalize_channel(const uchar* const input,
uchar* const output,
const uint* const lut,
int rows, int cols);

//launches histogram equliztion
void histogramEqualization(
	const uchar* const		inputChannel,
	uchar* const			outputChannel,
	int						rows,
	int						cols);

//calculates image histogram
void calculateHistogram(uint* const input, uint* const d_Histogram, int rows, int cols);

//generates LUT to equlize image
void generateLUT(uint* const lut,
	const uint* const hist,
	uint* const cdf,
	int levels, int nPixels);



extern "C" void RunAHEKernel(
	uchar* const			outputImage,					// Return value: rgba image 
	const uchar* const		originalImage,
	int* const				hue,	
	float* const			saturation,
	uchar* const			value,
	uchar* const			valueBlurred,
	uchar* const			valueContrast,
	uchar* const			mask,
	const float* const		filterWeight,					// gaussian filter weights. The weights look like a bell shape.
	int						filterWidth,					// number of pixels in x and y directions for calculating average blurring
	int						rows,							// image size: number of rows
	int						cols							// image size: number of columns
	)
{
	const char* func = "RunAHEKernel";

	cudaError hr = cudaSuccess;

	static const int BLOCK_WIDTH = 32;						// threads per block; because we are setting 2-dimensional block, the total number of threads is 32^2, or 1024
															// 1024 is the maximum number of threads per block for modern GPUs.

	int x = static_cast<int>(ceilf(static_cast<float>(cols) / BLOCK_WIDTH));
	int y = static_cast<int>(ceilf(static_cast<float>(rows) / BLOCK_WIDTH));

	const dim3 grid(x, y, 1);								// number of blocks
	const dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);			// block width: number of threads per block

	//const int grid = x*y;							// number of blocks
	//const int block = BLOCK_WIDTH*BLOCK_WIDTH;		// block width: number of threads per block

	// Convert RGB to HSV
	convert_to_hsv <<< grid, block >>>(originalImage, hue, saturation, value, rows, cols);
	Utilities::getError(cudaDeviceSynchronize());
	//CHECK_CUDA_ERROR(hr, func, "convert_to_hsv kernel failed.");

	// Call convolution kernel for channel
	gaussian_blur <<< grid, block >>>(valueBlurred, value, rows, cols, filterWeight, filterWidth);
	Utilities::getError(cudaDeviceSynchronize());																
	//CHECK_CUDA_ERROR(hr, func, "gaussian_blur kernel failed ");

	// Create mask of local contrast
	create_mask <<< grid, block >>>(value, valueBlurred, mask, rows, cols);
	Utilities::getError(cudaDeviceSynchronize());
	//hr = cudaDeviceSynchronize();																
	//CHECK_CUDA_ERROR(hr, func, "create mask kernel failed ");

	// Equalize image histogram
	histogramEqualization(value, valueContrast, rows, cols);

	// Overlay mask of local contrast
	mask_overlay << < grid, block >> >(value, valueContrast, mask, rows, cols);
	Utilities::getError(cudaDeviceSynchronize());

	//hr = cudaDeviceSynchronize();																
	//CHECK_CUDA_ERROR(hr, func, "mask_overlay kernel failed ");
	
	//Utilities::getError(cudaMemcpy(value, valueContrast, rows*cols*sizeof(uchar), cudaMemcpyDeviceToDevice));
	//Utilities::getError(cudaDeviceSynchronize());
	// Recombine HSV channels into an RGB image
	convert_to_rgb <<< grid, block >>>(outputImage, hue, saturation, value, rows, cols);
	Utilities::getError(cudaDeviceSynchronize());
	//hr = cudaDeviceSynchronize();																
	//CHECK_CUDA_ERROR(hr, func, "convert_to_rgb kernel failed.");
}


__global__
void gaussian_blur(
	uchar* const				blurredChannel,						// return value: blurred channel
	const uchar* const			inputChannel,						// channel from the original image
	int							rows,
	int							cols,
	const float* const			filterWeight,						// gaussian filter weights. The weights look like a bell shape.
	int							filterWidth							// number of pixels in x and y directions for calculating average blurring
	)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int c = blockIdx.x * blockDim.x + threadIdx.x;		// current column


	if ((r >= rows) || (c >= cols))
	{
		return;
	}

	int			  half = filterWidth / 2;
	float		  blur = 0.f;								// will contained blurred value
	int			  width = cols - 1;
	int			  height = rows - 1;

	for (int i = -half; i <= half; ++i)					// rows
	{
		for (int j = -half; j <= half; ++j)				// columns
		{
			// Clamp filter to the image border
			int		h = min(max(r + i, 0), height);
			int		w = min(max(c + j, 0), width);

			// Blur is a product of current pixel value and weight of that pixel.
			// Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
			int		idx = w + cols * h;											// current pixel index
			float	pixel = static_cast<float>(inputChannel[idx]);

			idx = (i + half) * filterWidth + j + half;
			float	weight = filterWeight[idx];

			blur += pixel * weight;
		}
	}

	blurredChannel[c + r * cols] = static_cast<uchar>(blur);
}

__global__
void create_mask(
	const uchar* const			inputChannel,
	const uchar* const			blurredChannel,
	uchar* const				mask,
	int							rows,
	int							cols
	)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column
	if ((x >= rows) || (y >= cols))
	{
		return;
	}

	int idx = y + cols * x;		// current pixel index
	mask[idx] = inputChannel[idx] - blurredChannel[idx];
}

__global__
void mask_overlay(
	uchar* const				inputChannel,
	const uchar* const			contrastChannel,
	const uchar* const			mask,
	int							rows,
	int							cols
	)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column
	if ((x >= rows) || (y >= cols))
	{
		return;
	}

	int idx = y + cols * x;		// current pixel index
	inputChannel[idx] = static_cast<uchar> ((1 - mask[idx] / 255.f)*contrastChannel[idx] + mask[idx] / 255.f * inputChannel[idx]);
}

__global__
void convert_to_rgb(
uchar*				image,
int*				hue,
float*				saturation,
uchar*				value,
int					rows,
int					cols)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column

	if ((x >= rows) || (y >= cols))
	{
		return;
	}

	int idx = y + cols * x;		// current pixel index

	int r, g, b;
	//copy values to local variables
	int v = value[idx];
	float s = saturation[idx];
	int h = hue[idx];

	int vmin = lrintf((1 - s) * v);
	int a = lrintf((v - vmin)*((h % 60) / 60.f));
	int vinc = vmin + a;
	int vdec = v - a;
	int hi = floor(static_cast<double>(h / 60));
	switch (hi)
	{
	case 0:
		r = v;
		g = vinc;
		b = vmin;
		break;
	case 1:
		r = vdec;
		g = v;
		b = vmin;
		break;
	case 2:
		r = vmin;
		g = v;
		b = vinc;
		break;
	case 3:
		r = vmin;
		g = vdec;
		b = v;
		break;
	case 4:
		r = vinc;
		g = vmin;
		b = v;
		break;
	case 5:
		r = v;
		g = vmin;
		b = vdec;
		break;
	};
	//save result to image
	image[idx * 3] = static_cast<uchar>(b);
	image[idx * 3 + 1] = static_cast<uchar>(g);
	image[idx * 3 + 2] = static_cast<uchar>(r);
}


// converts RGB image into HSV
__global__
void convert_to_hsv(
const uchar*		image,
int*				hue,
float*				saturation,
uchar*				value,
int					rows,
int					cols)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column

	if ((x >= rows) || (y >= cols))
	{
		return;
	}

	int idx = y + cols * x;		// current pixel index

	// Copy channels to the local variables
	uchar r = image[idx * 3 + 2];
	uchar g = image[idx * 3 + 1];
	uchar b = image[idx * 3];

	uchar max, min;

	// get max, min values between r, g, b
	if (r >= g)
	{
		if (r >= b)
		{
			max = r;
			min = (g >= b) ? b : g;
		}
		else
		{
			max = b;
			min = g;
		}
	}
	else
	{
		if (g >= b)
		{
			max = g;
			min = (r >= b) ? b : r;
		}
		else
		{
			max = b;
			min = r;
		}
	}

	value[idx] = max;
	saturation[idx] = (max == 0) ? 0.f : 1.f - ((double)min) / ((double)max);
	if (max == min)
	{
		hue[idx] = 0;
	}
	else
	{
		if (max == r)
			if (g >= b)
			{
				hue[idx] = 60 * (g - b) / ((double)(max - min));
			}
			else
			{
				hue[idx] = 60 * (g - b) / ((double)(max - min)) + 360;
			}
		else
		{
			if (max == g)
			{
				hue[idx] = 60 * (b - r) / ((double)(max - min)) + 120;
			}
			else
			{
				hue[idx] = 60 * (r - g) / ((double)(max - min)) + 240;
			}
		}
	}
}

void histogramEqualization(
	const uchar* const		inputChannel,
	uchar* const			outputChannel,
	int						rows,
	int						cols)
{
	const char* func = "histogramEqualization";

	cudaError hr = cudaSuccess;

	static const int BLOCK_WIDTH = 32;		// threads per block; because we are setting 2-dimensional block, the total number of threads is 32^2, or 1024
	// 1024 is the maximum number of threads per block for modern GPUs.

	int x = static_cast<int>(ceilf(static_cast<float>(cols) / BLOCK_WIDTH));
	int y = static_cast<int>(ceilf(static_cast<float>(rows) / BLOCK_WIDTH));


	const dim3 grid(x, y, 1);								// number of blocks
	const dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);			// block width: number of threads per block

	//int grid = x*y;							// number of blocks
	//int block = BLOCK_WIDTH*BLOCK_WIDTH;		// block width: number of threads per block
	
	uint* d_Data;
	uint* d_Histogram;
	uint* d_lut;
	uint* cdf;

	Utilities::getError(cudaMalloc((void **)&d_Histogram, HISTOGRAM_BIN_COUNT * sizeof(uint)));
	Utilities::getError(cudaMalloc((void **)&d_Data, rows*cols * sizeof(uint)));


	convert_to_uint <<<grid,block>>> (inputChannel, d_Data, rows, cols);
	hr = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(hr, func, "convert_to_uchar kernel failed.");

	calculateHistogram(d_Data, d_Histogram, rows, cols);
	Utilities::getError(cudaFree(d_Data));

	Utilities::getError(cudaMalloc((void**)&d_lut, HISTOGRAM_BIN_COUNT * sizeof(uint)));
	Utilities::getError(cudaMalloc((void**)&cdf, HISTOGRAM_BIN_COUNT * sizeof(uint)));

	generateLUT(d_lut, d_Histogram, cdf, HISTOGRAM_BIN_COUNT, rows*cols);
	hr = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(hr, func, "convert_to_uchar kernel failed.");
	
	Utilities::getError(cudaFree(d_Histogram));
	Utilities::getError(cudaFree(cdf));

	equalize_channel << <grid, block >> > (inputChannel, outputChannel, d_lut, rows, cols);
	hr = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(hr, func, "equalize_channel kernel failed.");

	Utilities::getError(cudaFree(d_lut));



}

void calculateHistogram(uint* const input, uint* const d_Histogram, int rows, int cols )
{
	const char* func = "calculateHistogram";

	cudaError hr = cudaSuccess;

	initHistogram();
	histogram(d_Histogram, input, rows*cols*sizeof(uchar));
	hr = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(hr, func, "histogram kernel failed.");
	closeHistogram();
}

void generateLUT(uint* const lut, 
				 const uint* const hist,
				 uint* const cdf, 
				 int levels, int nPixels)
{
	const char* func = "generateLUT";

	cudaError hr = cudaSuccess;
	size_t size = HISTOGRAM_BIN_COUNT*sizeof(uint);
	uint* h_cdf = (uint*)malloc(size);
	uint* h_hist = (uint*)malloc(size);
	
	Utilities::getError(cudaMemcpy(h_hist, hist, size, cudaMemcpyDeviceToHost));
	Utilities::getError(cudaDeviceSynchronize());
	
	h_cdf[0] = h_hist[0];
	//lut[0] = 0;
	for (int i = 1; i < HISTOGRAM_BIN_COUNT; i++)
	{
		h_cdf[i] = h_cdf[i - 1] + h_hist[i];
	}
	
	Utilities::getError(cudaMemcpy(cdf, h_cdf, size, cudaMemcpyHostToDevice));

	int block = 32;
	int grid = ((HISTOGRAM_BIN_COUNT-1) / block + 1);
	
	generate_LUT << <grid, block >> >(lut, cdf, HISTOGRAM_BIN_COUNT, nPixels);
	Utilities::getError(cudaDeviceSynchronize());
	//hr = cudaDeviceSynchronize();
	//CHECK_CUDA_ERROR(hr, func, "generate_LUT kernel failed.");
	
	free(h_cdf);
	free(h_hist);
}

__global__
void convert_to_uint(const uchar* const input, uint* const output, int rows, int cols)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column
	if ((x >= rows) || (y >= cols))
	{
		return;
	}
	int idx = y + cols * x;		// current pixel index
	output[idx] = static_cast<uint>(input[idx]);
}

__global__
void generate_LUT(uint* const lut, const uint* const cdf, int levels, int nPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= levels)
		return;

	lut[i] = round((cdf[i] - cdf[0])*(levels - 1) / (double)(nPixels - cdf[0]));
}

__global__
void equalize_channel(const uchar* const input,
					  uchar* const output, 
					  const uint* const lut, 
					  int rows, int cols)
{
	int x = blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int y = blockIdx.x * blockDim.x + threadIdx.x;		// current column
	if ((x >= rows) || (y >= cols))
	{
		return;
	}
	int idx = y + cols * x;		// current pixel index
	output[idx] = lut[input[idx]];
}