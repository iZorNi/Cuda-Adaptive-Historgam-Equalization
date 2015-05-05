#include "AheFilter.h"



AheFilter::AheFilter()
{
}


AheFilter::~AheFilter()
{
}

int AheFilter::runAHE(uchar* input, uchar* output, int rows, int cols)
{
	DataHandler::prepareInputImage(input, rows, cols, inputImage);

	createGaussianFilter();
	prepareDeviceMemory();
	RunAHEKernel(dev_image.outputImage,
				dev_image.inputImage,
				dev_image.hue,
				dev_image.saturation,
				dev_image.value,
				dev_image.valueBlurred,
				dev_image.valueContrast,
				dev_image.mask,
				dev_image.filter,
				dev_image.filterWidth,
				dev_image.height,
				dev_image.width);
	copyResultToHost();
	releaseDeviceMemory();
	return 0;
}

void AheFilter::prepareDeviceMemory()
{
	size_t size = dev_image.width * dev_image.height * sizeof(float);
	Utilities::getError(cudaMalloc(&dev_image.hue, size));
	Utilities::getError(cudaMemset(dev_image.hue, 0, size));
	
	Utilities::getError(cudaMalloc(&dev_image.saturation, size));
	Utilities::getError(cudaMemset(dev_image.saturation, 0, size));
	
	size = dev_image.width * dev_image.height * sizeof(unsigned char);

	Utilities::getError(cudaMalloc(&dev_image.value, size));
	Utilities::getError(cudaMemset(dev_image.value, 0, size));
	
	Utilities::getError(cudaMalloc(&dev_image.valueBlurred, size));
	Utilities::getError(cudaMemset(dev_image.valueBlurred, 0, size));
	
	Utilities::getError(cudaMalloc(&dev_image.valueContrast, size));
	Utilities::getError(cudaMemset(dev_image.valueContrast, 0, size));
	
	Utilities::getError(cudaMalloc(&dev_image.mask, size));
	Utilities::getError(cudaMemset(dev_image.mask, 0, size));

	size = inputImage.cols * inputImage.rows * 3 * sizeof(unsigned char);
	Utilities::getError(cudaMalloc(&dev_image.inputImage, size));
	Utilities::getError(cudaHostRegister(inputImage.data, size, 0));
	Utilities::getError(cudaMemcpy(dev_image.inputImage, inputImage.data, size, cudaMemcpyHostToDevice));
	
	Utilities::getError(cudaMalloc(&dev_image.outputImage, size));
	Utilities::getError(cudaHostRegister(outputImage.data, size, 0));
	Utilities::getError(cudaMemcpy(dev_image.outputImage, inputImage.data, size, cudaMemcpyHostToDevice));
	
	size = filter.weight.size() * sizeof(float);
	Utilities::getError(cudaMalloc(&dev_image.filter, size));
	Utilities::getError(cudaMemcpy(dev_image.filter, &filter.weight[0], size, cudaMemcpyHostToDevice));
	dev_image.filterWidth = filter.width;
}

void AheFilter::releaseDeviceMemory()
{
	Utilities::getError(cudaFree(&dev_image.hue));
	Utilities::getError(cudaFree(&dev_image.saturation));
	Utilities::getError(cudaFree(&dev_image.value));
	Utilities::getError(cudaFree(&dev_image.valueBlurred));
	Utilities::getError(cudaFree(&dev_image.valueContrast));
	Utilities::getError(cudaFree(&dev_image.mask));
	Utilities::getError(cudaFree(&dev_image.inputImage));
	Utilities::getError(cudaFree(&dev_image.outputImage));
}

void AheFilter::copyResultToHost()
{
	size_t size = inputImage.cols*inputImage.rows * 3 * sizeof(uchar);
	Utilities::getError(cudaMemcpy(outputImage.data, dev_image.outputImage, size, cudaMemcpyDeviceToHost));
}

void AheFilter::createGaussianFilter()
{
	const int   width = 9;				// This is stencil width, or how many pixels in each row or column should we include in blurring function. SHould be odd.
	const float sigma = 2.f;				// Standard deviation of the Gaussian distribution.

	const int	half = width / 2;
	float		sum = 0.f;

	filter.width = width;

	// Create convolution matrix
	filter.weight.resize(width * width);

	// Calculate filter sum first
	for (int r = -half; r <= half; ++r)
	{
		for (int c = -half; c <= half; ++c)
		{
			// e (natural logarithm base) to the power x, where x is what's in the brackets
			float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
			int idx = (r + half) * width + c + half;

			filter.weight[idx] = weight;
			sum += weight;
		}
	}

	// Normalize weight: sum of weights must equal 1
	float normal = 1.f / sum;

	for (int r = -half; r <= half; ++r)
	{
		for (int c = -half; c <= half; ++c)
		{
			int idx = (r + half) * width + c + half;

			filter.weight[idx] *= normal;
		}
	}
}

void AheFilter::equalizeHistogram(uchar* value, int rows, int cols)
{
	const int levels = 255;
	int size = rows*cols;
	int histogram[levels];		//histogram
	for (int i = 0; i < size; i++)
	{
		histogram[value[i]]++;
	}
	int cdf[levels];			//cumulative distribution function
	int scaled[levels];			//scaled values
	cdf[0] = histogram[0];
	scaled[0] = 0;
	for (int i = 1; i <= levels; i++)
	{
		cdf[i] = cdf[i - 1] + histogram[i];
		scaled[i] = round( (cdf[i] - cdf[0])*(levels - 1) / ((size - cdf[0]) * 1.0f) );
	}
	for (int i = 0; i < size; i++)
	{
		value[i] = scaled[value[i]];
	}
}