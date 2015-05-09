#include "AheFilter.h"



AheFilter::AheFilter()
{
	dev_image.width = 0;
	dev_image.height = 0;
	dev_image.hue = NULL;
	dev_image.saturation = NULL;
	dev_image.value = NULL;
	dev_image.valueBlurred = NULL;
	dev_image.valueContrast = NULL;
	dev_image.filter = NULL;
	dev_image.inputImage = NULL;
	dev_image.outputImage = NULL;
	dev_image.filterWidth = 0;
	createGaussianFilter();
	isMemoryAllocated = 0;
	isImageSizeChanged = 0;
}


AheFilter::~AheFilter()
{
}

int AheFilter::runAHE(uchar* input, uchar* output, int rows, int cols)
{
	DataHandler::prepareImage(input, rows, cols, inputImage);
	DataHandler::prepareImage(output, rows, cols, outputImage);
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
	
	
	//FILE *file;
	//char* fname = "value.txt";
	//file = fopen(fname, "w");
	//size_t size = rows*cols*sizeof(uchar);
	//uchar* value = (uchar*)malloc(size);
	//cudaMemcpy(value, dev_image.value, size, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		fprintf(file, "%d ", value[i*cols + j]);
	//	}
	//	fprintf(file, "\n");
	//}
	//fclose(file);
	//
	//size = rows*cols*sizeof(int);
	//fname = "hue.txt";
	//file = freopen(fname, "w", file);
	//int* hue = (int*)malloc(size);
	//cudaMemcpy(hue, dev_image.hue, size, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		fprintf(file, "%d ", hue[i*cols + j]);
	//	}
	//	fprintf(file, "\n");
	//}
	//fclose(file);

	//size = rows*cols*sizeof(float);
	//fname = "saturation.txt";
	//file = freopen(fname, "w", file);
	//float* saturation = (float*)malloc(size);
	//cudaMemcpy(saturation, dev_image.saturation, size, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		fprintf(file, "%f ", saturation[i*cols + j]);
	//	}
	//	fprintf(file, "\n");
	//}
	//fclose(file);

	copyResultToHost();
	releaseDeviceMemory();
	return 0;
}

void AheFilter::prepareDeviceMemory()
{
	if ((inputImage.rows != dev_image.height) || (inputImage.cols != dev_image.width))
	{ 
		isImageSizeChanged = 1;
	}
	if (isImageSizeChanged || !isMemoryAllocated)
	{
		if (isImageSizeChanged)
		{
			releaseDeviceMemory();
		}
		if (!isMemoryAllocated)
		{
			allocateDeviceMemory();
			initDevImage();
			initGaussianFilterDev();
		}
	}
	else
	{
		initDevImage();
	}
	
}

void AheFilter::allocateDeviceMemory()
{
	size_t size = inputImage.cols * inputImage.rows * sizeof(float);

	Utilities::getError(cudaMalloc((void**)&dev_image.saturation, size));
	Utilities::getError(cudaMemset(dev_image.saturation, 0, size));

	size = inputImage.cols * inputImage.rows * sizeof(int);

	Utilities::getError(cudaMalloc((void**)&dev_image.hue, size));
	Utilities::getError(cudaMemset(dev_image.hue, 0, size));

	size = inputImage.cols * inputImage.rows * sizeof(unsigned char);

	Utilities::getError(cudaMalloc((void**)&dev_image.value, size));
	Utilities::getError(cudaMalloc((void**)&dev_image.valueBlurred, size));
	Utilities::getError(cudaMalloc((void**)&dev_image.valueContrast, size));
	Utilities::getError(cudaMalloc((void**)&dev_image.mask, size));

	Utilities::getError(cudaMemset(dev_image.value, 0, size));
	Utilities::getError(cudaMemset(dev_image.valueBlurred, 0, size));
	Utilities::getError(cudaMemset(dev_image.valueContrast, 0, size));
	Utilities::getError(cudaMemset(dev_image.mask, 0, size));

	size = inputImage.cols * inputImage.rows * 3 * sizeof(unsigned char);

	Utilities::getError(cudaMalloc((void**)&dev_image.inputImage, size));
	Utilities::getError(cudaHostRegister(inputImage.data, size, 0));

	Utilities::getError(cudaMalloc((void**)&dev_image.outputImage, size));
	Utilities::getError(cudaHostRegister(outputImage.data, size, 0));

	isMemoryAllocated = 1;
}

void AheFilter::releaseDeviceMemory()
{
	if (isMemoryAllocated)
	{
		Utilities::getError(cudaFree(dev_image.hue));
		Utilities::getError(cudaFree(dev_image.saturation));
		Utilities::getError(cudaFree(dev_image.value));
		Utilities::getError(cudaFree(dev_image.valueBlurred));
		Utilities::getError(cudaFree(dev_image.valueContrast));
		Utilities::getError(cudaFree(dev_image.mask));
		Utilities::getError(cudaFree(dev_image.inputImage));
		Utilities::getError(cudaFree(dev_image.outputImage));
		isMemoryAllocated = 0;
	}
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

void AheFilter::initDevImage()
{
	dev_image.width = inputImage.cols;
	dev_image.height = inputImage.rows;


	size_t size = inputImage.cols * inputImage.rows * 3 * sizeof(unsigned char);
	Utilities::getError(cudaMemcpy(dev_image.inputImage, inputImage.data, size, cudaMemcpyHostToDevice));
	Utilities::getError(cudaMemset(dev_image.outputImage, 0, size));
	//Utilities::getError(cudaMemcpy(dev_image.outputImage, inputImage.data, size, cudaMemcpyHostToDevice));

}

void AheFilter::initGaussianFilterDev()
{
	size_t size = filter.weight.size() * sizeof(float);
	Utilities::getError(cudaMalloc(&dev_image.filter, size));
	Utilities::getError(cudaMemcpy(dev_image.filter, &filter.weight[0], size, cudaMemcpyHostToDevice));
	dev_image.filterWidth = filter.width;
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