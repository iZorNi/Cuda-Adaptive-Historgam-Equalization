#pragma once
#include "DataHandler.h"
#include "stdafx.h"


class AheFilter
{
public:
	AheFilter();
	~AheFilter();
	int runAHE(uchar* input, uchar* output, int rows, int cols);
private:

private:
	HostImage inputImage, outputImage;
	DeviceImage dev_image;
	GaussianFilter filter;
private:
	void createGaussianFilter();
	void prepareDeviceMemory();
	void convertRGBtoHSV();
	void equalizeHistogram(uchar* value, int height, int width);
	void releaseDeviceMemory();
	void copyResultToHost();
};

