#include "DataHandler.h"


DataHandler::DataHandler()
{
}


DataHandler::~DataHandler()
{
}

int DataHandler::imgRead(char* input, HostImage &image)
{
	cv::Mat img;
	img = cv::imread(input, CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!img.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	image.data = img.data;
	image.cols = img.cols;
	image.rows = img.rows;
	return 0;
}

int DataHandler::imgWrite(char* output, HostImage &image)
{
	cv::Mat img;
	img.data = image.data;
	img.cols = image.cols;
	img.rows = image.rows;
	try
	{
		cv::imwrite(output, img);
	}
	catch (std::runtime_error& ex)
	{
		fprintf(stderr, "Exception saving image: %s\n", ex.what());
		return 1;
	}

	return 0;
}

void DataHandler::prepareInputImage(uchar* input, int height, int width, HostImage &image)
{
	image.data = input;
	image.cols = width;
	image.rows = height;
}