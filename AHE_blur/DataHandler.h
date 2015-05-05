#pragma once
#include "stdafx.h"




class DataHandler
{
public:
	DataHandler();
	~DataHandler();
	static int imgRead(char* input, HostImage &image);
	static int imgWrite(char* output, HostImage &image);
	static void prepareInputImage(uchar* input, int height, int width, HostImage &image);


private:
	
};

