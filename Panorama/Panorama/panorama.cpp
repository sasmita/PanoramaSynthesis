#include<iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
using namespace std;
using namespace cv;
using namespace cv::detail;

int numImages = 0;

vector <string> imageNames;
vector <ImageFeatures>features;

void setup()
{
	imageNames.push_back("photo1.JPG");
	imageNames.push_back("photo2.JPG");
	imageNames.push_back("photo3.JPG");
	imageNames.push_back("photo4.JPG");
	imageNames.push_back("photo5.JPG");

	numImages = imageNames.size();
}

void findFeatures()
{
	features.resize(numImages);

	Mat img, img2;

	for(int i =0; i < numImages; i++)
	{
		img = imread(imageNames[i]);

		if (img.empty())
		{
			cout << "Unable to open the image" << imageNames[i] << endl;
			return;
		}

		double scale = 0.2;
		resize(img, img2, Size(), scale, scale);

		FeaturesFinder* finder = new SurfFeaturesFinder();

		(*finder)(img, features[i]);

		cout << "#features for img#" << i << "" << features[i].keypoints.size() << endl;
	}     
}

int main()
{
	setup();

	findFeatures();
	
	getchar();
	return 0;
}
