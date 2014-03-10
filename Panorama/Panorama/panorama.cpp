#include<iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

unsigned int numImages = 0;

double scale = 0;
double sscale = 0;
double aspect = 0;
double wrapedImgScale = 0;

vector <string> imageNames;
vector <ImageFeatures> features;
vector <Mat> images;
vector <Size> orgImagesizes;

void setup()
{
	cout << "In setup: ";

	imageNames.push_back("photo0.JPG");
	imageNames.push_back("photo1.JPG");
	imageNames.push_back("photo2.JPG");
	imageNames.push_back("photo3.JPG");
	imageNames.push_back("photo4.JPG");
	imageNames.push_back("photo5.JPG");
	imageNames.push_back("photo6.JPG");

	numImages = imageNames.size();

	images.resize(numImages);
	features.resize(numImages);
	orgImagesizes.resize(numImages);

	cout << "images" << numImages << endl;
}

void findingFeatures()
{
	cout << "Finding features .." << endl;

	Mat img, img2;
    FeaturesFinder* finder = new SurfFeaturesFinder();

	for(unsigned int i = 0; i < numImages; i++)
	{
		img = imread(imageNames[i]);
		orgImagesizes[i] = img.size();

		if (img.empty())
		{
			cout << "Unable to open the image " << imageNames[i] << endl;
			return;
		}

		scale = min(1.0, sqrt(0.6 * 1e6 / img.size().area()));
		cv::resize(img, img2, Size(), scale, scale);

		sscale = min(1.0, sqrt(0.1 * 1e6 / img.size().area()));
		aspect = sscale / scale;

		(*finder)(img2, features[i]);
		cout << "\t #features for img#" << i << " " << features[i].keypoints.size() << endl;
		features[i].img_idx = i;

		cv::resize(img, img2, Size(), scale, scale);
		images[i] = img2.clone();
	}

	finder->collectGarbage();
	img.release();
	img2.release();

}  

int main()
{
	setup();

	findingFeatures();
	
	getchar();
	return 0;
}
