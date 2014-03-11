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

vector <string> imageNames;
vector <ImageFeatures> features;
vector <Mat> images;
vector <Size> orgImagesizes;
vector <MatchesInfo> pairwiseMatches;
vector <int> indices;
vector <CameraParams> cameras;
vector <Size> sizes;
vector <Point> corners;
vector <Mat> masksWarped;
vector <Mat> imagesWarped;
vector <Mat> masks;
vector <Mat> imagesWarpedFloat;

unsigned int numImages = 0;

double scale = 0;
double sscale = 0;
double aspect = 0;
double warpedImgScale = 0;

int blendType = Blender::MULTI_BAND;
int straighten = 0;

Ptr <RotationWarper> warper;
Ptr <ExposureCompensator> compensator;

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

	cout << "#images " << numImages << endl;
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

		cv::resize(img, img2, Size(), sscale, sscale);
		images[i] = img2.clone();
	}

	finder->collectGarbage();
	img.release();
	img2.release();

}  

void pairwiseMatching(void)
{
	cout << "In pairwise matching.. ";

	BestOf2NearestMatcher matcher(false, 0.3f);
	matcher(features, pairwiseMatches);
	matcher.collectGarbage();

	indices = leaveBiggestComponent(features, pairwiseMatches, 1.0f);

	vector <Mat> subsetImages;
	vector <String> subsetImageNames;
	vector <Size> subsetOrgImageSizes;

	for(unsigned int i = 0; i < indices.size(); i++)
	{
		subsetImageNames.push_back(imageNames[indices[i]]);
		subsetImages.push_back(images[indices[i]]);
		subsetOrgImageSizes.push_back(orgImagesizes[indices[i]]);
	}

	images = subsetImages;
	imageNames = subsetImageNames;
	orgImagesizes = subsetOrgImageSizes;

	numImages = images.size();

	if(numImages < 2)
	{
		cout << "Need more images" << endl;
	}

	cout << "#images remaining.." << numImages << endl;

}

void homographyEstimation()
{
	cout << "In homography estimation .." << endl;
	 
	HomographyBasedEstimator estimator;
	estimator(features, pairwiseMatches, cameras);

	for(unsigned int i= 0; i < cameras.size(); i++)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
}

void bundleAdjustment(void)
{
	cout << "In bundle adjustment .. " << endl;

	BundleAdjusterRay bAdjuster;
	bAdjuster.setConfThresh(1.0f);

	Mat_<uchar> refineMask = Mat::zeros(3, 3, CV_8U);

	refineMask(0, 0) = 1; refineMask(0, 1) = 1; refineMask(0, 2) = 1;
	refineMask(1, 1) = 1;refineMask(1, 2) = 1;

	bAdjuster.setRefinementMask(refineMask);

	(bAdjuster)(features, pairwiseMatches, cameras);

}

void computeWarpedImageScale(void)
{
	vector <double> focals;

	for(unsigned int i= 0; i < cameras.size(); i++)
	{
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end()); 

	double medianFocal = 0;

	int size = focals.size();
	if(size % 2 == 0)
	{
		medianFocal = (focals[size/2 - 1] + focals[size/2]) / 2;
	}
	else
	{
		medianFocal = focals[size/2];
	}

	warpedImgScale = medianFocal * 0.5;
}

void straightening(void)
{
	cout << "In straightening.." << endl;
	
	vector <Mat> rmats;

	for(unsigned int i = 0; i < cameras.size(); i++)
	{
		rmats.push_back(cameras[i].R);
	}

	waveCorrect(rmats, WAVE_CORRECT_HORIZ);

	for(unsigned int i = 0; i < cameras.size(); i++)
	{
		cameras[i].R = rmats[i];
	}
}

void warpingImages(void)
{
	cout << "In warping images.." << endl;

	corners.resize(numImages);	 
	masksWarped.resize(numImages);
	imagesWarped.resize(numImages);
	sizes.resize(numImages);
	vector <Mat> masks(numImages);

	for(unsigned int i = 0; i < numImages; i++)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	warper = new SphericalWarper(static_cast<float>(aspect*warpedImgScale));

	for(unsigned int i = 0; i < numImages; i++)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		K(0,0) *= (float) aspect; K(0,2) *= (float) aspect;
		K(1,1) *= (float) aspect; K(1,2) *= (float) aspect;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, imagesWarped[i]);
		sizes[i] = imagesWarped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masksWarped[i]);
	}

	imagesWarpedFloat.resize(numImages);

	for(unsigned int i = 0; i < numImages; i++)
	{
		imagesWarped[i].convertTo(imagesWarpedFloat[i], CV_32F);
	}
}

void exposureCompensation(void)
{
	cout << "In exposure compensation .." << endl;

	compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	compensator->feed(corners, imagesWarped, masksWarped);
}

void findingSeamMasks(void)
{
	cout << " In finding seam .." << endl;

	Ptr <SeamFinder> seamFinder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	seamFinder->find(imagesWarpedFloat, corners, masksWarped);

	images.clear();
	imagesWarped.clear();
	imagesWarpedFloat.clear();
}

void compositingImages(void)
{
	cout << "In compositing images .." << endl;

	Mat img, img2, imgWarped, imgWarpedS;
	Mat dilatedMask, seamMask, mask, maskWarped;
	Ptr <Blender> blender;
	bool isComposeScaleSet = false;
	double composeScale = 0.0;

	RotationWarper* warper;

	for (unsigned int i = 0; i < numImages; i++)
	{
		cout << "\t compositing image" << indices[i] << endl;

		img = imread(imageNames[i]);

		if(!isComposeScaleSet)
		{
			isComposeScaleSet = true;

			composeScale = 1.0;
			double composeWorkAspect = composeScale / scale;

			warpedImgScale *= composeWorkAspect;

			warper = new SphericalWarper(static_cast<float>(warpedImgScale));

			//update corners and sizes
			for(unsigned int k = 0; k < numImages; k++)
			{
				cameras[k].focal *= composeWorkAspect;
				cameras[k].ppx *= composeWorkAspect;
				cameras[k].ppy *= composeWorkAspect;

				Size sz = orgImagesizes[k];

				if(std::abs(composeScale - 1) > 1e-1)
				{
					sz.width = cvRound(orgImagesizes[k].width * composeScale);
					sz.height = cvRound(orgImagesizes[k].height * composeScale);
				}

				Mat K;
				cameras[k].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[k].R);
				corners[k] = roi.tl();
				sizes[k] = roi.size();
			}
		}
		if (abs(composeScale - 1) > 1e-1)
			cv::resize(img, img2, Size(), composeScale, composeScale);
		else
			img2 = img;

		img.release();
		Size img_size= img2.size();
		
		Mat K;
        cameras[i].K().convertTo(K, CV_32F);

		warper->warp(img2, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, imgWarped);

		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));

		//Warp the current image mask
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, maskWarped);

		//Compensate exposure
		compensator->apply(i, corners[i], imgWarped, maskWarped);

		imgWarped.convertTo(imgWarpedS, CV_16S);
		imgWarped.release();
		img2.release();
		mask.release();

		cv::dilate(masksWarped[i], dilatedMask, Mat());
		cv::resize(dilatedMask, seamMask, maskWarped.size());
		maskWarped = seamMask & maskWarped;

		if (blender.empty())
		{
			blender = Blender::createDefault(blendType, 0);
			Size dst_sz = resultRoi(corners,sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
			if (blend_width < 1.f || blendType == Blender::NO)
			{
				blender = Blender::createDefault(Blender::NO, 0);
				cout << "No blending " << endl;
			}
			else if(blendType == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
				mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
				cout << "MULTI_BAND " << mb->numBands() << endl;
			}
			else if(blendType == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
				fb->setSharpness(1.f/blend_width);
				cout << "FEATHER " << fb->sharpness() << endl;
			}
			blender->prepare(corners, sizes);
		}
		//Blend the current image
		blender->feed(imgWarpedS,maskWarped, corners[i]);
	}
	Mat result, result_mask;
	blender->blend(result, result_mask);

	imwrite("output.jpg", result);
}

int main()
{
	setup();

	findingFeatures();
	
	pairwiseMatching();

	homographyEstimation();

	bundleAdjustment();

	computeWarpedImageScale();

	if(straighten == 1)
		straightening();

	warpingImages();

	exposureCompensation();

	findingSeamMasks();

	compositingImages();

	cout << "Finished stitching.. Enter any key to close\n";

	getchar();
	return 0;
}

