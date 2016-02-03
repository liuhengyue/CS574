//
//  CSCI574_HW3.cpp
//  574HW3
//  This program achieves the SIFT and locating the object from images in 3 steps:
//  1. SIFT feature extraction
//  2. Find match pairs of interest points
//  3. Locate object
//  Created by LiuHengyue on 10/18/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//
//

/*
 *   This is the demo of extracting SIFT features
 *   Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
 *   Mat descriptors;
 *   vector<KeyPoint> keypoints;
 *   sift.SIFT::operator()(img, noArray(), keypoints, descriptors);
 */

/*
 *   This is the demo of finding matched pairs
 *   //Mat descriptors, testDescriptors are the descriptors obtained in last step
 *   vector<pair<int,int>> pairs=matchPairs(descriptors, testDescriptors);
 */

/*
 *   This is the demo of locating object
 *   // These two vectors are the key points pairs in second step
 *   vector<Point2f> homoPoints, homoTestPoints;
 *   homography=algorithmRANSAC(homoPoints, homoPoints);
 */

/* 
 *   This is the demo of drawing
 *   circle(colorImgPanel, projectedKeypoints[i], 2, Scalar(0,255,0),-1,8,0);
 */

/* Function declaration Appendix *////////////////////////////////////////////////////////////////////////////////////
/* * Summary: check if value is in the vector v. * Parameters: vector v, int value. * Output: bool value of if the value is in the vector; int value of the index of the matched pair * Note: This function is for determing if a keypoint has already been matched. */pair<bool,int> InVector(vector<pair<int,int>> v, int value);
///////////////////////////////////////////////////////////////////////////////////
/* * This is another version for Point2f */
bool InVector(vector<Point2f> samples, Point2f test);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: calculate Euclidean distance of SIFT feature vectors of two images; then find the best matched keypoints between object image and test image. * Parameters: feature descriptors of two images * Output: matched pairs vector<>. */
vector<pair<int,int>>  matchPairs(Mat features, Mat testFeatures);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: This function selects non-repeated random numbers from [1,totalNum]. * Parameters: number of selection; total number select from. * Output: an array of random needed number of numbers */vector<int> RandomIndex(int totalNum, int neededNum);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: This function is for solving the linear equations for Homography coefficients. * Parameters: keypoints and keypoints of test image;sampled keypoints and sampled test image keypoints * Output: a homography model fits the 5 keypoints */
Mat HomographyModel(vector<Point2f> p1, vector<Point2f> p2,vector<Point2f> sampleP1, vector<Point2f> sampleP2);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: This function is for checking if the test point satisfied with the model. * Parameters: test point, real correct point, homography model, threshold t. * Output: bool value of if satisfying */
bool SatisfyModel(Point2f test, Point2f p,  Mat model,int t);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: This function is for caluculating the error of the model compared to the corresponding correct points in test image. * Parameters: inliers and their pairs of test image, homography model * Output: error distance of this model */
float CalculateError(vector<Point2f> p1,vector<Point2f> p2,Mat model);

///////////////////////////////////////////////////////////////////////////////////
/* * Summary: This function is the realization of RANSAC algorithm. * Parameters: keypoints and keypoints of test image * Output: a best homography model fits for the keypoints */
Mat algorithmRANSAC(vector<Point2f> p1, vector<Point2f> p2);
