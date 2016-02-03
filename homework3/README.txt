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

/* Function declaration Appendix */





///////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////


