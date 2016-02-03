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

#include <iostream>
#include <vector>
#include <math.h>
//opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;

/*
 * Summary: check if value is in the vector v.
 * Parameters: vector v, int value.
 * Output: bool value of if the value is in the vector; int value of the index of the matched pair
 * Note: This function is for determing if a keypoint has already been matched.
 */

pair<bool,int> InVector(vector<pair<int,int>> v, int value){
    for(int i=0;i<v.size();i++){
        if(v[i].second==value){
            return make_pair(true, i);
        }
    }
    return make_pair(false,-1);
}
/*
 * This is another version for Point2f
 */
bool InVector(vector<Point2f> samples, Point2f test){
    vector<Point2f>::iterator it;
    
    if(find(samples.begin(), samples.end(), test)!=samples.end()){
        return true;
    }
    else{
        return false;
    }

}
/*
 * Summary: calculate Euclidean distance of SIFT feature vectors of two images; then find the best matched keypoints between object image and test image.
 * Parameters: feature descriptors of two images
 * Output: matched pairs vector<>.
 */
vector<pair<int,int>>  matchPairs(Mat features, Mat testFeatures){
    //for each vector of the keypoints in the image, find its match in the test image
    //define matched pairs
    vector<pair<int,int>> matchIdx;
    //define smallest distance of each matched pair
    vector<float> smallestDistances;
    for(int i=0;i<features.rows;i++){
        //define the smallest and 2nd smallest Euclidean distance of two vectors
        float smallestDis=0, secondSmallestDis=0;
        //record the index of the vectors in the test image
        int smallestIdx=0, secondSmallestIdx=0;
        //take the first two vectors in the test image as the initial smallest and second smallest distances
        for(int col=0;col<features.cols;col++){
            smallestDis+=(features.at<float>(i, col)-testFeatures.at<float>(0, col))*(features.at<float>(i, col)-testFeatures.at<float>(0, col));
            smallestIdx=0;
            secondSmallestDis+=(features.at<float>(i, col)-testFeatures.at<float>(1, col))*(features.at<float>(i, col)-testFeatures.at<float>(1, col));
            secondSmallestIdx=1;
            //cout<<features.at<float>(i, col)<<endl;
        }
        //cout<<smallestDis<<endl;
        smallestDis=sqrt(smallestDis);
        secondSmallestDis=sqrt(secondSmallestDis);
        if (smallestDis>secondSmallestDis) {
            float tempDis=secondSmallestDis;
            secondSmallestDis=smallestDis;
            smallestDis=tempDis;
            smallestIdx=1;
            secondSmallestIdx=0;
        }
        
        //travesal the vectors of the test image
        for(int j=2;j<testFeatures.rows;j++){
            //calculate distance between i-th f1 and j-th f2
            float checkDis=0;
            for(int col=0;col<testFeatures.cols;col++){
                checkDis+=(features.at<float>(i, col)-testFeatures.at<float>(j, col))*(features.at<float>(i, col)-testFeatures.at<float>(j, col));
                
            }
            checkDis=sqrt(checkDis);
            //update smallest and 2nd smallest distances and indexes
            if (checkDis<smallestDis) {
                secondSmallestDis=smallestDis;
                smallestDis=checkDis;
                secondSmallestIdx=smallestIdx;
                smallestIdx=j;
            }
            else if (checkDis<secondSmallestDis){
                secondSmallestDis=checkDis;
                secondSmallestIdx=j;
            }
        }
        //if the second best match is close to the best match, consider the match is not robust and discard the match; set threshold=0.8
        //two vectors can be matched to one vector, avoid this by checking if already matched
        if (smallestDis/secondSmallestDis<0.8 && !InVector(matchIdx, smallestIdx).first ) {
            matchIdx.push_back(make_pair(i, smallestIdx));
            smallestDistances.push_back(smallestDis);
        }
        //if already matched, check if this match is better or not;if better, replace the original match pair
        if (smallestDis/secondSmallestDis<0.8 && InVector(matchIdx, smallestIdx).first ){
            if (smallestDis<smallestDistances[InVector(matchIdx, smallestIdx).second]) {
                //if found new matched pair has smaller distance, update the vector
                matchIdx[InVector(matchIdx, smallestIdx).second]=make_pair(i, smallestIdx);

            }
            //matchIdx.push_back(-1);//if not good match, set index as -1
            //cout<<smallestIdx<<endl;
        }
    }
    return matchIdx;
    
}
/*
 * Summary: This function selects non-repeated random numbers from [1,totalNum].
 * Parameters: number of selection; total number select from.
 * Output: an array of random needed number of numbers
 */

vector<int> RandomIndex(int totalNum, int neededNum){
    vector<int> indexes;
    srand((int)time(NULL));
    for(int i=0; i<totalNum; i++){
        //number range [1,totalNum]. Ex: 1 to 48.
        indexes.push_back(i);
    }
    for(int i=totalNum-1; i>0; i--){
        //swap the values beteen location i and random location
        swap(indexes[i], indexes[rand()%i]);
    }
    
    //indexes contain neededNum of random numbers
    indexes.erase(indexes.begin()+neededNum, indexes.end());
    return indexes;
}

/*
 * Summary: This function is for solving the linear equations for Homography coefficients.
 * Parameters: keypoints and keypoints of test image;sampled keypoints and sampled test image keypoints
 * Output: a homography model fits the 5 keypoints
 */
Mat HomographyModel(vector<Point2f> p1, vector<Point2f> p2,vector<Point2f> sampleP1, vector<Point2f> sampleP2){
    //define linear equations
    // a has 8 rows, a_x_1,a_y_1,a_x_2,a_y_2... contains coordinates
    //h has 9 Homography coefficients
    Mat a(10,9,CV_32F),h=Mat::zeros(9,1,CV_32F);
    //b is right side of equation
    Mat b=Mat::zeros(10, 1, CV_32F);
    for(int i=0;i<sampleP1.size();i++){
        //assign each column value
        //one pairs generate two equations
        //a_x * h = 0
        a.at<float>(i*2,0)=-sampleP1[i].x;
        a.at<float>(i*2,1)=-sampleP1[i].y;
        a.at<float>(i*2,2)=-1;
        a.at<float>(i*2,3)=0;
        a.at<float>(i*2,4)=0;
        a.at<float>(i*2,5)=0;
        a.at<float>(i*2,6)=sampleP1[i].x*sampleP2[i].x;
        a.at<float>(i*2,7)=sampleP1[i].y*sampleP2[i].x;
        a.at<float>(i*2,8)=sampleP2[i].x;
        //a_y * h =0
        a.at<float>(i*2+1,0)=0;
        a.at<float>(i*2+1,1)=0;
        a.at<float>(i*2+1,2)=0;
        a.at<float>(i*2+1,3)=-sampleP1[i].x;
        a.at<float>(i*2+1,4)=-sampleP1[i].y;
        a.at<float>(i*2+1,5)=-1;
        a.at<float>(i*2+1,6)=sampleP1[i].x*sampleP2[i].y;
        a.at<float>(i*2+1,7)=sampleP1[i].y*sampleP2[i].y;
        a.at<float>(i*2+1,8)=sampleP2[i].y;
    }
    solve( a, b, h,DECOMP_NORMAL);
    Mat homography(3, 3, CV_32F, &h);
    for (int i=0;i<homography.rows; i++) {
        cout<<homography.row(i)<<endl;
    }
    //for (int i=0;i<homography.rows; i++) {
    //    cout<<homography.row(i)/homography.at<float>(2,2)<<endl;
    //}
    //cout<<SatisfyModel(samplePoints[0],sampleTestPoints[0],homography,10)<<endl;
    
    /* solve() test case
     int aa[2][2]={2,1,1,1};
     int bb[2]={4,3};
     Mat A(2, 2, CV_32F,&aa);
     Mat B(2, 1, CV_32F,&bb);
     Mat x(2, 1, CV_32F);
     solve(A, B, x);
     for (int i=0;i<x.rows; i++) {
     cout<<x.row(i)<<endl;
     }
     */
    return homography;
}
/*
 * Summary: This function is for checking if the test point satisfied with the model.
 * Parameters: test point, real correct point, homography model, threshold t.
 * Output: bool value of if satisfying
 */
bool SatisfyModel(Point2f test, Point2f p,  Mat model,int t){
    vector<Point2f> projected;
    vector<Point2f> testPoint;
    testPoint.push_back(test);
    perspectiveTransform(testPoint, projected, model);
    cout<<p<<" "<<projected[0]<<endl;
    float dis= sqrt((p.x-projected[0].x)*(p.x-projected[0].x)+(p.y-projected[0].y)*(p.y-projected[0].y));
    if (dis<t) {
        return true;
    }
    else{
        return false;
    }
    
}
/*
 * Summary: This function is for caluculating the error of the model compared to the corresponding correct points in test image.
 * Parameters: inliers and their pairs of test image, homography model
 * Output: error distance of this model
 */
float CalculateError(vector<Point2f> p1,vector<Point2f> p2,Mat model){
    float errorDis=0;
    vector<Point2f> projected;
    perspectiveTransform(p1, projected, model);
    for (int i=0; i<p2.size(); i++) {
        errorDis+=sqrt((p2[i].x-projected[i].x)*(p2[i].x-projected[i].x)+(p2[i].y-projected[i].y)*(p2[i].y-projected[i].y));
    }
    return errorDis/p2.size();
}
/*
 * Summary: This function is the realization of RANSAC algorithm.
 * Parameters: keypoints and keypoints of test image
 * Output: a best homography model fits for the keypoints
 */
Mat algorithmRANSAC(vector<Point2f> p1, vector<Point2f> p2){
    int n=5;//the least number of points fits the model
    float p=0.99;//probability of good outcome
    int k=log(1-p)/log(1-pow(0.5, n));//iteration times
    float t=10;//threshold of perspect transform error
    int d=p1.size()*0.8;//above this number of points included, the model is defined as good model
    int iteration=0;
    Mat bestModel;//best model
    vector<Point2f> consensusSet;//inliers in the best model
    float bestError=INT32_MAX;//error rate of the best model
    while (iteration<k) {
        //define points satisfied this model
        vector<Point2f> inliers;
        //define inliers' corresponding pair in the test image
        vector<Point2f> inliersPair;
        //generate 5 random pairs
        vector<int> keypointIdx=RandomIndex(int(p1.size()),5);
        vector<Point2f> samplePoints, sampleTestPoints;
        for(int i=0;i<keypointIdx.size();i++){
            samplePoints.push_back(p1[keypointIdx[i]]);
            inliers.push_back(p1[keypointIdx[i]]);
            sampleTestPoints.push_back(p2[keypointIdx[i]]);
        }
        Mat model=HomographyModel(p1, p2,samplePoints,sampleTestPoints);
        //travesal each keypoints
        for(int i=0;i<p1.size();i++){
            //find points satisfying the model
            if (!InVector(inliers, p1[i]) && SatisfyModel(p1[i], p2[i], model,t) ) {
                    inliers.push_back(p1[i]);
                    inliersPair.push_back(p2[i]);
            }
        }
        //if enough inliers points found, calculate how good is the model
        if (inliers.size()>=d) {
            float modelError=CalculateError(inliers, inliersPair, model);
            //if error is less, assign this model as the best model
            if (modelError<bestError) {
                model.copyTo(bestModel);
                consensusSet=inliers;
                bestError=modelError;
                
            }
        }
        k++;//continuing finding
    }
    return bestModel;
}
int main(int argc, const char * argv[]) {
    if(argc!=3){
        cout<<"please enter correct number of arguments, for example: ./574HW3 arg1 arg2 "<<endl;
        exit(0);
    }
    //Read images
    Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat testImg = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    //define SIFT variables
    Mat descriptors, testDescriptors;
    vector<KeyPoint> keypoints, testKeypoints;
    SIFT sift;
    
    /* SIFT feature extraction */
    
    sift.SIFT::operator()(img, noArray(), keypoints, descriptors);
    sift.SIFT::operator()(testImg, noArray(), testKeypoints, testDescriptors);
    ///////////////////////////////////////////////////////////////////////////
    /* find matching pairs */
    
    vector<pair<int,int>> pairs=matchPairs(descriptors, testDescriptors);
    /* display two images in one window and draw matched pairs line */
    //define new larger image
    Mat imgPanel=Mat::zeros(MAX(img.rows, testImg.rows),img.cols+testImg.cols,img.type());
    Mat colorImgPanel(imgPanel.size(),CV_8UC3);
    //copy imges into one image using ROI
    Mat leftImg(imgPanel,Rect(0,0,img.cols,img.rows));
    img.copyTo(leftImg);
    Mat rightImg(imgPanel,Rect(img.cols,0,testImg.cols,testImg.rows));
    testImg.copyTo(rightImg);
    //for displaying red matching lines
    cvtColor(imgPanel, colorImgPanel, CV_GRAY2RGB);
    //draw matched pairs on the image
    for(int i=0;i<pairs.size() ;i++){
            //coordinates are changed based on the  image panel
            testKeypoints[pairs[i].second].pt.x=testKeypoints[pairs[i].second].pt.x+img.cols;
            //filled circles for keypoints
            circle(colorImgPanel, testKeypoints[pairs[i].second].pt, 2, Scalar(0,0,255),-1,8,0);
            circle(colorImgPanel, keypoints[pairs[i].first].pt, 2, Scalar(0,0,255),-1,8,0);
            //draw lines between matching pairs keypoints and testKeypoints
            line(colorImgPanel, keypoints[pairs[i].first].pt,testKeypoints[pairs[i].second].pt,Scalar(0,0,255));
        
    }
    
    imshow("Matching pairs",colorImgPanel);
    waitKey(0);
    ///////////////////////////////////////////////////////////////////////////
    /* locate object */
   
    //define homogenous coordinates
    vector<Point2f> homoPoints, homoTestPoints;
    //change the coordinates origin to the image center
    for(int i=0;i<pairs.size() ;i++){
        Point2f p1, p2;
        p1.x=keypoints[pairs[i].first].pt.x-img.cols/2;
        p1.y=keypoints[pairs[i].first].pt.y+img.rows/2;
        //p1.z=1;
        p2.x=testKeypoints[pairs[i].second].pt.x-img.cols-testImg.cols/2;
        p2.y=testKeypoints[pairs[i].second].pt.y+testImg.rows/2;
        //p2.z=1;
        homoPoints.push_back(p1);
        homoTestPoints.push_back(p2);
    }
    //apply RANSAC
    Mat homography;
    homography=algorithmRANSAC(homoPoints, homoPoints);
    //use findHomography results
    //homography=findHomography(homoPoints, homoTestPoints,CV_RANSAC);
    cout<<"Homography matrix:"<<endl;
    for (int i=0; i<homography.rows; i++) {
        cout<<homography.row(i)<<endl;
    }
    //project the keypoints on object to the test image
    vector<Point2f> projectedKeypoints(homoPoints.size());
    perspectiveTransform(homoPoints, projectedKeypoints, homography);
    //re-create the image for displaying objects
    cvtColor(imgPanel, colorImgPanel, CV_GRAY2RGB);
    for(int i=0;i<projectedKeypoints.size();i++){
        //change coordinates back to image coordinates
        projectedKeypoints[i].x+=testImg.cols/2+img.cols;
        projectedKeypoints[i].y-=testImg.rows/2;
        homoTestPoints[i].x+=testImg.cols/2+img.cols;
        homoTestPoints[i].y-=testImg.rows/2;
        //draw feature points on to the test image
        circle(colorImgPanel, projectedKeypoints[i], 2, Scalar(0,255,0),-1,8,0);
        circle(colorImgPanel, homoTestPoints[i], 2, Scalar(0,0,255),-1,8,0);
    }
    destroyAllWindows();
    imshow("locate object results", colorImgPanel);
    waitKey(0);

        return 0;
}
