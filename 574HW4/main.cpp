//
//  main.cpp
//  574HW4
//
//  Created by LiuHengyue on 11/8/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//

#include <iostream>
//opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    Mat test(200,200,CV_32F);
    circle(test, Point(100,100), 5, Scalar(255,0,0));
    imshow("test", test);
    waitKey();
    return 0;
}
