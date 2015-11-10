//
//  574HW4.cpp
//  CSCI 574 Homework 4
//  This program achieves the reconstruction of 3D scene structure and camera motion.
//  Then check the parallelism of reconstructed lines.
//  Created by LiuHengyue on 11/8/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//

#include <iostream>
#include <vector>
//opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;

/*************************************************************************
 * CheckParallelism
 * Function: Given a set of 3D points, check parallelism for line(p1,p2) and line(p3,p4), 
 * then show the mean square error of distortion of parallel lines.
 * Input: 3D points array;index of point 1 2 3 4.
 *************************************************************************/
 
void CheckParallelism(Mat points3d,int p1, int p2,int p3,int p4){
    cout<<"line("<<p1<<","<<p2<<") compares with line("<<p3<<","<<p4<<"):"<<endl;
    //calculate parameters a b c of 2 3d line euquations ax+by+cz+d=0, and check if parallel by ratio
    float x=(points3d.at<float>(0,p1-1)-points3d.at<float>(0,p2-1))/(points3d.at<float>(0,p3-1)-points3d.at<float>(0,p4-1));
    float y=(points3d.at<float>(1,p1-1)-points3d.at<float>(1,p2-1))/(points3d.at<float>(1,p3-1)-points3d.at<float>(1,p4-1));
    float z=(points3d.at<float>(2,p1-1)-points3d.at<float>(2,p2-1))/(points3d.at<float>(2,p3-1)-points3d.at<float>(2,p4-1));
    float MSE=((x-(x+y+z)/3)*(x-(x+y+z)/3)+(y-(x+y+z)/3)*(y-(x+y+z)/3)+(z-(x+y+z)/3)*(z-(x+y+z)/3))/3;
    cout<<"Ratio of x,y,z axis coefficient: "<<x<<" "<<y<<" "<<z<<". "<<endl;
    cout<<"Mean square error of parrallelism: "<<MSE*100<<"%."<<endl<<endl;

}

/*************************************************************************
 * DrawImage
 * Function: Given a set of 2D points, draw the wire-frame object in a image.
 * Input: image window name; image panel Mat; 2D points array.
 *************************************************************************/

void DrawImage(const char* name,Mat panel,vector<Point> points2d){
    //connectivity is  (1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7) and (4, 8)
    //draw lines
    line(panel, points2d[0], points2d[1], Scalar(0,0,255));//parallels 1
    line(panel, points2d[1], points2d[2], Scalar(0,0,255));//2
    line(panel, points2d[2], points2d[3], Scalar(0,0,255));//parallels 1
    line(panel, points2d[3], points2d[0], Scalar(0,0,255));//2
    line(panel, points2d[4], points2d[5], Scalar(0,0,255));//parallels 1
    line(panel, points2d[5], points2d[6], Scalar(0,0,255));//2
    line(panel, points2d[6], points2d[7], Scalar(0,0,255));//parallels 1
    line(panel, points2d[7], points2d[4], Scalar(0,0,255));//2
    line(panel, points2d[0], points2d[4], Scalar(0,0,255));//3
    line(panel, points2d[1], points2d[5], Scalar(0,0,255));//3
    line(panel, points2d[2], points2d[6], Scalar(0,0,255));//3
    line(panel, points2d[3], points2d[7], Scalar(0,0,255));//3
    //draw vertexes
    for(int i=0;i<8;i++){
        circle(panel, points2d[i], 2, Scalar(0,0,255),-1,8,0);
    }
    //show image
    imshow(name, panel);
    waitKey();

}
int main(int argc, const char * argv[]) {
    //Construct the D matrix for decomposition
    int p[12][8]={227,261,339,299,194,227,303,265,
                  341,400,277,218,341,402,275,214,
                  150,112,192,234,175,137,214,256,
                  208,197, 83, 88,228,220,105,111,
                   27, 67,133, 95, 48, 86,156,117,
                  225,225,126,123,247,246,147,145,
                   48,111,130, 62, 38,102,117, 48,
                   33, 55, 71, 46, 64, 86,105, 81,
                   85,126,235,197, 76,117,227,188,
                   93, 97,133,131,126,130,168,166,
                  227,253, 81, 66,228,253, 85, 67,
                   52, 52, 57, 57, 93, 95,102,100};
    Mat DMat(12,8,CV_32S,p);
    //define SVD matrices
    Mat W,U,Vt,D;
    //convert to float Mat
    DMat.convertTo(D, CV_32F);
    SVD::compute(D, W, U, Vt);
    //define and construct W3, U3,Vt3
    Mat W3Sqrt=Mat::zeros(3,3,CV_32F),U3(12,3,CV_32F),Vt3(3,8,CV_32F);
    W3Sqrt.at<float>(0,0)=sqrt(W.at<float>(0,0));
    W3Sqrt.at<float>(1,1)=sqrt(W.at<float>(0,1));
    W3Sqrt.at<float>(2,2)=sqrt(W.at<float>(0,2));
    U.colRange(0, 3).copyTo(U3);
    Vt.rowRange(0, 3).copyTo(Vt3);
    //define and construct A0, P0
    Mat A0(12,3,CV_32F),P0(3,8,CV_32F);
    A0=U3*W3Sqrt;
    //3D points coordinates
    P0=W3Sqrt*Vt3;
    //show 3D coordinates
    cout<<"The reconstructed 8 3D points world coordinates:"<<endl<<P0<<endl;
    //preserve parallelism with scaling the coordinates
    Mat P;
    P0.convertTo(P, CV_32S,20);
    P=P.t();
    cout<<"The reconstructed scaled 8 3D points world coordinates:"<<endl<<P<<endl;

    //define image panel,image1Panel is for showing the perspective projection image 1
    Mat panel(500,500,CV_8UC3),img1Panel(500,500,CV_8UC3);
    //define points
    vector<Point> points2d,image1;
    for(int i=0;i<P.rows;i++){
        //create points by weak perspective projection by setting Z=0 and shift by (0,200)
        points2d.push_back(Point(P.at<int>(i,0),P.at<int>(i,1)+200));
        image1.push_back(Point(p[0][i],p[1][i]));
            }
    //draw image 1
    DrawImage("image 1",img1Panel, image1);
    //draw reconstructed rectangular solid
    DrawImage("2d weak perspective projection of reconstructed scene", panel, points2d);
    //calculate slope of each lines from reconstructed 3D points
    //line (1,2) (3,4) (5,6) (7,8)
    CheckParallelism(P0,1,2,3,4);
    CheckParallelism(P0,1,2,5,6);
    CheckParallelism(P0,1,2,7,8);
    //line (2,3) (4,1) (6,7) (8,5)
    CheckParallelism(P0,2,3,4,1);
    CheckParallelism(P0,2,3,6,7);
    CheckParallelism(P0,2,3,8,5);
    //line (1,5) (2,6) (3,7) (4,8)
    CheckParallelism(P0,1,5,2,6);
    CheckParallelism(P0,1,5,3,7);
    CheckParallelism(P0,1,5,4,8);
    return 0;
}
