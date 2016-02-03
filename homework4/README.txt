//
//  574HW4.cpp
//  CSCI 574 Homework 4
//  This program achieves the reconstruction of 3D scene structure and camera motion.
//  Then check the parallelism of reconstructed lines.
//  Created by LiuHengyue on 11/8/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//


/*************************************************************************
 * CheckParallelism
 * Function: Given a set of 3D points, check parallelism for line(p1,p2) and line(p3,p4), 
 * then show the mean square error of distortion of parallel lines.
 * Input: 3D points array;index of point 1 2 3 4.
 *************************************************************************/
 
void CheckParallelism(Mat points3d,int p1, int p2,int p3,int p4);

/*************************************************************************
 * DrawImage
 * Function: Given a set of 2D points, draw the wire-frame object in a image.
 * Input: image window name; image panel Mat; 2D points array.
 *************************************************************************/

void DrawImage(const char* name,Mat panel,vector<Point> points2d);

