//  Editor: Xcode 7.0
//  main.cpp
//  CSCI574 Homework #2
//  Created by LiuHengyue on 9/26/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.

#include "main.hpp"
/*******************************
 below function is for problem 1
 ******************************/
//  Gray world Assumption Implementation
//  Given an image, export an new re-rendered image with "white" light conditions
void GrayWorldRender(){
    const char* PATH_1="/Users/liuhengyue/Desktop/574hw2/hw2/color1.bmp";
    const char* PATH_2="/Users/liuhengyue/Desktop/574hw2/hw2/color2.bmp";
    const char* SAVEPATH="";
    //enter the input: 1 2 . The index numbers represent 2 different images
    IplImage* img;//initial an image
    cout<<"please enter the index of images (1, 2) you want to re-render: ";
    cin.ignore();//dealing with '\n' input
    // Load an image from file - 1 parameter for color image, bmp colormodel is RGB
    char index;
    index=cin.get();
    if(index=='1'){
        img=cvLoadImage(PATH_1);
        SAVEPATH="/Users/liuhengyue/Desktop/574hw2/hw2/color1_rerendered.bmp";
    }
    else if(index=='2'){
        img=cvLoadImage(PATH_2);
        SAVEPATH="/Users/liuhengyue/Desktop/574hw2/hw2/color2_rerendered.bmp";
    }
    else{
        cout<<"error when load image."<<endl;
    }

    
    //IplImage* img2=cvLoadImage("/Users/liuhengyue/Desktop/574hw2/hw2/color2.bmp");
    //create a new image to store the gray world re-rendered image
    IplImage* img_rendered=cvCreateImage(cvGetSize(img), img->depth,img->nChannels);
    //IplImage* img2_rendered=cvCreateImage(cvGetSize(img), img->depth,img->nChannels);
    //cvCvtColor(img2, img2_gray, CV_RGB2GRAY);
    if(img==0)
    {
        cout<< "failed to load input image\n";
        //return -1;
    }
    //Calculate the average value for each channels of original image 'img', and assign mean values for each channel
    CvScalar avgChan=cvAvg(img);
    double avgB=avgChan.val[0];
    double avgG=avgChan.val[1];
    double avgR=avgChan.val[2];
    //Calculate the mean of 3 channels
    double avgRGB=(avgB+avgG+avgR)/3;
    //normalize each mean value, avgR(G,B) is the gain for each channel
    avgR=avgRGB/avgR;
    avgG=avgRGB/avgG;
    avgB=avgRGB/avgB;
    //search all image pixel values
    int col,row;
    for( row = 0; row < img->height; row++ )
    {
        for ( col = 0; col < img->width; col++ )
        {
            //Blue channel
            //find the re-rendered value of pixel at (row, col) in original image, and then assign the value to the new image
            if(u_int(img->imageData[img->widthStep * row + col * 3])*avgB <= 255){
                img_rendered->imageData[img_rendered->widthStep * row + col * 3]=img->imageData[img->widthStep * row + col * 3]*avgB;
            } else {
                //if the re-rendered value is over 255, make it to 255 in case of overflow
                img_rendered->imageData[img_rendered->widthStep * row + col * 3] = u_char(255);
                
            }
            //Green channel
            if (u_int(img->imageData[img->widthStep * row + col * 3 + 1])*avgG <= 255) {
                img_rendered->imageData[img_rendered->widthStep * row + col * 3 + 1]=img->imageData[img->widthStep * row + col * 3 + 1]*avgG;
            } else {
                img_rendered->imageData[img_rendered->widthStep * row + col * 3 + 1] = u_char(255);
            }
            //Red channel
            if (u_int(img->imageData[img->widthStep * row + col * 3 + 2])*avgR <= 255) {
                img_rendered->imageData[img_rendered->widthStep * row + col * 3 + 2]=img->imageData[img->widthStep * row + col * 3 + 2]*avgR;
            } else {
                img_rendered->imageData[img_rendered->widthStep * row + col * 3 + 2] = u_char(255);
            }
            
        }
    }

    //create a window to display the new image
    cvNamedWindow("gray world re-rendered image",CV_WINDOW_AUTOSIZE);
    //show the new image
    cvShowImage("gray world re-rendered image",img_rendered);
    //show the original image
    cvNamedWindow("original image",CV_WINDOW_AUTOSIZE);
    cvShowImage("original image", img);
    //Save the re-rendered image to file
    cvSaveImage(SAVEPATH, img_rendered);
    cout<<"press any key to close the image window"<<endl;
    cvWaitKey(0);//this commend is for displaying the image windows then hold
    //release images data and header
    cvReleaseImage(&img);
    cvReleaseImage(&img_rendered);
    
}
/*****************************
Below codes are for problem 2
*****************************/

//define global variables:
//img: the original loaded image matrix, 3 optional images to select
//img_Lab: convert img to this image with Lab color space
//img_segmented: filtered result of img_Lab
//spatialRad and colorRad stands for the filter parameter spatical window radius and color window radius

Mat img,img_Lab,img_LabFixed,img_segmented;
int spatialRad,colorRad;

//call back function for trackbar
//display current spatial/color radius
//do the meanshit filtering to the image, then show the result image
static void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
    << "colorRad=" << colorRad << "; "<<endl;
    //The maximun pyramid level is set to 1 according to problem 2 description
    pyrMeanShiftFiltering( img_Lab, img_segmented, spatialRad, colorRad, 1 );
    imshow( "filtering result", img_segmented );

}
//read image file then apply meanshift filter to it
int PyrMeanShift()
{
    //enter the input: 1 2 3. The index numbers represent 3 different images
    cout<<"please enter the index of images (1, 2 or 3) you want to use this filter: ";
    cin.ignore();//dealing with '\n'
    char index;
    index=cin.get();
    if(index=='1'){
        img=imread("/Users/liuhengyue/Desktop/574hw2/hw2/198023.jpg");
    }
    else if(index=='2'){
        img=imread("/Users/liuhengyue/Desktop/574hw2/hw2/46076.jpg");
    }
    else if(index=='3'){
        img=imread("/Users/liuhengyue/Desktop/574hw2/hw2/317080.jpg");
    }
    else{
        cout<<"input value invalid, please restart."<<endl;
        return -1;
    }
    //change the color space BGR to LAB, because of 8-bit image, the Lab values are shifted
    cvtColor(img, img_Lab, CV_BGR2Lab);
    /**** These codes are reserved for fixing the Lab color
    
    for(int k=0;k<img_Lab.rows;k++){
        //const uchar* inData=img_Lab.ptr<uchar>(k);
        char* outData=img_Lab.ptr<char>(k);
        for(int i=0;i<img_Lab.cols*img_Lab.channels();i=i+3){
            outData[i]=outData[i]*100/255;//L channel
            outData[i+1]=char(outData[i+1])-128;//a channel
            outData[i+2]=outData[i+2]-128;//b channel
            cout<<int(outData[i+2])<<endl;
        }
    }
    ***/
    //create a window to display the results
    namedWindow( "filtering result", WINDOW_AUTOSIZE );
    //create trackbars for spatical radius and color radius for pyramidMeanShift Filter
    createTrackbar( "spatialRad", "filtering result", &spatialRad, 80, meanShiftSegmentation );
    createTrackbar( "colorRad", "filtering result", &colorRad, 60, meanShiftSegmentation );
    meanShiftSegmentation(0, 0);//call back
    cout<<"press any key to exit program"<<endl;
    cvWaitKey(0);//for exit program
    return 0;
}
int main(int argc, char** argv)
{
    cout<<"type problem number 1 or 2: "<<endl;
    char index;
    index=cin.get();
    if(index=='1'){
        GrayWorldRender();
    }
    else if(index=='2'){
        PyrMeanShift();
    }
    else{
        cout<<"error input."<<endl;
    }
    return 0;
}