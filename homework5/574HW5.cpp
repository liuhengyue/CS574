//
//  574HW5.cpp
//  574hw5
//  Object classification program.
//  Methods: PCA-SIFT, k-means clustering, kNN.
//  Created by LiuHengyue on 11/19/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//

#include <iostream>
#include <vector>
#include <math.h>
#include <glob.h>
//opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;

//parameters for the program(change them here)
#define CATEGORIES 5
#define PCA_DIM 50
#define K_MEANS 120
#define N_NEARST_NEIGHBOR 8

/*************************************************************************
 * GetFiles
 * Function: Get all files' paths in particular folder.
 * Input: folder path.
 * Output: a vector containing all files' paths.
 *************************************************************************/

vector<const char*> GetFiles(const char* path)
{
    vector<const char*> files;
    glob_t glob_result;
    glob(path,GLOB_TILDE,NULL,&glob_result);
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        //cout << glob_result.gl_pathv[i] << endl;
        files.push_back(glob_result.gl_pathv[i]);
    }
    return files;
}

/*************************************************************************
 * PCA_SIFT
 * Function: Compute SIFT features of an image,then reduce feature dimension
 *           via PCA.
 * Input: Image path, PCA target dimension.
 * Output: PCA-SIFT feature vectors.
 *************************************************************************/

Mat PCA_SIFT(const char* path,int dim){
    Mat img = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
    //define SIFT variables
    Mat descriptors;
    vector<KeyPoint> keypoints;
    SIFT sift;
    
    /* SIFT feature extraction */
    
    sift.SIFT::operator()(img, noArray(), keypoints, descriptors);
    /* PCA */
    //n*20
    Mat reducedFeatures;
    PCA pca(descriptors,Mat(),0,dim);
    pca.project(descriptors, reducedFeatures);
    /* covar + eigen Note: this will work too.
    Mat covarMat,meanMat;
    calcCovarMatrix(descriptors,covarMat, meanMat,CV_COVAR_NORMAL|CV_COVAR_SCALE| CV_COVAR_ROWS);

    Mat newMat,eigenValues,eigenVec;
    eigen(covarMat, eigenValues, eigenVec);
    Mat basis,normalBasis,cvtVec;
    eigenVec.convertTo(cvtVec, CV_32F);
    //20*128
    basis=cvtVec.rowRange(0, 20);
    //20*n
    normalize(basis.t(), normalBasis);
    newMat=descriptors*normalBasis;
    return newMat;
     */
    return reducedFeatures;
}

/*************************************************************************
 * CodewordsHis
 * Function: Compute codewords for feature vectors of one image, and compute
 *           the histogram of the codewords via codeBook.
 * Input: Feature vectors n*dim, codeBook k*dim.
 * Output: Codewords histogram (normalized) 1*k.
 *************************************************************************/

Mat CodewordsHis(Mat features, Mat codeBook){
    //histogram of codewords
    Mat his=Mat::zeros(1,codeBook.rows,features.type());
    //for each feature 20-d vector
    for (int i=0; i<features.rows; i++) {
        //store the nearest codeword
        double dis=INT_MAX;
        int code=0;
        //compute the distance to each codeword
        for (int row=0; row<codeBook.rows; row++) {
            double curDis=norm(features.row(i), codeBook.row(row));
            //find the nearest neighbor and assign label
            if (curDis<dis) {
                dis=curDis;
                code=row;
            }

        }
        his.at<float>(0,code)+=1;
    }
    //normalized histogram
    for (int k=0; k<his.cols; k++) {
        his.at<float>(0,k)/=features.rows;
    }
    return his;
}

/*************************************************************************
 * WeightedVote
 * Function: Compute k-nearest neighbors' vote weighted by its distance to
 *           cluster means. Find the most votes and assign corresponding
 *           category label.
 * Input:  K-nearest-neighbors' category label and distance to cluster means.
 * Output: Prediction of category label (one image per row).
 *************************************************************************/

Mat WeightedVote(Mat neighbors,Mat dist){
    Mat prediction;
    for (int row=0; row<neighbors.rows; row++) {
        float votes[CATEGORIES]={0,0,0,0,0};
        for (int col=0; col<neighbors.cols; col++) {
            votes[int(neighbors.at<float>(row,col))]+=1.0/dist.at<float>(row, col);
        }
        float mostVote=votes[0];
        float vote=0;
        for (int i=1; i<CATEGORIES; i++) {
            if (votes[i]>mostVote) {
                mostVote=votes[i];
                vote=i;
            }
        }
        prediction.push_back(vote);
    }
    return prediction;
}

void testCase(){
    Mat reducedFeatures=PCA_SIFT("/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/car_side/train/image_0001.jpg", PCA_DIM);
    cout<<reducedFeatures.rowRange(0,5)<<endl;
}
int main(int argc, const char * argv[]) {
    cout<<"Parameters:"<<endl<<"PCA dimension: "<<PCA_DIM<<endl<<"K-Cluster No.: "<<K_MEANS<<endl<<"kNN neighbors: "<<N_NEARST_NEIGHBOR<<endl;
    /* training part */
    //define paths
    const char* CAR="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/car_side/train/*";
    const char* BUTTERFLY="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/butterfly/train/*";
    const char* FACE="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/faces/train/*";
    const char* WATCH="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/watch/train/*";
    const char* LILLY="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/water_lilly/train/*";
    const char* trainPaths[]={CAR,BUTTERFLY,FACE,WATCH,LILLY};
    
    const char* CARTEST="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/car_side/test/*";
    const char* BUTTERFLYTEST="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/butterfly/test/*";
    const char* FACETEST="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/faces/test/*";
    const char* WATCHTEST="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/watch/test/*";
    const char* LILLYTEST="/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/water_lilly/test/*";
    const char* testPaths[]={CARTEST,BUTTERFLYTEST,FACETEST,WATCHTEST,LILLYTEST};

    
    //PCA_SIFT("/Users/liuhengyue/Google Drive/CS574/hw5/hw5/images/car_side/train/image_0001.jpg",20);
    //Read images
    //100 training images
    //combined matrix for k-means
    Mat featureData;
    //store each image feature vectors
    vector<Mat> imgBag;
    cout<<"Reading training images..."<<endl;
    for (int i=0; i<CATEGORIES; i++) {
        //get all file paths for each category
        vector<const char*> files=GetFiles(trainPaths[i]);
        //20 training images for each category
        for (int j=0; j<files.size(); j++) {
            //reduce to 20-dimension feature vector
            Mat reducedFeatures=PCA_SIFT(files[j], PCA_DIM);
            featureData.push_back(reducedFeatures);
            imgBag.push_back(reducedFeatures);
        }
    }
    
    /* K-means */
    cout<<"conducting k-means..."<<endl;
    //means is 100*20
    Mat labels,codeBook;
    kmeans(featureData, K_MEANS, labels,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 50, 1.0),
           3, KMEANS_PP_CENTERS, codeBook);
    /* compute histogram of each training image */
    Mat trainHis;//size should be 100*k, one image per row
    Mat trainCategories;//size should be 100*1, train image category label 0,1,2,3,4 for 5 categories
    for (int m=0; m<imgBag.size(); m++) {
        trainHis.push_back(CodewordsHis(imgBag[m], codeBook));
        //assign category label
        trainCategories.push_back(m/20);
        }
    /* testing part */
    //store each image codewords histogram
    Mat testHis;
    cout<<"Reading testing images and generating codewords...";
    for (int i=0; i<CATEGORIES; i++) {
        //get all file paths for each category
        vector<const char*> files=GetFiles(trainPaths[i]);
        //20 training images for each category
        for (int j=0; j<files.size(); j++) {
            //reduce to 20-dimension feature vector
            Mat reducedFeatures=PCA_SIFT(files[j], PCA_DIM);
            testHis.push_back(CodewordsHis(reducedFeatures,codeBook));
        }
    }
    //n-nearest neighbor classifier
    cout<<"Done!"<<endl<<"Conducting categorizing via k-nearest neighbor classifier...";
    Mat prediction,neighbors,dist;
    KNearest kNN;
    //train kNN
    kNN.train(trainHis, trainCategories);
    //find k-nearest neighbors
    kNN.find_nearest(testHis,N_NEARST_NEIGHBOR,0,0,&neighbors,&dist);
    //using weighted voting
    prediction=WeightedVote(neighbors,dist);
    cout<<"Done!"<<endl;
    //count the number of correct labels
    int count=0;
    for (int i=0; i<prediction.rows; i++) {
        if(prediction.at<float>(i,0)==i/(prediction.rows/CATEGORIES)){
            count++;
        }
    }
    //cout<<prediction<<endl;
    cout<<"Correct classified category: "<<100*float(count)/prediction.rows<<"%"<<endl;
    //testCase();
    cout<<prediction.t()<<endl;
    //cout<<codeBook.rowRange(0, 5)<<endl;
    //cout<<trainHis.rowRange(0,5)<<endl;
    //cout<<neighbors.rowRange(0, 5)<<endl;
    return 0;
}
