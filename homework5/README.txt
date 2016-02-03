//
//  574HW5.cpp
//  CSCI 574 Assignment #5
//
//  Object classification program.
//  Methods: PCA-SIFT, k-means clustering, kNN.
//
//  There are 4 parameters for this implementation, if one want to change
//  parameters, just change the pre-defined values for each parameter.
//
//  For successfully run the program, one may need to change the file paths
//  in the main() function.
//
//  Created by LiuHengyue on 11/19/15.
//  Copyright © 2015 LiuHengyue. All rights reserved.
//

//parameters for the program(change them here)
#define CATEGORIES 5
#define PCA_DIM 50
#define K_MEANS 120
#define N_NEARST_NEIGHBOR 8

//Below are functions in this implementation

/*************************************************************************
 * GetFiles
 * Function: Get all files' paths in particular folder.
 * Input: folder path.
 * Output: a vector containing all files' paths.
 *************************************************************************/

vector<const char*> GetFiles(const char* path);

/*************************************************************************
 * PCA_SIFT
 * Function: Compute SIFT features of an image,then reduce feature dimension
 *           via PCA.
 * Input: Image path, PCA target dimension.
 * Output: PCA-SIFT feature vectors.
 *************************************************************************/

Mat PCA_SIFT(const char* path,int dim);

/*************************************************************************
 * CodewordsHis
 * Function: Compute codewords for feature vectors of one image, and compute
 *           the histogram of the codewords via codeBook.
 * Input: Feature vectors n*dim, codeBook k*dim.
 * Output: Codewords histogram (normalized) 1*k.
 *************************************************************************/

Mat CodewordsHis(Mat features, Mat codeBook);

/*************************************************************************
 * WeightedVote
 * Function: Compute k-nearest neighbors' vote weighted by its distance to
 *           cluster means. Find the most votes and assign corresponding
 *           category label.
 * Input:  K-nearest-neighbors' category label and distance to cluster means.
 * Output: Prediction of category label (one image per row).
 *************************************************************************/

Mat WeightedVote(Mat neighbors,Mat dist);
