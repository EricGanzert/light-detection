#ifndef LIGHT_DETECTION_H
#define LIGHT_DETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>

typedef struct {
	std::string name;
	std::string path;
	double timestamp;
} File;

typedef struct {
	std::string name;
	std::string path;
	std::vector<File> files;
} Directory;

typedef struct {
	cv::KeyPoint targ;
	int count;
	double lastTimeSeen; //ms
} HistoryMember;

//the main function which gets the image filepaths
//and calls the image processing function
int flashDetect( int argc, char** argv );


//a helper function used to get directory levels in filepath strings
std::vector<std::string> split_string_by_delim(std::string string_to_split, const std::string& delim);


//fill the directory structure will all the filepaths to images located in the directory path
//from the directory arg's path member
bool getFiles( Directory& dir );


//orders the images retreived by timestamp (which is embedded in the image file's name)
void sortFiles( std::vector<File>& datFiles );


void mySwap( std::vector<File> &v, int index1, int index2 );


//returns true if the string contains one of the common image file extensions
bool isImg(std::string file_name);

//the big function that loops through every image comparing it to the previous
//and calls functions to manage the history of points of interest
int process_images( Directory& testImages );


//sets the requirements for blobs in the difference image to be detected 
void setFlashDetectParams(cv::SimpleBlobDetector::Params &params);


void manageHistory( std::vector<cv::KeyPoint> &keypoints, std::vector<HistoryMember> &history, double currentImgTimestamp, HistoryMember &favorite);


//returns the index of the member in history if its in there, else returns -1
int inHistory( cv::KeyPoint input, std::vector<HistoryMember> &history );


//returns true if the 2 points have less than tolerance distance in x, y, size dimensions
bool closeEnough(cv::KeyPoint kp_1, cv::KeyPoint kp_2, float tolerance );


//draws circles around keypoints on the display image which is useful in debugging
void highlightKeypoints( std::vector<cv::KeyPoint>& keypoints, cv::Mat display );


double vecAvg(std::vector<double>& vec);

#endif //LIGHT_DETECTION_H
