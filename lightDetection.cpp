#include <lightDetection.h>

using namespace cv;
using namespace std;

const double PI = 3.14159265359;
RNG rng(12345);
double frameRate = 30; //Hz

string img_directory_id = "img_captures/";

//*check out the comments in the header file*

int flashDetect( int argc, char** argv )
{
	if (argc < 2)
	{
		//cout << "usage: ./light-detection ../directory/\n";
		return 0;
	}
	
	string dir_path = argv[1];
	size_t index = dir_path.find(img_directory_id);
	if (index == string::npos)
	{
		//cout << "fail: " << img_directory_id << " not found in directory path\n";
		return 0;
	}
	
	dir_path = dir_path.substr(0, (index + img_directory_id.length()));
	//cout << "input given is " << dir_path << "\n";

	Directory dir;
	vector<string> path_levels = split_string_by_delim(dir_path, "/");
	dir.path = dir_path;
	dir.name = path_levels.back();
	//cout << "path " << dir.path << ", name: " << dir.name << "\n";
	
	if (!getFiles( dir ))
	{
		return 0;
	}

	int result = process_images(dir);
	cout << result;
	return result;	
}

vector<string> split_string_by_delim(string string_to_split, const string& delim)
{
	vector<string> result;
	size_t pos = 0;
	string token;
	
	while ((pos = string_to_split.find(delim)) != string::npos)
	{
		token = string_to_split.substr(0, pos);
		if (!token.empty())
		{
			//cout << token << endl;
			result.push_back(token);
		}
		string_to_split.erase(0, pos + delim.length());
	}
	
	if (!string_to_split.empty())
	{
		//cout << string_to_split << endl;
		result.push_back(string_to_split);
	}
	return result;
}

bool getFiles( Directory& dir )
{
	DIR *dir_ptr;
	struct dirent *entry;
	//cout << "getting files\n";
	if ( dir_ptr = opendir( dir.path.c_str() ) )
	{
		while( entry = readdir(dir_ptr) )
		{	
			string file_name = entry->d_name;
			if ( isImg(file_name) )
			{
				File temp;
				temp.name = file_name;
				temp.path = dir.path + temp.name;
				//cout << "file name: " << temp.name << "\n";
				//cin.ignore();
				
				vector<string> parts_in_name = split_string_by_delim(temp.name, "_");
				
				temp.timestamp = stof(parts_in_name[parts_in_name.size()-1])*33.33333; //ms
				//cout << "timestamp: " << temp.timestamp << "\n";
				//cin.ignore();				
				dir.files.push_back(temp);
			}
		}
	}
	//cout << "done getting files\n";
	if ( dir.files.empty() )
		return false;
	else
	{
		sortFiles( dir.files );
		//cout << "sorted images by timestamp\n";
		return true;
	}
}

void sortFiles( vector<File>& datFiles )
{
	int numFiles = datFiles.size();
	for (int i=0; i<numFiles; i++)
	{
		double timeStampBase = datFiles[i].timestamp;
		
		int indexOfSmallest = i;
		for (int j=i; j<numFiles; j++)
		{
			double timeStampComp = datFiles[j].timestamp;
			if ( timeStampBase > timeStampComp )
			{
				timeStampBase = timeStampComp;
				indexOfSmallest = j;
			}
		}
		mySwap(datFiles, i, indexOfSmallest);
	}
}

void mySwap( vector<File>& v, int index1, int index2 )
{
	File temp = v[index1];
	v[index1] = v[index2];
	v[index2] = temp;
}

bool isImg(string file_name)
{
	return ( file_name.find(".jpeg") != string::npos ||
			 file_name.find(".JPEG") != string::npos ||
			 file_name.find(".jpg") != string::npos  ||
			 file_name.find(".JPG") != string::npos  ||
			 file_name.find(".png") != string::npos	 ||
			 file_name.find(".PNG") != string::npos);
}

int process_images( Directory& testImages )
{
	Mat currentImg;
	Mat prevImg;
	Mat diffImg;
	Mat display;
	
	SimpleBlobDetector::Params params;
	setFlashDetectParams(params);
	
	#if CV_MAJOR_VERSION < 3
		SimpleBlobDetector detector(params);
	
	#else
		//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	#endif
	
	vector<KeyPoint> keypoints; //blobs returned for each image
	vector<HistoryMember> history;	//here we store the confirmed targets and their count
	HistoryMember favorite;
	favorite.lastTimeSeen = 0;
	favorite.count = 0;
	favorite.targ.pt.x = 0;
	favorite.targ.pt.y = 0;
	favorite.targ.size = 0;
	
	vector<double> favFreq;
	double medianFreq = 0;
	
	int imgCount = 1; int flashSeenCount = 0;
	int missesInARow = 0; int maxMissesInARow = 30;
	int numImages = testImages.files.size();
	
	for (int imgIndex=1; imgIndex<numImages; imgIndex++)
	{
		currentImg = imread(testImages.files[imgIndex].path);
		display = currentImg.clone();
		cvtColor( currentImg, currentImg, CV_BGR2GRAY );
		prevImg = imread(testImages.files[imgIndex-1].path);
		cvtColor( prevImg, prevImg, CV_BGR2GRAY );
		imgCount++;
		
		absdiff( currentImg, prevImg, diffImg );
			
		threshold( diffImg, diffImg, 30, 255, THRESH_BINARY ); //why?
		
		//diffImg = Scalar::all(255) - diffImg;
		
		//get rid of random noise
		medianBlur(diffImg, diffImg, 3);			
		
		keypoints.clear();
	
		int erosion_size = 4;
		int erosion_type = MORPH_RECT;
		
		Mat element = getStructuringElement( erosion_type, Size(2*erosion_size + 1, 2*erosion_size + 1 ), Point( erosion_size, erosion_size ) );
		
		erode(diffImg, diffImg, element);
		//imshow( "input to blob detector", diffImg );
		//waitKey(0);
		
		#if CV_MAJOR_VERSION < 3
			detector.detect( diffImg, keypoints );
		#else
			detector->detect( diffImg, keypoints );
		#endif
		
		//look through keypoints, adding them to our record
		HistoryMember prevFav = favorite;
		manageHistory(keypoints, history, testImages.files[imgIndex].timestamp, favorite);
		
		//cout << "favorite is at(cols, rows): (" << favorite.targ.pt.x << ", " <<  favorite.targ.pt.y << ")\n";
		
		if (favorite.targ.pt.x != prevFav.targ.pt.x ||
			favorite.targ.pt.y != prevFav.targ.pt.y)
		{
			//cout << "New Favorite location!\n";
			favFreq.clear();
		}
		
		if (favorite.targ.pt.x == prevFav.targ.pt.x &&
			favorite.targ.pt.y == prevFav.targ.pt.y &&
			favorite.lastTimeSeen != prevFav.lastTimeSeen)
		{
			double favPeriod = (testImages.files[imgIndex].timestamp - prevFav.lastTimeSeen) + (1.0/frameRate)*1000;
			//cout << "time since the favorite was last seen: " << favPeriod << "ms\n";
			favFreq.push_back(1.0/(favPeriod/1000.0));
		}

		highlightKeypoints( keypoints, display );
		//put a green circle on the favorite target
		circle( display, favorite.targ.pt, favorite.targ.size*1.5, Scalar(0,255,0), 2 );	
		
		//*****************UNCOMMENT THE FOLLOWING 2 LINES TO SEE A DEGUB DEMONSTRATION****************
		//MUST HAVE OPENCV INSTALLED. WONT DISPLAY OVER A CONNECTION LIKE SSH(AND PROGRAM WOULD STOP).
		imshow( "display", display );
		waitKey(1);	
	}
	if(favFreq.size() > 2)
	{
		sort(favFreq.begin(), favFreq.end());
		int size = favFreq.size();
		medianFreq = favFreq[floor(size/2.0)];
		double avgFreq = vecAvg(favFreq);
		//cout << "Frequency Estimate(median): " << medianFreq << "Hz\n";
		//cout << "Frequency Estimate(average): " << avgFreq << "Hz\n";
	}
	if (favorite.count > 10 && medianFreq > 2)
	{
		cout << "location(row,col): (" << (int)favorite.targ.pt.y << "," << (int)favorite.targ.pt.x << ")\n";
		return 1;
	}
	else
	{
		return 0;
	}
}

void setFlashDetectParams(SimpleBlobDetector::Params &params)
{
	params.minThreshold = 0;
	params.maxThreshold = 256;
	params.minRepeatability = 1;
	params.filterByCircularity = 1;
	params.minCircularity = 0.38;
	params.maxCircularity = 1;
	params.filterByArea = 1;
	params.minArea = 50;
	params.maxArea = 75000;
	params.filterByColor = 0;
	params.filterByConvexity = 0;
	params.filterByInertia = 0;
	params.minDistBetweenBlobs = 10;
}

void manageHistory( vector<KeyPoint> &keypoints, vector<HistoryMember> &history, double currentImgTimestamp, HistoryMember &favorite )
{
	//cout << "----Managing History----\n" << "current Timestamp is " << currentImgTimestamp << "\n";
	int numKeypoints = keypoints.size();
	int histSize = history.size();
	
	//cout << "num keypoints : " << numKeypoints << "\n";
	//cout << "history size : " << history.size() << "\n";
	for (int i=0; i<numKeypoints; i++)
	{
		//cout << "keypoint area is " << pow((keypoints[i].size/2),2)*3.14159 << "\n";
		//cout << "at coordinates(x,y): (" << keypoints[i].pt.x << ", " << keypoints[i].pt.y << ")\n";
		
		
		//if the keypoint location is in the history already, increment its counter, time since last seen = 0.
		//if it is a new keypoint location, add to history, intitialize counter to 1
		int histIndex = inHistory(keypoints[i], history);
		if (histIndex >= 0)
		{
			history[histIndex].count++;
			history[histIndex].lastTimeSeen = currentImgTimestamp;
		}
		else
		{
			HistoryMember new_member;
			new_member.targ = keypoints[i];
			new_member.count = 1;
			new_member.lastTimeSeen = currentImgTimestamp;
			history.push_back(new_member);
		}
	}
	//waitKey(0);

	if (history.size() > 0)
	{
		//go through history to find the one with the higest count
		histSize = history.size();
		int indexOfHighestCount = 0;
		for (int j=0;j<history.size(); j++)
		{	
			if (history[j].count > history[indexOfHighestCount].count)
			{
				indexOfHighestCount = j;
				//cout << "new highest count: " <<  history[indexOfHighestCount].count << endl;
			}
		}
		HistoryMember fav = history[indexOfHighestCount];
		favorite.targ.pt.x = fav.targ.pt.x;
		favorite.targ.pt.y = fav.targ.pt.y;
		favorite.targ.size = fav.targ.size;
		favorite.targ.size = fav.targ.size;
		favorite.lastTimeSeen = fav.lastTimeSeen;
		favorite.count = fav.count;
		//cout << "time since the favorite was last seen: " << (currentImgTimestamp - favorite.lastTimeSeen) + (1.0/frameRate)*1000 << "ms\n";
		//~ cout << "In manageHistory, fav details are:\n" << "(x, y): (" << favorite.targ.pt.x << ", " << favorite.targ.pt.y << ")\n"
			 //~ << "lastTimeSeen: " << favorite.lastTimeSeen << "\nCount: " << favorite.count << "\n";
	}
}

int inHistory( KeyPoint input, vector<HistoryMember> &history )
{
	float tolerance = 20;
	int histSize = history.size();
	for (int i=0; i<histSize; i++)
	{
		if ( closeEnough(input, history[i].targ, tolerance) )
			return i;
	}
	return -1;
}

bool closeEnough(KeyPoint kp_1, KeyPoint kp_2, float tolerance )
{
	KeyPoint diff;
	diff.pt.x = fabs(kp_1.pt.x - kp_2.pt.x);
	diff.pt.y = fabs(kp_1.pt.y - kp_2.pt.y);
	
	float distance = sqrt( pow( diff.pt.x, 2 ) + pow( diff.pt.y, 2 ));
	
	return (distance < abs(tolerance));
}

void highlightKeypoints( vector<KeyPoint>& keypoints, Mat display )
{
	for (int i=0; i<keypoints.size(); i++ )
	{
		circle( display, keypoints[i].pt, keypoints[i].size, Scalar(0,0,255), 2 );
	}
	//imshow("keypoints", display);
}

double vecAvg(vector<double>& vec)
{
	double total=0;
	int size = vec.size();
	for (int i=0;i<size;i++)
	{
		total += vec[i];
	}
	return total/size;
}
