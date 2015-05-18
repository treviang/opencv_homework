#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//show help information
int print_help()
{
	std::cout << "*******************************************************" << std::endl;
	std::cout << "Stereo Calibration Camera:" << std::endl;
	std::cout << "   --help    <show_this_help>" << std::endl;
	std::cout << "   --data    <path_to_left_right_dataset_directory>" << std::endl;
	std::cout << "             directory must contain two distinct subdirectories for left and right images" << std::endl;
	std::cout << "*******************************************************" << std::endl;
	return 0;
}

//fill a string vector with all files located on a specified path
void getFiles(std::string path,std::vector<std::string> &files) //
{
	DIR *dirp = NULL;
	struct dirent *dp;
	//open directory specified in path
	dirp = opendir(path.c_str());

	while (dirp){ //for every file in dirp
	if ((dp = readdir(dirp)) != NULL) {
		if(strcmp(dp->d_name, ".")!=0 && strcmp(dp->d_name, "..")!=0)
			files.push_back(dp->d_name); //save filename in files
		}
	else{
		closedir(dirp);
		dirp = NULL;
		}
	}
}

int main(int argc, char** argv){
	//default path to left and right calibration images
	std::string dataset_path="../dataset/calibration";

	//manage command line options
	if(argc==2 && (std::string(argv[1])=="-h" || std::string(argv[1])=="--help")) //help command
		return print_help();
	else if((argc==2 && (std::string(argv[1])!="-h" && std::string(argv[1])!="--help")) || argc>3){ //invalid arguments
		std::cerr << "Usage: --help MORE USAGE INFORMATION" << std::endl;
		std::cerr << "       --data PATH TO DATASET IMAGES TO CALIBRATE" << std::endl;
		return 0;
	}
	else if(argc==3){ //dataset option
		for (int i = 1; i < argc; ++i) {
        if ((std::string(argv[i]) == "--data") || std::string(argv[i]) == "-d") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                dataset_path = argv[i+1]; //update dataset_path
            	}
            else { //bad argument
                std::cerr << "Invalid argument! type -h or --help for more information" << std::endl;
                return 1;
            	}
        	}
    	}
	}
	//parameters

	//timers
	clock_t init, timeLeft, timeRight, timeStereo,timeTot;

	//number of chessboard corners
	int numCornersHor=8;	//horizontal
    int numCornersVer=6;	//vertical

    int numSquares = numCornersHor * numCornersVer;  //number of squares in the cessboard

    cv::Size board_sz = cv::Size(numCornersHor, numCornersVer); //pattern size
    cv::Size winSize = cv::Size(5,5); //search window used in cornerSubPix
    cv::Size zeroZone = cv::Size(-1,-1); //size of the dead region if (-1,-1) there is not a size

    //vectors of chessboard corners for left calibration, right calibration and stereo calibration
    std::vector<std::vector<cv::Point2f> > leftImagePoints;
    std::vector<std::vector<cv::Point2f> > rightImagePoints;
    std::vector<std::vector<cv::Point2f> > stereoLeftImagePoints;
    std::vector<std::vector<cv::Point2f> > stereoRightImagePoints;

    //vectors of objectPoints for left calibration, right calibration and stereo calibration
    std::vector<std::vector<cv::Point3f> >  leftObjectPoints;
    std::vector<std::vector<cv::Point3f> >  rightObjectPoints;
    std::vector<std::vector<cv::Point3f> >  objectPoints;

   	cv::Mat leftDistCoeffs,rightDistCoeffs; //left and right distortion coefficient
    std::vector<cv::Mat> leftRvecs,leftTvecs,rightRvecs,rightTvecs; //left and right rotation and traslation vectors
    cv::Mat img,img_gray;//Mat to load images and fin chessboard corners

    //object point for a single image
    std::vector<cv::Point3f> obj;

    //fill obj vector
    for(int j=0;j<numSquares;j++){
        	obj.push_back(cv::Point3f(j/numCornersHor, j%numCornersHor, 0.0f));
    	}

    if(argc>2){ //default path has changed
    	std::vector<std::string> files;
		std::string leftPath = dataset_path;
		std::string rightPath = dataset_path;
		std::size_t foundLeft; //found left directory
		std::size_t foundRight; //found right directory

		getFiles(dataset_path,files); //get files from path

		//complete left and right path with its folder
		for(int i=0;i<files.size();i++){
			foundLeft=files.at(i).find("left");
			foundRight=files.at(i).find("right");
			if(foundLeft!=std::string::npos)
				leftPath=leftPath+"/"+files.at(i);
			if(foundRight!=std::string::npos)
				rightPath=rightPath+"/"+files.at(i);
		}

		std::vector<std::string> leftFiles;
		std::vector<std::string> rightFiles;

		getFiles(leftPath,leftFiles); //get left files
		getFiles(rightPath,rightFiles); //get right files
		//combine directory path and filenames to obtain complete paths
		for(int i=0;i<leftFiles.size();++i){
			leftFiles.at(i)=leftPath+"/"+leftFiles.at(i);
			rightFiles.at(i)=rightPath+"/"+rightFiles.at(i);
		}
		//store new dataset in images.xml
		cv::FileStorage fsC("config/images.xml", cv::FileStorage::WRITE);
		fsC << "images_left" << leftFiles;
		fsC << "images_right" << rightFiles;
		fsC.release();
	}

	//load images.xml
    cv::FileStorage fs("config/images.xml", cv::FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
      std::cout << "Could not open the configuration file: images.xml" << std::endl;
      return -1;
	}

	std::cout << "Welcome to Stereo Calibration Camera" << std::endl;
	init=clock(); //start program initial timer

	/// calibrate left camera
	cv::FileNode imL = fs["images_left"]; //find node images_left in images.xml
	cv::FileNodeIterator currentImageLeft = imL.begin(); //iterator to current image
    cv::FileNodeIterator lastImageLeft = imL.end(); //iterator to last image

    std::cout <<"Start left camera calibration" << std::endl;
    std::vector<bool> left_found; //corners found
    std::vector<cv::Point2f> left_corners; //this will be filled by the detected left corners of actual image
    int numImg=0; //number of image processed
    int error=0; //number of image that failed the process

    while(currentImageLeft != lastImageLeft){

		std::cout <<"Processing " << ((std::string)*currentImageLeft).substr(((std::string)*currentImageLeft).size()-17, 50);
		img = cv::imread((std::string)*currentImageLeft); //read actual image

		left_found.push_back(findChessboardCorners(img, board_sz, left_corners,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK));

		if (left_found.at(numImg))  // If done with success,
		{
			std::cout <<" OK"<< std::endl;
      		// improve the found corners' coordinate accuracy for chessboard
			cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        	cornerSubPix(img_gray, left_corners, winSize, zeroZone, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ));
        	leftImagePoints.push_back(left_corners);
        	leftObjectPoints.push_back(obj);
        }
        else{
        	std::cout <<" ERR"<< std::endl;
        	error++;
        }
    	stereoLeftImagePoints.push_back(left_corners);
    	objectPoints.push_back(obj);
    	++currentImageLeft;
    	++numImg;
    }
    cv::Mat leftCameraMatrix = cv::Mat(3, 3,  CV_32FC1); //3x3 floating-point camera matrix
    leftDistCoeffs = cv::Mat::zeros(8, 1, CV_64F); //8x1 vector of distortion coefficients

    double left_error = calibrateCamera(leftObjectPoints, leftImagePoints, img.size(), leftCameraMatrix,
                                 leftDistCoeffs, leftRvecs, leftTvecs);

    timeLeft=clock()-init; //time of left calibration

	std::cout<<"Left calibration done in " <<(double)timeLeft / ((double)CLOCKS_PER_SEC) <<" seconds." <<std::endl;
	std::cout <<error <<" images failed to find chessboard corners." <<std::endl;
	std::cout<<"Calibration Error: "<<left_error<<std::endl;

    /// calibrate right camera
    cv::FileNode imR = fs["images_right"]; //find node images_right in images.xml
	cv::FileNodeIterator currentImageRight = imR.begin(); //iterator to current image
    cv::FileNodeIterator lastImageRight = imR.end(); //iterator to last image

    std::cout <<"Start right camera calibration" << std::endl;
    std::vector<bool> right_found; //corners found
    std::vector<cv::Point2f> right_corners;	//this will be filled by the detected right corners of actual image
    numImg=0; //number of image processed
    error=0; //number of image that failed the process
    while(currentImageRight != lastImageRight){

		std::cout <<"Processing " << ((std::string)*currentImageRight).substr(((std::string)*currentImageRight).size()-18, 50);
		img = cv::imread((std::string)*currentImageRight); //read actual image

		right_found.push_back(findChessboardCorners(img, board_sz, right_corners,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK));

		if (right_found.at(numImg))  // If done with success,
		{
			std::cout <<" OK"<< std::endl;
      		// improve the found corners' coordinate accuracy for chessboard
			cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        	cornerSubPix(img_gray, right_corners, winSize, zeroZone, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ));
        	rightImagePoints.push_back(right_corners);
        	rightObjectPoints.push_back(obj);
        }
        else{
        	std::cout <<" ERR"<< std::endl;
        	error++;
        }

    	stereoRightImagePoints.push_back(right_corners);
    	++currentImageRight;
    	++numImg;
    }
    fs.release();

    cv::Mat rightCameraMatrix = cv::Mat(3, 3,  CV_32FC1); //3x3 floating-point camera matrix
    rightDistCoeffs = cv::Mat::zeros(8, 1, CV_64F); //8x1 vector of distortion coefficients

    double right_error = calibrateCamera(rightObjectPoints, rightImagePoints, img.size(), rightCameraMatrix,
                                 rightDistCoeffs, rightRvecs, rightTvecs);

    timeRight=clock()-timeLeft; //time of right calibration

	std::cout<<"Right calibration done in " <<(double)timeRight / ((double)CLOCKS_PER_SEC) <<" seconds." <<std::endl;
	std::cout <<error <<" images failed to find chessboard corners." <<std::endl;
	std::cout<<"Calibration Error: "<<right_error<<std::endl;

    /// calibrate stereo

	// prune left corners and right corners so that they contain only stereo-couples in which the pattern has been found both on left and right images.
	for(int i=0;i<std::min(left_found.size(), right_found.size());++i){
		if(!(left_found[i] && right_found[i])){
			stereoLeftImagePoints.erase(stereoLeftImagePoints.begin() + i);
			stereoRightImagePoints.erase(stereoRightImagePoints.begin() + i);
			objectPoints.erase(objectPoints.begin() + i);
		}
	}

	cv::Mat R;	// Rotation matrix between the 1st and the 2nd camera coordinate systems
	cv::Mat T;	// Translation vector between the coordinate systems of the cameras
	cv::Mat E;	// Essential matrix
	cv::Mat F;	// Fundamental matrix

	double stereo_error = cv::stereoCalibrate(
		objectPoints, stereoLeftImagePoints, stereoRightImagePoints,
		leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs,
		img.size(),
		R, T, E, F,
		cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 1e-6), cv::CALIB_USE_INTRINSIC_GUESS);

	//print stereo calibration time
	timeStereo=clock()-init; //total stereo time
	std::cout<<"Stereo calibration done in " <<(double)timeStereo / ((double)CLOCKS_PER_SEC) <<" seconds." <<std::endl;
	std::cout<<"Stereo Calibration Error: "<<stereo_error<<std::endl;

	/// create rectification maps
	cv::Mat R1;		// 3x3 rectification transform (rotation matrix) for the first camera
	cv::Mat R2;		// 3x3 rectification transform (rotation matrix) for the second camera
	cv::Mat P1;		// 3x4 projection matrix in the new (rectified) coordinate systems for the first camera
	cv::Mat P2;		// 3x4 projection matrix in the new (rectified) coordinate systems for the second camera
	cv::Mat Q;		// 4x4 disparity-to-depth mapping matrix

	stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, img.size(), R, T, R1, R2, P1, P2, Q);

	///save results
	std::cout << "Saving configuration..." << std::endl;

	cv::FileStorage fsW("config/outputCalibration.xml", cv::FileStorage::WRITE);

	//save parameters to reconstruct 3D image
	fsW << "leftCameraMatrix" << leftCameraMatrix;
	fsW << "rightCameraMatrix" << rightCameraMatrix;
	fsW << "leftDistCoeffs" << leftDistCoeffs;
	fsW << "rightDistCoeffs" << rightDistCoeffs;
	fsW << "R" << R;
    fsW << "T" << T;
    fsW << "E" << E;
    fsW << "F" << F;
	fsW << "R1" << R1;
	fsW << "R2" << R2;
	fsW << "P1" << P1;
	fsW << "P2" << P2;
	fsW << "Q" << Q;
	fsW.release();
	timeTot=clock()-init; //total time used
	std::cout << "Results saved." << std::endl;
	std::cout << "Total time: " <<timeTot <<std::endl;
	return 0;
}
