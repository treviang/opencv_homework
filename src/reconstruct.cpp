#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
	/// load calibration data
	cv::Mat leftCameraMatrix; //3x3 floating-point left camera matrix
	cv::Mat rightCameraMatrix; //3x3 floating-point right camera matrix
	cv::Mat leftDistCoeffs; //8x1 vector of left distortion coefficients 
	cv::Mat rightDistCoeffs; //8x1 vector of right distortion coefficients 
	cv::Mat R1;			// 3x3 rectification transform (rotation matrix) for the first camera
	cv::Mat R2;			// 3x3 rectification transform (rotation matrix) for the second camera
	cv::Mat P1;			// 3x4 projection matrix in the new (rectified) coordinate systems for the first camera
	cv::Mat P2;			// 3x4 projection matrix in the new (rectified) coordinate systems for the second camera
	cv::Mat Q;			// 4x4 disparity-to-depth mapping matrix

	//load all calibration parameters from outputCalibration.xml
	cv::FileStorage fsR("config/outputCalibration.xml", cv::FileStorage::READ);
	fsR["leftCameraMatrix"] >> leftCameraMatrix;                                      
    fsR["rightCameraMatrix"] >> rightCameraMatrix;
    fsR["leftDistCoeffs"] >> leftDistCoeffs;
    fsR["rightDistCoeffs"] >> rightDistCoeffs;
    fsR["R1"] >> R1;
    fsR["R2"] >> R2;
    fsR["P1"] >> P1;
    fsR["P2"] >> P2;
    fsR["Q"] >> Q;

    //load test images
	std::string left_filename, right_filename;
	left_filename="../../dataset/test/left.jpg";
	right_filename="../../dataset/test/right.jpg";
	cv::Mat left_image = cv::imread(left_filename);
	cv::Mat right_image = cv::imread(right_filename);

	cv::Size image_size = left_image.size(); //size of test images

	//left and right undistort and rectification maps
	cv::Mat left_map1;
	cv::Mat left_map2;
	cv::Mat right_map1;
	cv::Mat right_map2;

	//timers
	clock_t init, timeComplete;
	init=clock(); //start timer

	/// create rectification maps
	cv::initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1, image_size, CV_32FC1, left_map1, left_map2);
	cv::initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2, image_size, CV_32FC1, right_map1, right_map2);
	
	//remap images
	cv::Mat left_image_remap;
	cv::Mat right_image_remap;
	
	/// use the maps to rectificate images
	cv::remap(left_image, left_image_remap, left_map1, left_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
	cv::remap(right_image, right_image_remap, right_map1, right_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	//show undistorted and rectified test images;
	/*cv::namedWindow("Original left image");
	cv::imshow("Original left image",left_image);
	cv::namedWindow("Original right image");
	cv::imshow("Original right image",right_image);
	cv::namedWindow("Rectified left image");
	cv::imshow("Rectified left image",left_image_remap);
	cv::namedWindow("Rectified right image");
	cv::imshow("Rectified right image",right_image_remap);
	cv::waitKey(0);
	return 0;*/
		
	/// compute the disparity
	cv::StereoSGBM sgbm;
	sgbm.preFilterCap = 100;
	sgbm.SADWindowSize = 5;
	sgbm.P1 = 8 * left_image_remap.channels() * sgbm.SADWindowSize * sgbm.SADWindowSize;
	sgbm.P2 = 32 * left_image_remap.channels() * sgbm.SADWindowSize * sgbm.SADWindowSize;
	sgbm.minDisparity = 40;
	sgbm.numberOfDisparities = 256;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 200;
	sgbm.speckleRange = 2;
	sgbm.disp12MaxDiff = 0;
		
	cv::Mat disparity_image;
	sgbm(left_image_remap, right_image_remap, disparity_image);

	//show disparity image;
	/*
	cv::namedWindow("Disparity");
	cv::imshow("Disparity",disparity_image);
	cv::waitKey(0);
	return 0;*/

	/// convert to 3D points
	cv::Mat cloud_image; 
	cv::reprojectImageTo3D(disparity_image, cloud_image, Q);

	//convert cloud_image into point format of PCL
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	float px, py, pz;
	int pr, pg, pb;

	//scan cloud_image and convert its information in a Point Cloud
  	for (int i = 0; i < cloud_image.rows; i++)
  	{
  		for (int j = 0; j < cloud_image.cols; j++)
  		{

  			px= cloud_image.at<cv::Vec3f>(i,j)[0];
  			py= cloud_image.at<cv::Vec3f>(i,j)[1];
  			pz= cloud_image.at<cv::Vec3f>(i,j)[2];

  			pb= left_image.at<cv::Vec3b>(i,j)[0];
  			pg= left_image.at<cv::Vec3b>(i,j)[1];
  			pr= left_image.at<cv::Vec3b>(i,j)[2];

      		//Insert info into point cloud structure
      		pcl::PointXYZRGB point;
      		point.x = px;
      		point.y = py;
      		point.z = pz;

      		point.r = pr;
      		point.g = pg;
      		point.b = pb;

      		//delete unwanted infinites points and outliers
      		if(point.x>-30 && point.x<30 && point.y>-30 && point.y<30 && point.z>-3 && point.z <3){ 
      			point_cloud->points.push_back (point);
      		}
      	}
  	}

	/// visualize 3D points
	pcl::visualization::PCLVisualizer visualizer("PCL visualizer");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud);
	visualizer.addPointCloud<pcl::PointXYZRGB> (point_cloud, rgb, "point_cloud");
	timeComplete=clock()-init; //final time
	std::cout <<"Reconstruction complete in: "<<timeComplete <<std::endl;
	visualizer.spin();
	return 0;
}
