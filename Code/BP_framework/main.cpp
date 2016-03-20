#include <QCoreApplication>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

#include "utility.hpp"
#include "detection.hpp"
#include "description.hpp"

using namespace BP;


//MAIN ========================================================================
int main(int argc, char *argv[])
{
    for(int i = 0; i < argc; ++i ){
        std::cout << "\narg " << i << ": " << argv[i] << "\n";
    }

    if (argc < 2){
      std::cout << "usage: BP pathtopicture";
      return 1;
    }

//  source image
    cv::Mat src, src_color;
//  image key points
    std::vector<cv::KeyPoint> kpoints;
//  descriptors of keypoints
    cv::Mat descriptors;

    src_color = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::cvtColor(src_color, src, cv::COLOR_BGR2GRAY);

    if (!src.data){
        std::cout << "Working directory == " << std::string(argv[0]) << "\n";
        std::cout << "Failed to load image " << std::string(argv[1]);
        return 1;
    }

    BP::detection_method method = BP::DETECTION_HARRIS;
    int maxPts = 100;

    Detection det(src, method, maxPts);
    kpoints = det.getKeypoints();

    Description desc( src, kpoints, BP::DESCRIPTION_ORB );
    descriptors = desc.getDescriptors();

    std::cout << "descriptors: " << descriptors;

    cv::Mat drawTest;
    src.copyTo(drawTest);

    cv::namedWindow("Wrapper_test", cv::WINDOW_AUTOSIZE);
    showKeypoints(src_color, kpoints, "Wrapper_test");

    std::cout << "detected " <<  kpoints.size()  << "points";
//    PrintKPVector(kpoints);

    cv::waitKey(0);
    return 0;
}



















