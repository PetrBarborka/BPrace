#include <QCoreApplication>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

#include "utility.hpp"
#include "detection.hpp"
#include "description.hpp"
#include "homography.hpp"

using namespace BP;


//MAIN ========================================================================
int main(int argc, char *argv[])
{
    for(int i = 0; i < argc; ++i ){
        std::cout << "\narg " << i << ": " << argv[i] << "\n";
    }

    if (argc < 2){
      std::cout << "usage: BP img1 img2";
      return 1;
    }

//  source image
    cv::Mat src1, src2;
//  image key points
    std::vector<cv::KeyPoint> kpoints1, kpoints2;
//  descriptors of keypoints
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::DMatch> matches, good_matches;

    float matchingThreshold = 3;

    src1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    src2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (!src1.data ){
        std::cout << "Working directory == " << std::string(argv[0]) << "\n";
        std::cout << "Failed to load image " << std::string(argv[1]);
        return 100;
    }

    if (!src2.data){
        std::cout << "Working directory == " << std::string(argv[0]) << "\n";
        std::cout << "Failed to load image " << std::string(argv[2]);
        return 100;
    }

    detection_method det_method = DETECTION_SIFT;
    description_method desc_method = DESCRIPTION_SIFT;
    int maxPts = 100;

    Detection det1(src1, det_method, maxPts);
    Detection det2(src2, det_method, maxPts);
    kpoints1 = det1.getKeypoints();
    kpoints2 = det2.getKeypoints();

    Description desc1( src1, kpoints1, desc_method );
    Description desc2( src2, kpoints2, desc_method );
    descriptors1 = desc1.getDescriptors();
    descriptors2 = desc2.getDescriptors();

    Homography hmg(descriptors1, descriptors2,
                    kpoints1, kpoints2, matchingThreshold);

    cv::Mat hmgr = hmg.getHomography();
    good_matches = hmg.getGoodMatches();

    cv::Mat img_matches;

    cv::drawMatches(src1, kpoints1, src2, kpoints2, good_matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );

//    std::cout << "descriptors: " << descriptors;

//    cv::Mat drawTest;
//    src.copyTo(drawTest);

//    cv::namedWindow("Wrapper_test", cv::WINDOW_AUTOSIZE);
//    showKeypoints(src_color, kpoints, "Wrapper_test");
//
//    std::cout << "detected " <<  kpoints.size()  << "points";
//    PrintKPVector(kpoints);

    cv::waitKey(0);
    return 0;
}



















