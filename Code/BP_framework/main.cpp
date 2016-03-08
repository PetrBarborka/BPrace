#include <QCoreApplication>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>

// ------- constants ----------
int METHOD_HARRIS = 1, METHOD_GFTT = 2, METHOD_SIFT = 3;


void detect(cv::InputArray src, std::vector<cv::KeyPoint> pts,
            int maxPts, int method);

void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec);

void showKeypoints(cv::InputArray &in_mat, cv::InputOutputArray &out_mat,
                   std::vector<cv::KeyPoint> &kpts, std::string winname);

//MAIN ========================================================================
int main(int argc, char *argv[])
{

//    for(int i = 0; i < argc; ++i ){
//        std::cout << "\narg " << i << ": " << argv[i] << "\n";
//    }

    cv::Mat src;

    src = cv::imread(argv[1], 0);
    if (!src.data){
        std::cout << "Failed to load image " << argv[1];
        return 1;
    }

    cv::namedWindow("testwin", cv::WINDOW_AUTOSIZE);
    cv::imshow("testwin", src);

    std::vector<cv::KeyPoint> kpoints;
    int method = METHOD_SIFT;
    int maxPts = 100;

    detect(src, kpoints, maxPts, method);

    cv::Mat drawTest;
    src.copyTo(drawTest);

    cv::namedWindow("Wrapper_test", cv::WINDOW_AUTOSIZE);
    showKeypoints(src, drawTest, kpoints, "Wrapper_test");

    cv::waitKey(0);
    return 0;
}

//DETECTION ===================================================================
void detect(cv::InputArray &src, std::vector<cv::KeyPoint> pts,
            int maxPts, int method){
    std::cout << "\ndetect() method runs\n";
    if (method == METHOD_HARRIS){
        std::cout << "\ngoing for Harris\n";

        /// Parameters for Harris algorithm
        std::vector<cv::Point2f> corners;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3;
        bool useHarrisDetector = true;
        double k = 0.04;

        cv::goodFeaturesToTrack( src,
                     corners,
                     maxPts,
                     qualityLevel,
                     minDistance,
                     cv::Mat(),
                     blockSize,
                     useHarrisDetector,
                     k );
//      convert vector of Points2f to a vector of KeyPoints
        pointsToKeypoints(corners, pts);
    } else if (method == METHOD_GFTT){
        std::cout << "\ngoing for GFTT\n";
        /// Parameters for GFTT algorithm
        std::vector<cv::Point2f> corners;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;

        cv::goodFeaturesToTrack( src,
                     corners,
                     maxPts,
                     qualityLevel,
                     minDistance,
                     cv::Mat(),
                     blockSize,
                     useHarrisDetector,
                     k );
    } else if (method == METHOD_SIFT) {
        std::cout << "going for SIFT";

//        int minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SIFT> detector;
        detector.create(maxPts, 3, 0.04, 10, 1.6);
        detector->detect(src, pts);

    } else {
        std::cout << "error: unknown detection method";
    }
//    cv::Point2f pnt(0,0);
//    cv::KeyPoint kp(pnt, 1., -1., 0., 0, -1);
//    std::vector<cv::KeyPoint> kpvec;
//    kpvec.push_back(kp);
//    std::cout << "testing KeyPoint: " << kpvec[0].pt;
}

//VISUAL - DRAWING ============================================================
void showKeypoints(cv::InputArray &in_mat, cv::InputOutputArray &out_mat,
                   std::vector<cv::KeyPoint> &kpts, std::string winname){
    cv::drawKeypoints(in_mat, kpts, out_mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
    cv::imshow(winname, out_mat);
}

//MISC - UTILITIES ============================================================
void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec){
    std::vector<cv::Point2f>::const_iterator itb = in_vec.begin();
    std::vector<cv::Point2f>::const_iterator ite = in_vec.end();
    for (; itb < ite; itb++) {
//        std::cout << "pTK() is pushing " << *itb << std::endl;
        out_vec.push_back(cv::KeyPoint(*itb, 1., -1., 0., 0, -1));
    }
}
