#include <QCoreApplication>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <iostream>

//using namespace cv;
//using namespace cv::xfeatures2d;

// ------- constants ----------
int METHOD_HARRIS = 1, METHOD_GFTT = 2, METHOD_SIFT = 3, METHOD_SURF = 4;

// ------- fcn declarations -------------
void detect(cv::InputArray src, std::vector<cv::KeyPoint> &pts,
            int maxPts, int method);
void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec);
void topKeypoints(std::vector<cv::KeyPoint> &pts, int ammount);
void showKeypoints(cv::InputArray &in_mat, std::vector<cv::KeyPoint> &kpts, std::string winname);
void PrintKeyPoint(const cv::KeyPoint &kp);
void PrintKPVector(const std::vector<cv::KeyPoint> &kpv);

//MAIN ========================================================================
int main(int argc, char *argv[])
{
//    for(int i = 0; i < argc; ++i ){
//        std::cout << "\narg " << i << ": " << argv[i] << "\n";
//    }

    cv::Mat src, src_color;

    src_color = cv::imread(argv[1], 1);
    cv::cvtColor(src_color, src, cv::COLOR_BGR2GRAY);

//    std::cout << "src.data: " << src.data;

    if (!src.data){
        std::cout << "Failed to load image " << std::string(argv[0]) + "/" + std::string(argv[1]);
        return 1;
    }

    std::vector<cv::KeyPoint> kpoints;
    int method = METHOD_SIFT;
    int maxPts = 100;

    detect(src, kpoints, maxPts, method);

    cv::Mat drawTest;
    src.copyTo(drawTest);

    cv::namedWindow("Wrapper_test", cv::WINDOW_AUTOSIZE);
    showKeypoints(src_color, kpoints, "Wrapper_test");

    std::cout << "detected " <<  kpoints.size()  << "points";
    PrintKPVector(kpoints);

    cv::waitKey(0);
    return 0;
}

//DETECTION ===================================================================
void detect(cv::InputArray &src, std::vector<cv::KeyPoint> &pts,
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
        // convert vector of Points2f to a vector of KeyPoints
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
        // convert vector of Points2f to a vector of KeyPoints
        pointsToKeypoints(corners, pts);

    } else if (method == METHOD_SIFT) {
        std::cout << "going for SIFT = unimplemented";

//        int minHessian = 400;
//        cv::Ptr<cv::xfeatures2d::SIFT> detector;
//        detector->create(maxPts, 3, 0.04, 10, 1.6);
//        detector->detect(src, pts);
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;

        cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                                                edgeThreshold, sigma);
        detector->detect(src, pts );

    } else if (method == METHOD_SURF) {
        std::cout << "going for SURF";

//      //  SURF parameters
        double hessianThreshold = 100;
        int nOctaves = 4;
        int nOctaveLayers = 3;
        bool extended = false;
        bool upright = false;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
        detector->detect( src, pts );

    } else {
        std::cout << "error: unknown detection method";
    }
    topKeypoints(pts, maxPts);
}


//VISUAL - DRAWING ============================================================
void showKeypoints(cv::InputArray &in_mat, std::vector<cv::KeyPoint> &kpts, std::string winname){
    cv::Mat out_mat;
    in_mat.copyTo(out_mat);
    cv::drawKeypoints(in_mat, kpts, out_mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
    cv::imshow(winname, out_mat);
}

//MISC - UTILITIES ============================================================
void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec){
    std::vector<cv::Point2f>::const_iterator itb = in_vec.begin();
    std::vector<cv::Point2f>::const_iterator ite = in_vec.end();
    for (; itb < ite; itb++) {
        std::cout << "pTK() is pushing " << *itb << std::endl;
        out_vec.push_back(cv::KeyPoint(*itb, 30., 180., 1000., 0, -1));
    }
    std::cout << "points to keypoints:\n";
    PrintKPVector(out_vec);
}
void topKeypoints(std::vector<cv::KeyPoint> &pts, int ammount){
 int remove = pts.size() - ammount;
 for (; remove > 0; remove --){
    pts.pop_back();
 }
}
void PrintKeyPoint(const cv::KeyPoint &kp){
      std::cout << "--\nKeypoint: " << kp.pt <<
            " size: " << kp.size <<
            " angle: " << kp.angle <<
            " response: " << kp.response <<
            " octave: " << kp.octave <<
            " class_id: " << kp.class_id << std::endl;}
void PrintKPVector(const std::vector<cv::KeyPoint> &kpv){

    if (kpv.empty()){
        std::cout << "PrintKPV error: input vector empty!\n";
    }
    std::vector<cv::KeyPoint>::const_iterator it = kpv.begin();
    std::vector<cv::KeyPoint>::const_iterator ite = kpv.end();
    for(; it < ite; it++){
        PrintKeyPoint(*it);
    }
}
