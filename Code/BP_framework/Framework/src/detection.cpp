//
// Created by Tyler Durden on 3/17/16.
//

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "detection.hpp"

#include "utility.hpp"

namespace BP {

//  Constructor ------------------

    Detection::Detection(const cv::Mat &src_in,
                         const detection_method method_in,
                         const int maxPts_in)
        : src(src_in), method(method_in), maxPts(maxPts_in)
    {
        this->detect();
    }

//  Algorithm ---------------------
    void Detection::detect()  {

        std::vector<cv::KeyPoint> kpts;



//        std::cout << "detect() method runs. Detecting with ";
        if (getMethod() == DETECTION_HARRIS){
//            std::cout << "Harris\n";

            /// Parameters for Harris algorithm
            std::vector<cv::Point2f> corners;
            double qualityLevel = 0.01;
            double minDistance = 3;
            int blockSize = 3;
            bool useHarrisDetector = true;
            double k = 0.04;

            cv::goodFeaturesToTrack( getSrc(),
                                     corners,
                                     getMaxPts(),
                                     qualityLevel,
                                     minDistance,
                                     cv::Mat(),
                                     blockSize,
                                     useHarrisDetector,
                                     k );
            // convert vector of Points2f to a vector of KeyPoints
            pointsToKeypoints(corners, kpts);

        } else if (getMethod() == DETECTION_GFTT){
//            std::cout << "GFTT\n";
            /// Parameters for GFTT algorithm
            std::vector<cv::Point2f> corners;
            double qualityLevel = 0.01;
            double minDistance = 10;
            int blockSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;

            cv::goodFeaturesToTrack( getSrc(),
                                     corners,
                                     getMaxPts(),
                                     qualityLevel,
                                     minDistance,
                                     cv::Mat(),
                                     blockSize,
                                     useHarrisDetector,
                                     k );
            // convert vector of Points2f to a vector of KeyPoints
            pointsToKeypoints(corners, kpts);

        } else if (getMethod() == DETECTION_SIFT) {
//            std::cout << "SIFT\n";

            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.6;

            cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                                                    edgeThreshold, sigma);
            detector->detect(getSrc(), kpts);

//            patchSIFTOctaves(kpts);

//            std::vector<cv::KeyPoint>::iterator it = kpts.begin();
//            std::vector<cv::KeyPoint>::const_iterator ite = kpts.end();
//            for(; it < ite; it++){
//                (*it).octave = (*it).octave & 255;
//            }

        } else if (getMethod() == DETECTION_SURF) {
//            std::cout << "SURF\n";

//      //  SURF parameters
            double hessianThreshold = 100;
            int nOctaves = 4;
            int nOctaveLayers = 3;
            bool extended = false;
            bool upright = false;

            cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
            detector->detect(getSrc(), kpts );

        } else if (getMethod() == DETECTION_FAST) {
//            std::cout << "FAST\n";

//      //  FAST
            int threshold = 5;
            bool nonmaxSupression = true;
            int neighbourhood = cv::FastFeatureDetector::TYPE_9_16;

            cv::FAST(getSrc(), kpts, threshold, nonmaxSupression, neighbourhood);
//            std::cout << "FAST got " << kpts.size() << "kpts\n";

        } else if (getMethod() == DETECTION_MSER) {
//            std::cout << "MSER\n";

//      //  MSER
            int _delta = 5;
            int _min_area=60;
            int _max_area=14400;
            double _max_variation=0.25;
            double _min_diversity=.2;
            int _max_evolution=200;
            double _area_threshold=1.01;
            double _min_margin=0.003;
            int _edge_blur_size=5;

            cv::Ptr<cv::MSER> detector = cv::MSER::create( _delta, _min_area, _max_area,
                                                           _max_variation, _min_diversity,
                                                           _max_evolution, _area_threshold,
                                                           _min_margin, _edge_blur_size);

            detector->detect(getSrc(), kpts, cv::noArray());

        } else if (getMethod() == DETECTION_ORB) {
//            std::cout << "ORB\n";

//      //  ORB
            int nfeatures= maxPts;
            float scaleFactor=1.2f;
            int nlevels=8;
            int edgeThreshold=31;
            int firstLevel=0;
            int WTA_K=2;
            int scoreType=cv::ORB::HARRIS_SCORE;
            int patchSize=31;
            int fastThreshold=20;

            cv::Ptr<cv::ORB> detector = cv::ORB::create( nfeatures, scaleFactor, nlevels, edgeThreshold,
                                                         firstLevel, WTA_K, scoreType,
                                                         patchSize, fastThreshold);

            detector->detect(getSrc(), kpts, cv::noArray());

        } else {
            std::cout << "Detection error: unknown detection method";
        }
        std::sort(kpts.begin(), kpts.end(), compareKeypointsByResponse);
        topKeypoints(kpts, getMaxPts());

        this->keypoints = kpts;
    }

//  getters -----------------------
    std::vector<cv::KeyPoint> Detection::getKeypoints() {
        return this->keypoints;
    }

    cv::Mat Detection::getSrc(){
        return this->src;
    }
    detection_method Detection::getMethod(){
        return this->method;
    }
    int Detection::getMaxPts(){
        return this->maxPts;
    }


}
