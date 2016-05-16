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

//  FAST Detector implementation --------------------------------------------------------------------------------

    FastDetector::FastDetector(int threshold_in, bool nonmaxSupression_in, int neighbourhood_in)
                : threshold(threshold_in), nonmaxSupression(nonmaxSupression_in),
                  neighbourhood(neighbourhood_in)
                { }

    cv::Ptr<FastDetector> FastDetector::create(int threshold, bool nonmaxSupression, int neighbourhood)
    {
        FastDetector d = FastDetector(threshold, nonmaxSupression, neighbourhood);
        return cv::makePtr<FastDetector>(d);
    }

    void FastDetector::detect(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints,
                              const cv::_InputArray &mask)
    {
        cv::FAST(image, keypoints, threshold, nonmaxSupression, neighbourhood);
    }

//  Harris and GFTT Detector implementation ----------------------------------------------------------------------

    HarrisDetector::HarrisDetector( int maxPts_in,
                                    double qualityLevel_in,
                                    double minDistance_in,
                                    int blockSize_in,
                                    bool useHarrisDetector_in,
                                    double k_in)
            : maxPts(maxPts_in), qualityLevel(qualityLevel_in), minDistance(minDistance_in), blockSize(blockSize_in),
              useHarrisDetector(useHarrisDetector_in), k(k_in) {}

    cv::Ptr<HarrisDetector> HarrisDetector::create(int maxPts, double qualityLevel,
                                                   double minDistance, int blockSize,
                                                   bool useHarrisDetector, double k)
    {
        HarrisDetector d = HarrisDetector(maxPts, qualityLevel, minDistance, blockSize,
                                          useHarrisDetector, k);
        return cv::makePtr<HarrisDetector>(d);
    }



    void HarrisDetector::detect(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints,
                                const cv::_InputArray &mask) {

        std::vector<cv::Point2f> corners;

        cv::goodFeaturesToTrack( image,
                                 corners,
                                 maxPts,
                                 qualityLevel,
                                 minDistance,
                                 mask,
                                 blockSize,
                                 useHarrisDetector,
                                 k );
        // convert vector of Points2f to a vector of KeyPoints
        pointsToKeypoints(corners, keypoints);
    }

    typedef HarrisDetector GFTTDetector;

//  Detection class implementation ----------------------------------------------------------------------

//  Constructor ------------------

    Detection::Detection(const cv::Mat &src_in,
                         const std::string method_in,
                         const int maxPts_in)
        : src(src_in), method(method_in), maxPts(maxPts_in)
    {
        this->detect();
    }

//  Algorithm ---------------------
    void Detection::detect()  {

        std::vector<cv::KeyPoint> kpts;
        cv::Ptr<cv::Feature2D> detector;
        cv::Mat mask = cv::Mat();

//        std::cout << "detect() method runs. Detecting with ";
        if (getMethod() == "Harris"){
//            std::cout << "Harris\n";

            /// Parameters for Harris algorithm
            double qualityLevel = 0.01;
            double minDistance = 3;
            int blockSize = 3;
            bool useHarrisDetector = true;
            double k = 0.04;

            detector = HarrisDetector::create(getMaxPts(), qualityLevel, minDistance,
                                              blockSize, useHarrisDetector, k);

        } else if (getMethod() == "GFTT"){
//            std::cout << "GFTT\n";
            /// Parameters for GFTT algorithm
            double qualityLevel = 0.01;
            double minDistance = 10;
            int blockSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;

            detector = GFTTDetector::create(getMaxPts(), qualityLevel, minDistance,
                                            blockSize, useHarrisDetector, k);

        } else if (getMethod() == "SIFT") {
//            std::cout << "SIFT\n";

            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.6;

            detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                     edgeThreshold, sigma);

        } else if (getMethod() == "SURF") {
//            std::cout << "SURF\n";

//      //  SURF parameters
            double hessianThreshold = 100;
            int nOctaves = 4;
            int nOctaveLayers = 3;
            bool extended = false;
            bool upright = false;

            detector = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

        } else if (getMethod() == "FAST") {
//            std::cout << "FAST\n";

//      //  FAST
            int threshold = 5;
            bool nonmaxSupression = true;
            int neighbourhood = cv::FastFeatureDetector::TYPE_9_16;

            detector = FastDetector::create(threshold, nonmaxSupression, neighbourhood);

        } else if (getMethod() == "MSER") {
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

            detector = cv::MSER::create( _delta, _min_area, _max_area,
                                         _max_variation, _min_diversity,
                                         _max_evolution, _area_threshold,
                                         _min_margin, _edge_blur_size);

        } else if (getMethod() == "ORB") {
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

            detector = cv::ORB::create( nfeatures, scaleFactor, nlevels, edgeThreshold,
                                                         firstLevel, WTA_K, scoreType,
                                                         patchSize, fastThreshold);

        } else {
            std::cout << "Detection error: unknown detection method: " << getMethod() << "\n";
        }

        detector->detect(getSrc(), kpts, mask);

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

    std::string Detection::getMethod(){
        return this->method;
    }

    int Detection::getMaxPts(){
        return this->maxPts;
    }


}
