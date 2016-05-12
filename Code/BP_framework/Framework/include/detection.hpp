//
// Created by Tyler Durden on 3/17/16.
//

#ifndef BP_DETECTION_H
#define BP_DETECTION_H

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

namespace BP {

    enum detection_method { DETECTION_HARRIS = 0,
        DETECTION_GFTT = 1,
        DETECTION_SIFT = 2,
        DETECTION_SURF = 3,
        DETECTION_FAST = 4,
        DETECTION_MSER = 5,
        DETECTION_ORB = 6  };

    class CV_EXPORTS_W HarrisDetector: public cv::Feature2D {
    public:
        CV_WRAP static cv::Ptr<HarrisDetector> create(  int maxPts = 1000,
                                                        double qualityLevel = 0.01,
                                                        double minDistance = 3,
                                                        int blockSize = 3,
                                                        bool useHarrisDetector = true,
                                                        double k = 0.04);

        CV_WRAP void detect( cv::InputArray image,
                             CV_OUT std::vector<cv::KeyPoint>& keypoints,
                             cv::InputArray mask=cv::noArray() );
    private:
        HarrisDetector( int maxPts_in,
                        double qualityLevel_in,
                        double minDistance_in,
                        int blockSize_in,
                        bool useHarrisDetector_in,
                        double k_in);
        int maxPts;
        double qualityLevel;
        double minDistance;
        int blockSize;
        bool useHarrisDetector;
        double k;
    };

    class CV_EXPORTS_W FastDetector: public cv::Feature2D {
    public:
        CV_WRAP static cv::Ptr<FastDetector> create( int threshold = 5,
                                                     bool nonmaxSupression = true,
                                                     int neighbourhood = cv::FastFeatureDetector::TYPE_9_16);

        CV_WRAP void detect( cv::InputArray image,
                             CV_OUT std::vector<cv::KeyPoint>& keypoints,
                             cv::InputArray mask=cv::noArray() );
    private:
        FastDetector( int threshold_in = 5,
                      bool nonmaxSupression_in = true,
                      int neighbourhood_in = cv::FastFeatureDetector::TYPE_9_16);
        int threshold;
        bool nonmaxSupression;
        int neighbourhood;
    };

    class Detection {

    private:
        const cv::Mat &src;
        const detection_method method;
        const int maxPts;
        std::vector<cv::KeyPoint> keypoints;

        void detect();

    public:
        Detection(const cv::Mat &src, const detection_method method,
                  const int maxPts);
        std::vector<cv::KeyPoint> getKeypoints();
        cv::Mat getSrc();
        detection_method getMethod();
        int getMaxPts();
    };

}

#endif //BP_DETECTION_H
