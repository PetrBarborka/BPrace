//
// Created by Tyler Durden on 3/17/16.
//

#ifndef BP_DETECTION_H
#define BP_DETECTION_H

#include <iostream>

#include "opencv2/core.hpp"

namespace BP {

    enum detection_method { DETECTION_HARRIS = 0,
        DETECTION_GFTT = 1,
        DETECTION_SIFT = 2,
        DETECTION_SURF = 3,
        DETECTION_FAST = 4,
        DETECTION_MSER = 5,
        DETECTION_ORB = 6  };

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
