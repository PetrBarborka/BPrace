//
// Created by Tyler Durden on 3/17/16.
//

#ifndef BP_DETECTION_H
#define BP_DETECTION_H

namespace BP {

    enum detection_method { DETECTION_HARRIS = 1,
        DETECTION_GFTT = 2,
        DETECTION_SIFT = 3,
        DETECTION_SURF = 4,
        DETECTION_FAST = 5,
        DETECTION_MSER = 6,
        DETECTION_ORB = 7  };

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
