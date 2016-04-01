//
// Created by Tyler Durden on 3/18/16.
//

#ifndef BP_UTILITY_H
#define BP_UTILITY_H

namespace BP {

    void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec);
    void topKeypoints(std::vector<cv::KeyPoint> &pts, int ammount);
    void showKeypoints(cv::InputArray &in_mat, std::vector<cv::KeyPoint> &kpts, std::string winname);

    void PrintKeyPoint(const cv::KeyPoint &kp);
    void PrintKPVector(const std::vector<cv::KeyPoint> &kpv);
    void PrintMatch(const cv::DMatch &match);
    void PrintMatchVector(const std::vector<cv::DMatch> &mv);

    bool compareKeypointsByResponse(const cv::KeyPoint &k1, const cv::KeyPoint &k2);
    bool compareMatchesByDistance(const cv::DMatch &m1, const cv::DMatch &m2);
}

#endif //BP_UTILITY_H
