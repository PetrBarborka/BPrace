//
// Created by Tyler Durden on 3/18/16.
//

#ifndef BP_HOMOGRAPHY_H
#define BP_HOMOGRAPHY_H

//Including in header file because
//headers use types
#include "detection.hpp"
#include "description.hpp"

#include "json.hpp"

using json = nlohmann::json;

namespace BP {

    struct homography_t {
        //  source images
        cv::Mat src1, src2;
        //  image key points
        std::vector<cv::KeyPoint> kpoints1, kpoints2;
        //  descriptors of keypoints
        cv::Mat descriptors1, descriptors2;
        //  keypoint matches between images
        std::vector<cv::DMatch> matches, good_matches;
        //  Homography matrix
        cv::Mat homography, homography_gt;
        //  Maximum number of keypoints returned
        int maxPts;
        std::vector<int> det_methods;
        std::vector<int> desc_methods;
        // draw results?
        bool show;
        // inlier matches indicator
        cv::Mat mask;
        // execution times
        double time_desc, time_det, time_homography;
    };

    void computeHg(homography_t &hg, int detIdx = 0, int descIdx = 0);
    void computeAllHGs( std::vector<homography_t>& hgs, json pictures, json config, json output);


    class Homography {
        private:
            // how many times the minimal distance is still a good match?
            float threshold;
            cv::Mat desc1;
            cv::Mat desc2;
            std::vector<cv::DMatch> matches, good_matches;
            std::vector<cv::KeyPoint> kpts1, kpts2;
            cv::Mat hmgr, mask;
            bool flann;
            int descType;
            void compute();
        public:
            Homography( cv::Mat desc1_in,
                        cv::Mat desc2_in,
                        std::vector<cv::KeyPoint> kpts1_in,
                        std::vector<cv::KeyPoint> kpts2_in,
                        bool flann_in=1,  // bruteforce matcher = 0, flann matcher = 1
                        int descType_in=0 );  // 0 = BRIEF, 1 = SIFT, 2 = SURF, 3 = ORB;
            std::vector<cv::KeyPoint> getDesc1();
            std::vector<cv::KeyPoint> getDesc2();
            std::vector<cv::DMatch> getMatches();
            std::vector<cv::DMatch> getGoodMatches();
            std::vector<cv::KeyPoint> getKpts1();
            std::vector<cv::KeyPoint> getKpts2();
            cv::Mat getHomography();
            cv::Mat getMask();
            bool getFlann();
            int getDescType();
    };

}

#endif //BP_HOMOGRAPHY_H
