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
//     ------- Initialization
        //  source images
        cv::Mat src1, src2;
        //  names of methods to use
        std::string det_method, desc_method;
        //  homography ground truth if available
        cv::Mat homography_gt;
        //  path to output picture if saving
        std::string out_pic_path, csv_path;
        //  draw results?
        bool show_pic, save_pic, save_csv;
        //  Maximum number of keypoints returned in detection
        int maxPts;
//     ------- Detection
        //  image key points
        std::vector<cv::KeyPoint> kpoints1, kpoints2;
        // execution times
        double time_det = 0;
//     ------- Description
        //  descriptors of keypoints
        cv::Mat descriptors1, descriptors2;
        // execution times
        double time_desc = 0;
//     ------- Matching
        //  keypoint matches between images
        std::vector<cv::DMatch> matches;
        //  picture showing matches on a pair
        cv::Mat outPic;
        // inlier matches indicator
        cv::Mat mask;
        double time_matching = 0;
//     ------- Homography
        //  Homography matrix
        cv::Mat homography;
        // matches actually used in homography generation
        std::vector<cv::DMatch> good_matches;
        //  data string about the whole operation & results
        std::string label, csv_row, filename;
        double hmg_distance = 0;
        // execution times
        double time_homography = 0;
    };

    class Homography {
        private:
            // how many times the minimal distance is still a good match?
            float threshold;
            cv::Mat desc1;
            cv::Mat desc2;
            std::vector<cv::DMatch> matches, good_matches;
            std::vector<cv::KeyPoint> kpts1, kpts2;
            cv::Mat hmgr, hmgr_gt, mask;
            double hmg_distance;
            bool flann;
            std::string descType;
            void compute();
        public:
            Homography( cv::Mat desc1_in,
                        cv::Mat desc2_in,
                        std::vector<cv::KeyPoint> kpts1_in,
                        std::vector<cv::KeyPoint> kpts2_in,
                        std::string descType_in,
                        cv::Mat hmgr_gt_in,
                        std::vector<cv::DMatch> matches_in);  // 0 = BRIEF, 1 = SIFT, 2 = SURF, 3 = ORB;
            std::vector<cv::KeyPoint> getDesc1();
            std::vector<cv::KeyPoint> getDesc2();
            std::vector<cv::DMatch> getMatches();
            std::vector<cv::DMatch> getGoodMatches();
            std::vector<cv::KeyPoint> getKpts1();
            std::vector<cv::KeyPoint> getKpts2();
            cv::Mat getHomography();
            cv::Mat getHomographyGt();
            cv::Mat getMask();
            std::string getDescType();
            double getHmgDistance() ;
    };

}

#endif //BP_HOMOGRAPHY_H
