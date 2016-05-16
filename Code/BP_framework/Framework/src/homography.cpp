//
// Created by Tyler Durden on 3/20/16.
//

#include "detection.hpp"
#include "description.hpp"
#include "utility.hpp"

//#include <iostream>
#include <fstream>
//#include <ctime>
#include <exception>

#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "boost/algorithm/string.hpp"

#include "homography.hpp"
#include "utility.hpp"

#include "json.hpp"

namespace BP {

    Homography::Homography( cv::Mat desc1_in,
                            cv::Mat desc2_in,
                            std::vector<cv::KeyPoint> kpts1_in,
                            std::vector<cv::KeyPoint> kpts2_in,
                            std::string descType_in,
                            cv::Mat hmgr_gt_in,
                            std::vector<cv::DMatch> matches_in)  // 0 = BRIEF, 1 = SIFT, 2 = SURF, 3 = ORB;
    : desc1(desc1_in), desc2(desc2_in), kpts1(kpts1_in),
      kpts2(kpts2_in), descType(descType_in), hmgr_gt(hmgr_gt_in),
      matches(matches_in)
    {
        compute();
    }


    void Homography::compute() {

        std::vector<cv::Point2f> pts1, pts2;

        pts1.resize(matches.size());
        pts2.resize(matches.size());

        for (int i = 0; i < matches.size(); i++){
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx;
            cv::Point2f pt1_to_push = kpts1[ queryIdx ].pt;
            cv::Point2f pt2_to_push = kpts2[ trainIdx ].pt;
            pts1[i] = pt1_to_push;
            pts2[i] = pt2_to_push;
        }

        if (matches.size() > 3) {
//            int homography_method = CV_RANSAC; //(alt: 0 for basic least squares, CV_LMEDS for least medians)
            int homography_method = CV_RANSAC;
            double ransacReprojThreshold = 2;
            hmgr = cv::findHomography(pts1, pts2, homography_method, ransacReprojThreshold, mask);

            if (mask.empty()){
                mask.create(matches.size(), 1, CV_8U);
            }

            std::vector<cv::DMatch>::iterator mitr = matches.begin();
            cv::MatIterator_<uchar> mask_itr = mask.begin<uchar>();
            std::vector<cv::DMatch>::const_iterator end_mitr = matches.cend();

            for (; mitr < end_mitr; mitr++, mask_itr++) {
                if (static_cast<bool>(*(mask_itr))) {
                    good_matches.push_back(*mitr);
                }
            }
        }else{
            std::cout << "WARNING: not enough matches to compute homography. Skipping\n";
            good_matches = matches;
        }
        if (good_matches.size() < 3){
            good_matches = matches;
        }

        if (hmgr_gt.size().width > 0 && hmgr.size().width > 0) {
            hmg_distance = getHomographyDistance(hmgr, hmgr_gt);
        } else {
            hmg_distance = -1;
        }

    }

    std::vector<cv::KeyPoint> Homography::getDesc1(){
        return this->desc1;
    }
    std::vector<cv::KeyPoint> Homography::getDesc2(){
        return this->desc2;
    }
    std::vector<cv::DMatch> Homography::getMatches(){
        return this->matches;
    }
    std::vector<cv::DMatch> Homography::getGoodMatches(){
        return this->good_matches;
    }
    std::vector<cv::KeyPoint> Homography::getKpts1(){
        return this->kpts1;
    }
    std::vector<cv::KeyPoint> Homography::getKpts2(){
        return this->kpts2;
    }
    cv::Mat Homography::getHomography(){
        return this->hmgr;
    }
    cv::Mat Homography::getHomographyGt(){
        return this->hmgr_gt;
    }
    cv::Mat Homography::getMask(){
        return this->mask;
    }
    std::string Homography::getDescType() {
        return this->descType;
    }
    double Homography::getHmgDistance() {
        return this->hmg_distance;
    }

}