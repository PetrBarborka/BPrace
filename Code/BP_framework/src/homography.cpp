//
// Created by Tyler Durden on 3/20/16.
//

#include <iostream>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgproc.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "description.hpp"
#include "homography.hpp"
#include "utility.hpp"

namespace BP {

    Homography::Homography( cv::Mat desc1_in,
                            cv::Mat desc2_in,
                            std::vector<cv::KeyPoint> kpts1_in,
                            std::vector<cv::KeyPoint> kpts2_in,
                            float threshold_in)
    : desc1(desc1_in), desc2(desc2_in), threshold(threshold_in),
      kpts1(kpts1_in), kpts2(kpts2_in)
    {
        compute();
    }

    void Homography::compute() {

        std::cout << "\nHomography::compute() method runs\n";

        cv::BFMatcher matcher(cv::NORM_L2, 1);
//        cv::FlannBasedMatcher matcher;

//      Flann compatibility conversion
        if(desc1.type()!=CV_32F) {
            desc1.convertTo(desc1, CV_32F);
            desc2.convertTo(desc2, CV_32F);
        }
        matcher.match(desc1, desc2, matches);

        double min_dist = matches[0].distance, max_dist = 0;

        for( int i = 0; i < matches.size(); i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        std::cout << "min and max distance between matched descriptors: "
                  << min_dist << " &  " << max_dist << "\n";

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
        int homography_method = CV_RANSAC; //(alt: 0 for basic least squares, CV_LMEDS for least medians)
        double ransacReprojThreshold=3;
        hmgr = cv::findHomography(pts1, pts2, homography_method, ransacReprojThreshold, mask);
        std::vector<cv::DMatch>::iterator mitr = matches.begin();
        cv::MatIterator_<uchar> mask_itr = mask.begin<uchar>();
        std::vector<cv::DMatch>::const_iterator end_mitr = matches.cend();
        for (; mitr<end_mitr; mitr++, mask_itr++)
        {
            if (static_cast<bool>(*(mask_itr)))
            {
                good_matches.push_back(*mitr);
            }
        }
        std::cout << "Matches = " << good_matches.size() << " inliers out of " << matches.size() << "\n";

    }

    float Homography::getThreshold(){
        return this->threshold;
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
    cv::Mat Homography::getMask(){
        return this->mask;
    }


}