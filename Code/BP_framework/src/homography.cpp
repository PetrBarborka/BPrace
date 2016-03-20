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

        cv::FlannBasedMatcher matcher;
        matcher.match(desc1, desc2, matches);

        std::cout << "got matches";

        int min_dist = 0, max_dist = 0;

        for( int i = 0; i < matches.size(); i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        std::cout << "min and max distance between matched descriptors: "
                  << min_dist << " &  " << max_dist;

        std::sort(matches.begin(), matches.end(), compareMatchesByDistance);

        for( int i = 0; i < matches.size(); i++ )
        {
            if( matches[i].distance > threshold*min_dist )
            {
                break;
            }
            good_matches.push_back( matches[i] );
        }

        std::vector<cv::Point2f> pts1, pts2;

        for (int i = 0; i < good_matches.size(); i++){
            pts1.push_back(kpts1[ good_matches[i].queryIdx ].pt );
            pts2.push_back(kpts2[ good_matches[i].trainIdx ].pt );
        }

        int homography_method = CV_RANSAC; //(alt: 0 for basic least squares, CV_LMEDS for least medians)
        hmgr = cv::findHomography(pts1, pts2, homography_method);

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


}