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

//        std::cout << "got matches of size " << matches.size() << "\n";
//        std::cout << "matches: ";
//        PrintMatchVector(matches);

        int min_dist = matches[0].distance, max_dist = 0;

        for( int i = 0; i < matches.size(); i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        std::cout << "min and max distance between matched descriptors: "
                  << min_dist << " &  " << max_dist << "\n";

        std::sort(matches.begin(), matches.end(), compareMatchesByDistance);

//        std::cout << "matches sorted by distance: ";
//        PrintMatchVector(matches);

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
            cv::Point2f pt1_to_push = kpts1[ good_matches[i].queryIdx ].pt;
            cv::Point2f pt2_to_push = kpts2[ good_matches[i].queryIdx ].pt;
//            std::cout << "pushing Point2f pt1_to_push: " << pt1_to_push << "\n";
//            std::cout << "pushing Point2f pt2_to_push: " << pt2_to_push << "\n";
            pts1.push_back( pt1_to_push );
            pts2.push_back( pt2_to_push );
        }

        int homography_method = CV_RANSAC; //(alt: 0 for basic least squares, CV_LMEDS for least medians)
        double ransacReprojThreshold=3;
        cv::OutputArray mask=cv::noArray();
//        std::cout << "running \"findhomography()\"\n" ;
//        std::cout << "\npts1: \n" << pts1;
//        std::cout << "\npts2: \n" << pts2;
        hmgr = cv::findHomography(pts1, pts2, homography_method, ransacReprojThreshold, mask);

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