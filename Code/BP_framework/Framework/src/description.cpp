//
// Created by Tyler Durden on 3/20/16.
//

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "description.hpp"
#include "utility.hpp"

namespace BP {


    Description::Description(cv::Mat src_in, std::vector<cv::KeyPoint> kpoints_in, std::string method_in)
            : src(src_in), keypoints(kpoints_in), method(method_in){
        this->describe();
    }

    void Description::describe() {

        cv::Mat desc;
        cv::Ptr<cv::Feature2D> extractor;

        if (getMethod() == "BRIEF"){

            int bytes = 32;
            bool use_orientation = true;

            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);

        }else if (getMethod() == "SIFT"){

            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.6;

            extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                      edgeThreshold, sigma);

        } else if (getMethod() == "SURF"){

            double hessianThreshold = 100;
            int nOctaves = 4;
            int nOctaveLayers = 3;
            bool extended = true;
            bool upright = false;

            extractor = cv::xfeatures2d::SURF::create(hessianThreshold,
                                                      nOctaves,
                                                      nOctaveLayers,
                                                      extended,
                                                      upright);

        } else if (getMethod() == "ORB"){

            int nfeatures = getKeypoints().size();
            float scaleFactor=1.2f;
            int nlevels=8;
            int edgeThreshold=31;
            int firstLevel=0;
            int WTA_K=2;
            int scoreType=cv::ORB::HARRIS_SCORE;
            int patchSize=31;
            int fastThreshold=20;

            extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                                                         WTA_K, scoreType, patchSize, fastThreshold);
        } else {
            std::cout << "error: unknown description method: " << getMethod() << "\n";
        }

        std::vector<cv::KeyPoint> kpts = getKeypoints();
        extractor->compute(getSrc(), kpts, desc);

        this->descriptors = desc;
    }
    std::vector<cv::KeyPoint> Description::getKeypoints(){
        return this->keypoints;
    }
    cv::Mat Description::getDescriptors(){
        return this->descriptors;
    }
    std::string Description::getMethod(){
        return this->method;
    }
    cv::Mat Description::getSrc(){
        return this->src;
    }


}