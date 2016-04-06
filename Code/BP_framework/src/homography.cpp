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

#include "json.hpp"

namespace BP {

    void computeHg(homography_t &hg, int detIdx, int descIdx)
    {

        std::clock_t begin = std::clock();
        Detection det1(hg.src1, hg.det_methods[detIdx], hg.maxPts);
        Detection det2(hg.src2, hg.det_methods[detIdx], hg.maxPts);
        hg.kpoints1 = det1.getKeypoints();
        hg.kpoints2 = det2.getKeypoints();

        // if description is SIFT, but detection is not SIFT, we need to unpack
        // and patch octave
        if (detIdx == 2 & descIdx != 1 ){
            patchSIFTOctaves(hg.kpoints1);
            patchSIFTOctaves(hg.kpoints2);
        }
        std::clock_t end = std::clock();
        hg.time_det = double(end-begin)/CLOCKS_PER_SEC;

//    std::cout << "Keypoints 1: \n";
//    PrintKPVector(hg.kpoints1, 5);
//    std::cout << "Keypoints 2: \n";
//    PrintKPVector(hg.kpoints2, 5);

        begin = std::clock();
        Description desc1( hg.src1, hg.kpoints1, hg.desc_methods[descIdx] );
        Description desc2( hg.src2, hg.kpoints2, hg.desc_methods[descIdx] );
        hg.descriptors1 = desc1.getDescriptors();
        hg.descriptors2 = desc2.getDescriptors();
        end = std::clock();
        hg.time_desc = double(end-begin)/CLOCKS_PER_SEC;

//    std::cout << "descriptors1: \n" << hg.descriptors1 << "\n";
//    std::cout << "descriptors2: \n" << hg.descriptors2 << "\n";

        if ( hg.descriptors1.empty() ) {
            std::cout << "WARNING: descriptors 1 empty";
        }
        if ( hg.descriptors2.empty() ) {
            std::cout << "WARNING: descriptor 2 empty";
        }

        begin = std::clock();
        Homography hmg(hg.descriptors1, hg.descriptors2,
                       hg.kpoints1, hg.kpoints2, hg.matchingThreshold,
                       0, hg.desc_methods[descIdx]);
        end = std::clock();
        hg.time_homography = double(end-begin)/CLOCKS_PER_SEC;
        hg.homography = hmg.getHomography();
        hg.matches = hmg.getMatches();
        hg.good_matches = hmg.getGoodMatches();
        hg.mask = hmg.getMask();

//        std::cout << "Got " << hg.matches.size() << " matches, " << hg.good_matches.size() << " good ones.\n";

    }
    void computeAllHGs( std::vector<homography_t> & hgs, json pictures, json config, json output )
    {
        std::string pics_output_path;
        pics_output_path = *(output.find("picspath"));
        std::string csv_output_path;
        csv_output_path = *(output.find("csvpath"));

        homography_t hg;

        std::ofstream csv_out;
        if (!csv_output_path.empty()) {
            try {
                csv_out.open(csv_output_path);
//                std::cout << "opening  file w/ filename: " << csv_output_path << "\n";
            }
            catch (char const *e)
            {
                std::cout << e;
            }
        }

        for(json::iterator it = pictures.begin(); it != pictures.end(); it++)
        {
//            std::cout << "=========================\n";
//            std::cout << it.key() << ": picture1: " << it.value()[0] << " picture2: " << it.value()[1] << "\n";

            // load ground truth
            if (it.value().size() > 2) {
                try
                {
                    hg.homography_gt = readMatFromTextFile(it.value()[2]);
//                    std::cout << "homography ground truth: \n" << hg.homography_gt;
                }
                catch (std::exception e)
                {
                    std::cout << "Ground truth matrix parsing failed with exception: " << e.what() << "\n";
                }
            }
            else
            {
//                std::cout << "Homography ground truth unavailable\n";
            }

            hg.src1 = cv::imread(it.value()[0], cv::IMREAD_GRAYSCALE);
            hg.src2 = cv::imread(it.value()[1], cv::IMREAD_GRAYSCALE);

            if (!hg.src1.data)
            {
                std::cout << "ERROR: Failed to load image " << it.value()[0] << "\n";
            }

            if (!hg.src2.data)
            {
                std::cout << "ERROR: Failed to load image " << it.value()[1] << "\n";
            }

            parseConfig(config, hg);
            for (int detIdx = 0; detIdx < hg.det_methods.size(); detIdx++)
            {
                for (int descIdx = 0; descIdx < hg.desc_methods.size(); descIdx++)
                {

                    computeHg(hg, detIdx, descIdx);

                    double hmg_distance;

//                    std::cout << "homography: \n" << hg.homography << "\n\n";
                    if (hg.homography_gt.size().width > 0)
                    {
//                        std::cout << "homography ground truth: \n" << hg.homography_gt << "\n\n";
                        hmg_distance = getHomographyDistance(hg.homography, hg.homography_gt);
//                        std::cout << "homography distance: " << hmg_distance << "\n";
                    }
                    else
                    {
                        std::cout << "WARNING: Homography ground truth unavailable";
                        hmg_distance = -1;
                    }

//                    std::cout   << "det time: " << hg.time_det
//                    << " desc time: " << hg.time_desc
//                    << " hmg time: " << hg.time_homography << "\n";

                    hgs.push_back(hg);

                    std::vector<std::string> detLegend = {  "Harris",
                                                            "GFTT",
                                                            "SIFT",
                                                            "SURF",
                                                            "FAST",
                                                            "MSER",
                                                            "ORB"};

                    std::vector<std::string> descLegend = {  "BRIEF",
                                                             "SIFT",
                                                             "SURF",
                                                             "ORB"};

                    std::stringstream label;
                    cv::Mat img_matches;
                    std::string lpic1 = it.value()[0];
                    std::string lpic2 = it.value()[1];
                    std::vector<std::string> words;
                    boost::split(words, lpic1, boost::is_any_of(" \\/\"."));
                    lpic1 = *(words.end()-2);
                    boost::split(words, lpic2, boost::is_any_of(" \\/\"."));
                    lpic2 = *(words.end()-2);
                    label << lpic1 << "_" << lpic2 << "_det_" << detLegend[hg.det_methods[detIdx]]
                    << "_desc_" << descLegend[hg.desc_methods[descIdx]];


                    cv::Scalar good_color = cv::Scalar(35, 180, 40);
                    cv::Scalar bad_color = cv::Scalar(35, 40, 180);

//                std::cout << "Size of mask & matches: " << hg.matches.size() << " & " << hg.mask.size() << "\n";

//                img_matches.resize(cv::Size(hg.src1.size().width + hg.src2.size().width, hg.src2.size().height));
                    cv::drawMatches(hg.src1, hg.kpoints1, hg.src2, hg.kpoints2, hg.matches, img_matches,
                                    good_color, bad_color, hg.mask,
                                    cv::DrawMatchesFlags::DEFAULT);
                    if (hg.show)
                    {
                        //-- Show detected matches
                        cv::imshow(label.str(), img_matches);
                    }
                    std::string filename;
                    if (!pics_output_path.empty())
                    {
                        try
                        {
                            filename = pics_output_path + label.str() + ".png";
                            std::cout << "writing file w/ filename: " << filename.c_str() << "\n";
                            cv::imwrite(filename, img_matches);
                        }
                        catch (std::exception e)
                        {
                            std::cout << e.what();
                            break;
                        }
                    }
                    else
                    {
                        std::cout << "Pics path empty, not writing png file.\n";
                    }
                    if (!csv_output_path.empty()) {
                        try {
                            csv_out << pics_output_path << ", " // folder
                            << it.value()[0] << ", " // pic1
                            << it.value()[1] << ", " // pic2
                            << detLegend[hg.det_methods[detIdx]] << ", " // detection method
                            << descLegend[hg.desc_methods[descIdx]] << ", " // detection method
                            << hg.matches.size() << ", " // matches
                            << hg.good_matches.size() << ", " // inliers
                            << hmg_distance << ", " // hmg_distance
                            << hg.time_det << ", " // detection time
                            << hg.time_desc << ", " // description time
                            << hg.time_homography << ", " // homography & matching time
                            << filename << ", " // saved as
                            << "\n";
                        }
                        catch (std::exception e)
                        {
                            std::cout << e.what();
                            break;
                        }
                    }
                    else
                    {
                        std::cout << "Csv path empty, not writing csv file.\n";
                    }
                }
            }
        }
        csv_out.close();

    }

    Homography::Homography( cv::Mat desc1_in,
                            cv::Mat desc2_in,
                            std::vector<cv::KeyPoint> kpts1_in,
                            std::vector<cv::KeyPoint> kpts2_in,
                            float threshold_in,
                            bool flann_in,  // bruteforce matcher = 0, flann matcher = 1
                            int descType_in )  // 0 = BRIEF, 1 = SIFT, 2 = SURF, 3 = ORB
    : desc1(desc1_in), desc2(desc2_in), threshold(threshold_in),
      kpts1(kpts1_in), kpts2(kpts2_in), flann(flann_in), descType(descType_in)
    {
        compute();
    }

    void Homography::compute() {

//        std::cout << "\nHomography::compute() method runs\n";

        cv::Ptr<cv::DescriptorMatcher> matcher;

        if (getFlann()) {  //Flann based matcher
            std::cout << "Lsh indexing doesnt work in Flann. Using Brite Force matcher. \n";
        }
//            //      Flann compatibility conversion
//            if(desc1.type()!=CV_32F) {
//                desc1.convertTo(desc1, CV_32F);
//                desc2.convertTo(desc2, CV_32F);
//            }
//
//            if (getDescType() == 0 | getDescType() == 3){
//                //      ORB, BRIEF
//                std::cout << "Using Lsh Flann for binary descriptors \n";
//                cv::Ptr<cv::flann::IndexParams> flann_index_binary = new cv::flann::LshIndexParams(6, 12, 2);
//                matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher::FlannBasedMatcher(flann_index_binary));
//            }
//            else{
//                std::cout << "Using KDtree Flann for non-binary descriptors \n";
//                // SIFT, SUFR
//                cv::Ptr<cv::flann::IndexParams> flann_index_sift = new cv::flann::KDTreeIndexParams(5);
//                matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher::FlannBasedMatcher(flann_index_sift));
//            }
//
//        }
//        else{  // Bruteforce matcher
//            if (getDescType() == 0 | getDescType() == 3){
//                // ORB, BRIEF
//                std::cout << "Using HAMMING BFMATCHER for binary descriptors \n";
//                matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_HAMMING, 0));
//            }
//            else{
//                // SIFT, SURF
//                std::cout << "Using L2 BFMATCHER for non-binary descriptors \n";
//                matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_L2, 0));
//            }
//        }

        if (getDescType() == 0 | getDescType() == 3){
            // ORB, BRIEF
//            std::cout << "Using HAMMING BFMATCHER for binary descriptors \n";
            matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_HAMMING, 0));
        }
        else{
            // SIFT, SURF
//            std::cout << "Using L2 BFMATCHER for non-binary descriptors \n";
            matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_L2, 0));
        }

        matcher->match(desc1, desc2, matches);

        double min_dist = matches[0].distance, max_dist = 0;

        for( int i = 0; i < matches.size(); i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

//        std::cout << "min and max distance between matched descriptors: "
//                  << min_dist << " &  " << max_dist << "\n";

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
//        std::cout << "Matches = " << good_matches.size() << " inliers out of " << matches.size() << "\n";

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
    bool Homography::getFlann() {
        return this->flann;
    }
    int Homography::getDescType() {
        return this->descType;
    }


}