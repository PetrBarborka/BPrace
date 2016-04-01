//
// Created by Tyler Durden on 3/18/16.
//

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include "utility.hpp"

namespace BP {

    //VISUAL - DRAWING ============================================================
    void showKeypoints(cv::InputArray &in_mat, std::vector<cv::KeyPoint> &kpts, std::string winname){
    cv::Mat out_mat;
    in_mat.copyTo(out_mat);
    cv::drawKeypoints(in_mat, kpts, out_mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
    cv::imshow(winname, out_mat);
}

    //MISC - UTILITIES ============================================================
    void pointsToKeypoints(const std::vector<cv::Point2f> &in_vec, std::vector<cv::KeyPoint> &out_vec){
    std::vector<cv::Point2f>::const_iterator itb = in_vec.begin();
    std::vector<cv::Point2f>::const_iterator ite = in_vec.end();
    for (; itb < ite; itb++) {
//        std::cout << "pTK() is pushing " << *itb << std::endl;
        out_vec.push_back(cv::KeyPoint(*itb, 30., 180., 1000., 0, -1));
    }
}
    void topKeypoints(std::vector<cv::KeyPoint> &pts, int ammount){
 int remove = pts.size() - ammount;
 for (; remove > 0; remove --){
    pts.pop_back();
 }
}

    void PrintKeyPoint(const cv::KeyPoint &kp){
        std::cout << "--\nKeypoint: " << kp.pt <<
            " size: " << kp.size <<
            " angle: " << kp.angle <<
            " response: " << kp.response <<
            " octave: " << kp.octave <<
            " class_id: " << kp.class_id << std::endl;
}
    void PrintKPVector(const std::vector<cv::KeyPoint> &kpv){

    if (kpv.empty()){
        std::cout << "PrintKPV error: input vector empty!\n";
    }
    std::vector<cv::KeyPoint>::const_iterator it = kpv.begin();
    std::vector<cv::KeyPoint>::const_iterator ite = kpv.end();
    for(; it < ite; it++){
        PrintKeyPoint(*it);
    }
}
    void PrintMatch(const cv::DMatch &match){
        std::cout << "--\nDMatch: " <<
                " distance: " << match.distance <<
                " imgIdx: " << match.imgIdx <<
                " queryIdx: " << match.queryIdx <<
                " trainIdx: " << match.trainIdx << "\n";
    }
    void PrintMatchVector(const std::vector<cv::DMatch> &mv){
        if (mv.empty()){
            std::cout << "PrintMatchVector error: input vector empty!\n";
        }
        std::vector<cv::DMatch>::const_iterator it = mv.begin();
        std::vector<cv::DMatch>::const_iterator ite = mv.end();
        for(; it < ite; it++){
            PrintMatch(*it);
        }
    }

    bool compareKeypointsByResponse (const cv::KeyPoint &k1, const cv::KeyPoint &k2){
        return k1.response > k2.response;
    }
    bool compareMatchesByDistance(const cv::DMatch &m1, const cv::DMatch &m2){
        return m1.distance < m2.distance;
    }
}