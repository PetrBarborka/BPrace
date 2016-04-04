//
// Created by Tyler Durden on 3/18/16.
//

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <fstream>

#include "boost/algorithm/string.hpp"

#include "utility.hpp"

namespace BP
{
    //VISUAL - DRAWING ============================================================
    void showKeypoints(cv::InputArray &in_mat, std::vector<cv::KeyPoint> &kpts, std::string winname)
    {
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
    void unpackSIFTOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale){
        octave = kpt.octave & 255;
        layer = (kpt.octave >> 8) & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
    }
    void patchSIFTOctaves(std::vector<cv::KeyPoint> &kpv){
        std::vector<cv::KeyPoint>::iterator it = kpv.begin();
        std::vector<cv::KeyPoint>::const_iterator ite = kpv.end();
        for(; it < ite; it++){
            int octave, layer; float scale;
            unpackSIFTOctave((*it), octave, layer, scale);
            (*it).octave = octave + 1;
        }
    }

    void PrintKeyPoint(const cv::KeyPoint &kp)
    {
        std::cout << "--\nKeypoint: " << kp.pt <<
            " size: " << kp.size <<
            " angle: " << kp.angle <<
            " response: " << kp.response <<
            " octave: " << kp.octave <<
            " class_id: " << kp.class_id << std::endl;
    }
    void PrintKPVector(const std::vector<cv::KeyPoint> &kpv, int max)
    {
        if (kpv.empty())
        {
            std::cout << "PrintKPV error: input vector empty!\n";
        }
        std::vector<cv::KeyPoint>::const_iterator it = kpv.begin();
        std::vector<cv::KeyPoint>::const_iterator ite = kpv.end();
        int ctr = 0;
        for(; it < ite; it++)
        {
            if (max > 0 && ctr >= max)
            {
                break;
            }
            PrintKeyPoint(*it);
            ctr++;
        }
    }
    void PrintMatch(const cv::DMatch &match)
    {
        std::cout << "--\nDMatch: " <<
                " distance: " << match.distance <<
                " imgIdx: " << match.imgIdx <<
                " queryIdx: " << match.queryIdx <<
                " trainIdx: " << match.trainIdx << "\n";
    }
    void PrintMatchVector(const std::vector<cv::DMatch> &mv, int max)
    {
        if (mv.empty()){
            std::cout << "PrintMatchVector error: input vector empty!\n";
        }
        std::vector<cv::DMatch>::const_iterator it = mv.begin();
        std::vector<cv::DMatch>::const_iterator ite = mv.end();
        int ctr = 0;
        for(; it < ite; it++){
            if (max > 0 && ctr >= max){
                break;
            }
            PrintMatch(*it);
            ctr++;
        }
    }
    bool compareKeypointsByResponse (const cv::KeyPoint &k1, const cv::KeyPoint &k2)
    {
        return k1.response > k2.response;
    }
    bool compareMatchesByDistance(const cv::DMatch &m1, const cv::DMatch &m2)
    {
        return m1.distance < m2.distance;
    }

    cv::Mat readMatFromTextFile(const std::string & path)
    {
        cv::Mat out;
        std::ifstream f(path);
        std::string mtrx((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        std::vector<std::string> rows;
        boost::split(rows, mtrx, boost::is_any_of("\n"));
        int r_count = 0,  c_count = 0;
        for (std::vector<std::string>::iterator ritr = rows.begin(); ritr < rows.end(); ritr++ ) {
            if ((*ritr).size() > 0)
            {
                r_count++;
            }
        }
        std::vector<std::string> col;
        boost::split(col, rows[0], boost::is_any_of(" ,;\t"));
        for (std::vector<std::string>::iterator citr = col.begin(); citr < col.end(); citr++ ) {
            if ((*citr).size() > 0)
            {
                c_count++;
            }
        }
        out.create(r_count, c_count, CV_32F);
        int rindex = 0;
        std::vector<std::string>::iterator ritr = rows.begin();
        cv::MatIterator_<float> mitr = out.begin<float>();
        for (; ritr < rows.end(); ritr++){
            boost::split(col, *ritr, boost::is_any_of(" "));
            int cindex = 0;
//            std::cout << "MATRIX ROW: \n";
            for (std::vector<std::string>::iterator citr = col.begin(); citr < col.end(); citr++ ) {
//                out.at(rindex).at(cindex) = stof(*citr);
                if ((*citr).size() > 0)
                {
//                    std::cout << "Individual matrix cell: [" << stof(*citr) << "]\n";
                    *(mitr) = stof(*citr);
//                    std::cout << "mitr: " << *(mitr) << "\n";
                    mitr++;
                }
                cindex++;
            }
            rindex++;
        }
        return out;
    }
}