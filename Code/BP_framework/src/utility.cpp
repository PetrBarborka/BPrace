//
// Created by Tyler Durden on 3/18/16.
//


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <fstream>

#include "json.hpp"

#include "boost/algorithm/string.hpp"

#include "homography.hpp"
#include "utility.hpp"

#include "boost/program_options/options_description.hpp"
#include "boost/program_options/option.hpp"
#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/parsers.hpp"
#include "boost/filesystem.hpp"


using json = nlohmann::json;

namespace po = boost::program_options;

namespace fs = boost::filesystem;

namespace BP
{
    jsons_t parseArgs(int argc, char *argv[])
    {
        fs::path pwd = fs::system_complete(fs::current_path());

        // Declare the supported options.
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("pjsn", po::value< std::string > (), "load picture list from json: \n {label: [pic1path, pic2path [optional - homography matrix path]], label2: ... , ...}")
                ("cjsn", po::value< std::string > (), "load config json: \n {\t\"matchingThreshold\": \"5\",\n"
                        "\t\"maxPts\": \"10000\",\n"
                        "\t\"detection_methods\": [\"0\", \"1\", \"2\", \"3\", \"4\", \"5\", \"6\"] ,\n"
                        "\t\"description_methods\": [\"0\", \"1\", \"2\", \"3\"] ,\n"
                        "\t\"show\": \"0\" }")
                ("out", po::value< std::string > (), "parse .json output config for saving and displaying: \n {\"picspath\" : \"out/ensimag_complete_flann/\", \"csvpath\": \"out/ensimag_complete_flann/data.csv\" }")
                ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        jsons_t out;

//    --help option
        if ( vm.count("help")  )
        {
            std::cout << "Basic Command Line Parameter App" << std::endl
            << desc << "\n see the enclosed config generator and config examples"
            << std::endl;
            return out;
        }

//    --picture pairs in json
        if ( vm.count("pjsn")  )
        {
            std::string pjsn_path;
            try
            {
                pjsn_path = pwd.string() + "/" + vm["pjsn"].as<std::string>();
                std::ifstream f(pjsn_path);
                if (!f)
                {
                    std::cout << "pjsn file doesn't exist!\n";
                    throw std::invalid_argument("file does not exist");
                }
                std::string str((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
                out.pictures = json::parse(str);
            }
            catch (std::exception e)
            {
                std::cout << "picture " << pjsn_path << " loading failed with exception: " <<  e.what() << "\n"
                          << "in folder" << pwd << "\n";
            }
        }
        else
        {
//            out.pictures = {{"graf1.png", "graf3.png"}};
        }

//    --config in json
        if ( vm.count("cjsn")  )
        {
            std::string  config_path;
            try
            {
                config_path = pwd.string() + "/" + vm["cjsn"].as<std::string>();
                std::ifstream f(config_path);
                if (!f)
                {
                    std::cout << "cjsn file doesn't exist!\n";
                    throw std::invalid_argument("file does not exist");
                }
                std::string str((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
                out.config = json::parse(str);
            }
            catch (std::exception e)
            {
                std::cout   << "config " << config_path << " loading failed with exception: " << e.what() << "\n"
                            << "in folder" << pwd << "\n";
            }
        }
        else
        {
            out.config = R"( {"matchingThreshold": "3",
                    "maxPts": "10000",
                    "detection_methods": ["2"] ,
                    "description_methods": ["1"] ,
                    "show": "1"} )"_json;
        }

//    --out in json
        if ( vm.count("out")  )
        {
            std::string  output_path;
            try
            {
                output_path = pwd.string() + "/" + vm["out"].as<std::string>();
                std::ifstream f(output_path);
                if (!f)
                {
                    std::cout << "out file doesn't exist!\n";
                    throw std::invalid_argument("file does not exist");
                }
                std::string str((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
                out.output = json::parse(str);
            }
            catch (std::exception e)
            {
                std::cout   << "output " << output_path << " loading failed with exception: " << e.what() << "\n"
                            << "in folder" << pwd << "\n";
            }
        }
        else
        {
            out.output = R"( {"picspath" : "",
                      "csvpath": "" } )"_json;
        }
        return out;
    }

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

    void parseConfig(const json & config, homography_t &hg){
        std::string tmp;
        tmp = *(config.find("matchingThreshold"));
        hg.matchingThreshold = std::stof(tmp);
        tmp = *(config.find("maxPts"));
        hg.maxPts = std::stoi(tmp);
        tmp = *(config.find("show"));
        hg.show = (std::stoi(tmp) != 0);
        std::vector<detection_method> detection_methods;
        std::vector<description_method> description_methods;
        std::vector<std::string> tmp2 = *(config.find("detection_methods"));
        for (int i = 0; i < tmp2.size(); i++){
            detection_methods.push_back(detection_method(std::stoi(tmp2[i])));
        }
        std::vector<std::string> tmp3 = *(config.find("description_methods"));
        for (int i = 0; i < tmp3.size(); i++){
            description_methods.push_back(description_method(std::stoi(tmp3[i])));
        }
        hg.det_methods = detection_methods;
        hg.desc_methods = description_methods;

//    std::cout <<    "-------------------------\n" <<
//                    "Parsed config: " <<
//                    "matchingThreshold: " <<  hg.matchingThreshold << "\n"
//                <<  "Detection methods: [ ";
//    for (std::vector<detection_method>::iterator i = detection_methods.begin(); i != detection_methods.end(); i++){
//        std::cout << *i << ", ";
//    }
//    std::cout << "]\n Description methods: [ ";
//    for (std::vector<description_method>::iterator i = description_methods.begin(); i != description_methods.end(); i++){
//        std::cout << *i << ", ";
//    }
//    std::cout << "]\n";
//    std::cout <<    "maxPts: " << hg.maxPts << "\n" <<
//                    "show: " << hg.show <<
//                "\n-------------------------\n";
    }

    double getHomographyDistance(const cv::Mat & hmg1, const cv::Mat & hmg2){
        cv::Mat eigen1, eigen2;
        cv::eigen(hmg1, eigen1);
        cv::eigen(hmg2, eigen2);
        cv::Mat pi1 = hmg1*eigen1;
        cv::Mat pi2 = hmg2*eigen2;
//    std::cout << "pi 1 and 2:" << pi1 << " & " << pi2 << "\n";
        cv::Mat dif(pi1.size(), pi1.type());
        pi2.convertTo(pi2, CV_32F);
        pi1.convertTo(pi1, CV_32F);
        dif = pi2-pi1;
//    std::cout << "norm= " << cv::norm(dif) << "\n";

        return cv::norm(dif);
    }
}