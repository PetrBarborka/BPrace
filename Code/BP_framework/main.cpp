#include <QCoreApplication>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include <regex>

#include "utility.hpp"
#include "detection.hpp"
#include "description.hpp"
#include "homography.hpp"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/variables_map.hpp>
#include "boost/program_options/parsers.hpp"
#include "boost/algorithm/string.hpp"

#include <ctime>

#include "json.hpp"

#include <exception>

using json = nlohmann::json;

using namespace BP;

namespace po = boost::program_options;

struct homography_t;
void parseConfig(const json & config, homography_t &hg);
double getHomographyDistance(const cv::Mat & hmg1, const cv::Mat & hmg2);
void computeHg(homography_t &hg, int detIdx = 0, int descIdx = 0);
void computeAllHGs( std::vector<homography_t>& hgs, json pictures, json config, json output);

struct homography_t {
    //  source images
    cv::Mat src1, src2;
    //  image key points
    std::vector<cv::KeyPoint> kpoints1, kpoints2;
    //  descriptors of keypoints
    cv::Mat descriptors1, descriptors2;
    //  keypoint matches between images
    std::vector<cv::DMatch> matches, good_matches;
    //  Homography matrix
    cv::Mat homography, homography_gt;
    //  Multiple of minimal descriptor distance still
    //  considered a good match
    float matchingThreshold;
    //  Maximum number of keypoints returned
    int maxPts;
    std::vector<detection_method> det_methods;
    std::vector<description_method> desc_methods;
    // draw results?
    bool show;
    // inlier matches indicator
    cv::Mat mask;
    // execution times
    double time_desc, time_det, time_homography;
};
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

void computeHg(homography_t &hg, int detIdx, int descIdx){

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
                   hg.kpoints1, hg.kpoints2, hg.matchingThreshold);
    end = std::clock();
    hg.time_homography = double(end-begin)/CLOCKS_PER_SEC;
    hg.homography = hmg.getHomography();
    hg.matches = hmg.getMatches();
    hg.good_matches = hmg.getGoodMatches();
    hg.mask = hmg.getMask();

    std::cout << "Got " << hg.matches.size() << " matches, " << hg.good_matches.size() << " good ones.\n";

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
            std::cout << "opening  file w/ filename: " << csv_output_path << "\n";
        }
        catch (char const *e)
        {
            std::cout << e;
        }
    }

    for(json::iterator it = pictures.begin(); it != pictures.end(); it++)
    {
        std::cout << "=========================\n";
        std::cout << it.key() << ": picture1: " << it.value()[0] << " picture2: " << it.value()[1] << "\n";

        // load ground truth
        if (it.value().size() > 2) {
            try
            {
                hg.homography_gt = readMatFromTextFile(it.value()[2]);
                std::cout << "homography ground truth: \n" << hg.homography_gt;
            }
            catch (std::exception e)
            {
                std::cout << "Ground truth matrix parsing failed with exception: " << e.what();
            }
        }

        hg.src1 = cv::imread(it.value()[0], cv::IMREAD_GRAYSCALE);
        hg.src2 = cv::imread(it.value()[1], cv::IMREAD_GRAYSCALE);

        if (!hg.src1.data)
        {
            std::cout << "ERROR: Failed to load image " << it.key();
        }

        if (!hg.src2.data)
        {
            std::cout << "ERROR: Failed to load image " << it.value();
        }

        parseConfig(config, hg);
        for (int detIdx = 0; detIdx < hg.det_methods.size(); detIdx++)
        {
            for (int descIdx = 0; descIdx < hg.desc_methods.size(); descIdx++)
            {

                computeHg(hg, detIdx, descIdx);

                double hmg_distance;

                std::cout << "homography: \n" << hg.homography << "\n\n";
                if (hg.homography_gt.size().width > 0)
                {
                    std::cout << "homography ground truth: \n" << hg.homography_gt << "\n\n";
                    hmg_distance = getHomographyDistance(hg.homography, hg.homography_gt);
                    std::cout << "homography distance: " << hmg_distance << "\n";
                }
                else
                {
                    std::cout << "WARNING: Homography ground truth unavailable";
                }

                std::cout   << "det time: " << hg.time_det
                << " desc time: " << hg.time_desc
                << " hmg time: " << hg.time_homography << "\n";

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
                        if (!cv::imwrite(filename, img_matches))
                        {
                            throw std::string("imwrite failed, filename: ") + filename.c_str();
                        }
                    }
                    catch (char const *e)
                    {
                        std::cout << e;
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
                    catch (char const *e)
                    {
                        std::cout << e;
                        break;
                    }
                }
                else
                {
                    std::cout << "Pics path empty, not writing png file.\n";
                }
            }
        }
    }
    csv_out.close();

}

int main(int argc, char *argv[])
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("pjsn", po::value< std::string > (), "load picture list from json")
        ("cjsn", po::value< std::string > (), "load config json")
        ("out", po::value< std::string > (), "parse .json output config for saving and displaying")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    json pictures, config, output;

//    --help option
    if ( vm.count("help")  )
    {
        std::cout << "Basic Command Line Parameter App" << std::endl
        << desc << std::endl;
        return 0;
    }

//    --picture pairs in json
    if ( vm.count("pjsn")  )
    {
        try
        {
            std::string pjsn_path = vm["pjsn"].as<std::string>();
            std::ifstream f(pjsn_path);
            std::string str((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
            pictures = json::parse(str);
        }
        catch (std::exception e)
        {
            std::cout << "picture json loading failed with exception: " <<  e.what() << "\n";
        }
    }
    else
    {
        pictures = {{"graf1.png", "graf3.png"}};
    }

//    --config in json
    if ( vm.count("cjsn")  )
    {
        try
        {
            std::string  config_path = vm["cjsn"].as<std::string>();
            std::ifstream f(config_path);
            if (!f)
            {
                std::cout << "cjsn file doesn't exist!\n";
                throw std::invalid_argument("file does not exist");
            }
            std::string str((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
            config = json::parse(str);
        }
        catch (std::exception e)
        {
            std::cout << "config json loading failed with exception: " << e.what() << "\n";
        }
    }
    else
    {
        config = R"( {"matchingThreshold": "3",
                    "maxPts": "10000",
                    "detection_methods": ["2"] ,
                    "description_methods": ["1"] ,
                    "show": "1"} )"_json;
    }

//    --out in json
    if ( vm.count("out")  )
    {
        try
        {
            std::string  config_path = vm["out"].as<std::string>();
            std::ifstream f(config_path);
            if (!f)
            {
                std::cout << "out file doesn't exist!\n";
                throw std::invalid_argument("file does not exist");
            }
            std::string str((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
            output = json::parse(str);
        }
        catch (std::exception e)
        {
            std::cout << "output json loading failed with exception: " << e.what() << "\n";
        }
    }
    else
    {
        output = R"( {"picspath" : "",
                      "csvpath": "" } )"_json;
    }

    std::vector<homography_t> hgs;

    computeAllHGs(hgs, pictures, config, output);

//    std::cout << "trying to read matrix from file\n";
//
//    cv::Mat test = readMatFromTextFile("ENSIMAG/H0to1");
//
//    std::cout << "Test of reading matrix from textfile:\n"
//              << test;

    cv::waitKey(0);
    return 0;
}















