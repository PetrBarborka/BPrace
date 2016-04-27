//#include <QCoreApplication>

#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
//#include <streambuf>
//#include <string>
#include <regex>

#include "utility.hpp"
#include "detection.hpp"
#include "description.hpp"
#include "homography.hpp"

//#include <boost/program_options/options_description.hpp>
//#include <boost/program_options/option.hpp>
//#include <boost/program_options/variables_map.hpp>
//#include "boost/program_options/parsers.hpp"
//#include "boost/algorithm/string.hpp"

//#include <ctime>
#include "json.hpp"
//#include <exception>


using json = nlohmann::json;

using namespace BP;

//namespace BP {

int main(int argc, char *argv[]) {

//    std::cout << "argc: " << argc << "\n";
//    for (int i = 0; i < argc; i++) {
//        std::cout << "argv[" << i << "]: " << argv[i] << "\n";
//    }

    jsons_t conf_files = parseArgs(argc, argv);

//    std::cout << "parseArgs ok";

    if (conf_files.pictures.begin()++ == conf_files.pictures.end()) {
        std::cout << "No pictures to test. --help for usage.\n";
        return 0;
    }

    std::vector<homography_t> hgs;

    computeAllHGs(hgs, conf_files.pictures, conf_files.config, conf_files.output);

    if (hgs[0].show) {
        cv::waitKey(0);
    }
    return 0;
}

//} //namespace BP















