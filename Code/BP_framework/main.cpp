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
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/calib3d.hpp>

#include "utility.hpp"
#include "process.hpp"
#include "detection.hpp"
#include "description.hpp"
#include "homography.hpp"

//#include <boost/program_options/options_description.hpp>
//#include <boost/program_options/option.hpp>
//#include <boost/program_options/variables_map.hpp>
//#include "boost/program_options/parsers.hpp"
#include "boost/algorithm/string.hpp"

//#include <ctime>
#include "json.hpp"
//#include <exception>


using json = nlohmann::json;

using namespace BP;

int main(int argc, char *argv[]) {


    BP::jsons_t js = BP::parseArgs(argc, argv);
    if (js.pictures.empty()){
        std::cout << "No pictures to analyze" << "\n";
        return 100;
    }

    std::vector<std::vector<std::string>> pic_pairs;
    std::vector<std::string> det_methods, desc_methods, ground_truths;

    BP::homography_t init_hg = BP::initialize(js,
                                              pic_pairs,
                                              det_methods,
                                              desc_methods,
                                              ground_truths);


    std::ofstream csv_out;
    if (init_hg.save_csv){
        try {
            csv_out.open(init_hg.csv_path);
            //  std::cout << "opening  file w/ filename: " << csv_output_path << "\n";
        } catch (char const *e) {
            std::cout << e;
        }
    }

    for (int p = 0; p<pic_pairs.size(); p++){
        for (int dt = 0; dt<det_methods.size(); dt++){
            for (int dc = 0; dc<desc_methods.size(); dc++){
                BP::homography_t hg = init_hg;
                hg.det_method = det_methods[dt];
                hg.desc_method = desc_methods[dc];

                hg.src1 = cv::imread(pic_pairs[p][0], cv::IMREAD_GRAYSCALE);
                hg.src2 = cv::imread(pic_pairs[p][1], cv::IMREAD_GRAYSCALE);
                hg.homography_gt = BP::readMatFromTextFile(ground_truths[p]);

                detect(hg);
                describe(hg);
                match(hg);
                computeHomography(hg, pic_pairs[p][0], pic_pairs[p][1]);
                save(hg);
                show(hg);

                if (hg.save_csv){
                    csv_out << hg.csv_row;
                }
            }
        }
    }
    if (init_hg.save_csv) {
        csv_out.close();
    }
   if (init_hg.show_pic) {
        cv::waitKey(0);
   }
    return 0;
}
















