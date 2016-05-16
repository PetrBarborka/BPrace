//
// Created by Tyler Durden on 4/25/16.
//

#ifndef CH
#define CH (char*)
#endif


#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "gtest/gtest.h"


#include "utility.hpp"
#include "process.hpp"

#include "boost/algorithm/string.hpp"

//json = nlohmann::json;

namespace {

// The fixture for testing class Foo.
    class FrameworkTest : public ::testing::Test {
    public:
        std::vector<std::string> detm = {   "Harris",
                                            "GFTT",
                                            "SIFT",
                                            "SURF",
                                            "FAST",
                                            "MSER",
                                            "ORB"};

        std::vector<std::string> descm = {"BRIEF", "SIFT", "SURF", "ORB"};

    };

    TEST_F(FrameworkTest, parseArgs){
        // just pics
        int argc = 3;
        char *argv[100] = {CH "dummypath", CH "--pjsn", CH "pics_minimal.json"};
        BP::jsons_t conf_files = BP::parseArgs(argc, argv);
        ASSERT_FALSE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());

        // just config
        argc = 3;
        argv[1]  = CH "--cjsn"; argv[2]  = CH "config.json";
        conf_files = BP::parseArgs(argc, argv);
        ASSERT_TRUE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());

        // just out
        argc = 3;
        argv[1] = CH "--out"; argv[2] = CH "tst_outconf.json";
        conf_files = BP::parseArgs(argc, argv);
        ASSERT_TRUE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());

        // all
        argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH "--out"; argv[6] = CH "tst_outconf.json";
        conf_files = BP::parseArgs(argc, argv);

        // all jsons loaded?
        ASSERT_FALSE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());

        // pictures json contents ok?
        json::iterator it = conf_files.pictures.begin();
        ASSERT_EQ(it.key(), "pair1");
        ASSERT_EQ(it.value()[0], "tstpics/pic11.png");
        ASSERT_EQ(it.value()[1], "tstpics/pic12.png");
        ASSERT_EQ(it.value()[2], "tstpics/H11to12");

        // config json contents ok?
        std::vector<std::string> dm = *(conf_files.config.find("description_methods"));
        for (int i = 0; i<4;i++){
            ASSERT_EQ(dm[i], descm[i]);
        }

        std::vector<std::string> dme = *(conf_files.config.find("detection_methods"));
        for (int i = 0; i<7;i++){
            ASSERT_EQ(dme[i], detm[i]);
        }

        ASSERT_EQ(*(conf_files.config.find("matchingThreshold")), "5");
        ASSERT_EQ(*(conf_files.config.find("maxPts")), "10000");
        ASSERT_EQ(*(conf_files.config.find("show")), "0");

        // output json contents ok?

        ASSERT_EQ(*(conf_files.output.find("csvpath")), "tstout/data.csv");
        ASSERT_EQ(*(conf_files.output.find("picspath")), "tstout/");

    }

    TEST_F(FrameworkTest, parseConfig){
        char * argv[7];
        int argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH"--out"; argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        ASSERT_FALSE(js.pictures.empty());
        ASSERT_FALSE(js.config.empty());
        ASSERT_FALSE(js.output.empty());

        BP::homography_t hg;
        std::vector<std::string> det_methods;
        std::vector<std::string> desc_methods;

        BP::parseConfig(hg, js.config,
                        det_methods, desc_methods);

        ASSERT_EQ(hg.maxPts, 10000);
        ASSERT_EQ(hg.show_pic, 0);
        for (int i = 0; i<detm.size(); i++){
            ASSERT_EQ(det_methods[i], detm[i]);
        }
        for (int i = 0; i<descm.size(); i++){
            ASSERT_EQ(desc_methods[i], descm[i]);
        }
    }

    TEST_F(FrameworkTest, initialize){

        char * argv[7];
        int argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH"--out"; argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        ASSERT_FALSE(js.pictures.empty());
        ASSERT_FALSE(js.config.empty());
        ASSERT_FALSE(js.output.empty());

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t hg = BP::initialize( js,
                                              pic_pairs,
                                              det_methods,
                                              desc_methods,
                                              ground_truths );


        ASSERT_EQ(pic_pairs[0][0], "tstpics/pic11.png");
        ASSERT_EQ(pic_pairs[0][1], "tstpics/pic12.png");
        ASSERT_EQ(ground_truths[0], "tstpics/H11to12");

        for (int i = 0; i<detm.size(); i++){
            ASSERT_EQ(det_methods[i], detm[i]);
        }
        for (int i = 0; i<descm.size(); i++){
            ASSERT_EQ(desc_methods[i], descm[i]);
        }

        ASSERT_EQ(hg.out_pic_path, "tstout/");
        ASSERT_EQ(hg.csv_path, "tstout/data.csv");
        ASSERT_FALSE(hg.show_pic);
        ASSERT_TRUE(hg.save_pic);
        ASSERT_TRUE(hg.save_csv);
        ASSERT_EQ(hg.maxPts, 10000);
    }

    TEST_F(FrameworkTest, detect){

        char * argv[7];
        int argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH"--out"; argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        ASSERT_FALSE(js.pictures.empty());
        ASSERT_FALSE(js.config.empty());
        ASSERT_FALSE(js.output.empty());

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t hg = BP::initialize( js,
                                              pic_pairs,
                                              det_methods,
                                              desc_methods,
                                              ground_truths );

        hg.det_method = det_methods[2];
        hg.desc_method = det_methods[3];

        hg.src1 = cv::imread(pic_pairs[0][0], cv::IMREAD_GRAYSCALE);
        hg.src2 = cv::imread(pic_pairs[0][1], cv::IMREAD_GRAYSCALE);
        hg.homography_gt = BP::readMatFromTextFile(ground_truths[0]);

        BP::detect(hg);

        ASSERT_GT(hg.src1.size().width, 0);
        ASSERT_GT(hg.src2.size().width, 0);
        ASSERT_GT(hg.homography_gt.size().width, 0);
        ASSERT_GT(hg.kpoints1.size(), 0);
        ASSERT_GT(hg.kpoints2.size(), 0);
        ASSERT_GT(hg.time_det, 0);
    }

    TEST_F(FrameworkTest, describe){

        char * argv[7];
        int argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH"--out"; argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t hg = BP::initialize( js,
                                              pic_pairs,
                                              det_methods,
                                              desc_methods,
                                              ground_truths );

        hg.det_method = det_methods[2];
        hg.desc_method = det_methods[3];

        hg.src1 = cv::imread(pic_pairs[0][0], cv::IMREAD_GRAYSCALE);
        hg.src2 = cv::imread(pic_pairs[0][1], cv::IMREAD_GRAYSCALE);
        hg.homography_gt = BP::readMatFromTextFile(ground_truths[0]);

        BP::detect(hg);
        BP::describe(hg);

        ASSERT_GT(hg.descriptors1.size().width, 0);
        ASSERT_GT(hg.descriptors2.size().width, 0);
        ASSERT_GT(hg.time_desc, 0);
    }

    TEST_F(FrameworkTest, match){

        char * argv[7];
        int argc = 7;
        argv[1]  = CH "--pjsn"; argv[2]  = CH "pics_minimal.json";
        argv[3]  = CH "--cjsn"; argv[4]  = CH "config.json";
        argv[5]  = CH "--out"; argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t hg = BP::initialize( js,
                                              pic_pairs,
                                              det_methods,
                                              desc_methods,
                                              ground_truths );

        hg.det_method = det_methods[2];
        hg.desc_method = det_methods[3];

        hg.src1 = cv::imread(pic_pairs[0][0], cv::IMREAD_GRAYSCALE);
        hg.src2 = cv::imread(pic_pairs[0][1], cv::IMREAD_GRAYSCALE);
        hg.homography_gt = BP::readMatFromTextFile(ground_truths[0]);

        BP::detect(hg);
        BP::describe(hg);
        BP::match(hg);

        ASSERT_GT(hg.matches.size(), 0);
    }

    TEST_F(FrameworkTest, computeHomography) {

        char *argv[7];
        int argc = 7;
        argv[1] = CH "--pjsn";
        argv[2] = CH "pics_minimal.json";
        argv[3] = CH "--cjsn";
        argv[4] = CH "config.json";
        argv[5] = CH "--out";
        argv[6] = CH "tst_outconf.json";
        BP::jsons_t js = BP::parseArgs(argc, argv);

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t hg = BP::initialize(js,
                                             pic_pairs,
                                             det_methods,
                                             desc_methods,
                                             ground_truths);

        hg.det_method = det_methods[2];
        hg.desc_method = det_methods[3];

        hg.src1 = cv::imread(pic_pairs[0][0], cv::IMREAD_GRAYSCALE);
        hg.src2 = cv::imread(pic_pairs[0][1], cv::IMREAD_GRAYSCALE);
        hg.homography_gt = BP::readMatFromTextFile(ground_truths[0]);

        BP::detect(hg);
        BP::describe(hg);
        BP::match(hg);
        BP::computeHomography(hg, pic_pairs[0][0], pic_pairs[0][1]);

        ASSERT_GT(hg.homography.size().width, 0);
        ASSERT_GT(hg.mask.rows, 0);
        ASSERT_GT(hg.matches.size(), 0);
        ASSERT_GT(hg.good_matches.size(), 0);
        ASSERT_GT(hg.hmg_distance, 0);

        ASSERT_EQ(hg.label, "pic11_pic12_det_SIFT_desc_SURF");
        ASSERT_EQ(hg.filename, "tstout/pic11_pic12_det_SIFT_desc_SURF.png");
        ASSERT_GT(hg.outPic.rows, 0);

        ASSERT_GT(hg.time_homography, 0);

        std::vector<std::string> cr;
        boost::split(cr,hg.csv_row, boost::is_any_of(", \n"), boost::token_compress_on);
        std::vector<std::string> tv =  {"tstout/", "tstpics/pic11.png",  "tstpics/pic12.png",
                                         "SIFT", "SURF", "36", "18", "3873.96", "0.03926",
                                         "0.009957", "0.004832",
                                         "tstout/pic11_pic12_det_SIFT_desc_SURF.png"};


        cr.resize(12);

        std::vector<std::string> tvec;

        for (int a=0;a<10;a++){
            tvec.push_back("aaaa");

        }

        std::vector<std::string> tvec2 = {"aaaa", "aaaa", "aaaa", "aaaa"};

        for (int i = 0; i<cr.size(); i++){
//            std::cout << "[i: " << i << "  ] " << (cr[i] == tv[i]) << "\n"
//                      << "cr: " << cr[i]
//                      << "\ntv: " << tv[i] << "\n";
            if (i<8 || i>10){
                ASSERT_EQ(cr[i], tv[i]);
            }
            else{
                ASSERT_GT(std::stod(cr[i]), 0);
            }
        }

    }

    TEST_F(FrameworkTest, computeHGS){
        /* Test Homography computation on minimal examples
         *
         */
        char *argv[7];
        int argc = 7;
        argv[1] = CH "--pjsn";
        argv[2] = CH "pics_minimal.json";
        argv[3] = CH "--cjsn";
        argv[4] = CH "config.json";
        argv[5] = CH "--out";
        argv[6] = CH "tst_outconf.json";

        BP::jsons_t js = BP::parseArgs(argc, argv);

        std::vector<std::vector<std::string>> pic_pairs;
        std::vector<std::string> det_methods, desc_methods, ground_truths;

        BP::homography_t init_hg = BP::initialize(js,
                                                  pic_pairs,
                                                  det_methods,
                                                  desc_methods,
                                                  ground_truths);

        std::vector<BP::homography_t> hgs;

        for (int p = 0; p<pic_pairs.size(); p++){
            for (int dt = 0; dt<det_methods.size(); dt++){
                for (int dc = 0; dc<desc_methods.size(); dc++){
                    BP::homography_t hg = init_hg;
                    hg.det_method = det_methods[dt];
                    hg.desc_method = desc_methods[dc];

                    hg.src1 = cv::imread(pic_pairs[p][0], cv::IMREAD_GRAYSCALE);
                    hg.src2 = cv::imread(pic_pairs[p][1], cv::IMREAD_GRAYSCALE);
                    hg.homography_gt = BP::readMatFromTextFile(ground_truths[p]);

                    BP::detect(hg);
                    BP::describe(hg);
                    BP::match(hg);
                    BP::computeHomography(hg, pic_pairs[p][0], pic_pairs[p][1]);

                    hgs.push_back(hg);
                }
            }
        }

//      7x4 methods:
        ASSERT_EQ(hgs.size(), 28);

        // for every tested instance:
        for (int i = 0; i<hgs.size(); i++){
            // pics are loaded
            BP::homography_t h = hgs[i];
            ASSERT_FALSE(h.src1.empty());
            ASSERT_FALSE(h.src2.empty());
            ASSERT_FALSE(h.kpoints1.empty());
            ASSERT_FALSE(h.kpoints2.empty());
            ASSERT_FALSE(h.matches.empty());
            ASSERT_FALSE(h.good_matches.empty());
            ASSERT_FALSE(h.homography_gt.empty());
            ASSERT_EQ(h.maxPts, 10000);
            ASSERT_EQ(h.show_pic, 0);
            ASSERT_GT(h.time_homography, 0);
            ASSERT_GT(h.time_desc, 0);
            ASSERT_GT(h.time_det, 0);

        }
    }

    TEST_F(FrameworkTest, getLabel) {

        BP::homography_t hg;
        hg.det_method = "MSER";
        hg.desc_method = "ORB";
        std::string l = BP::getLabel("a", "b", hg);
        std::string s = "a_b_det_MSER_desc_ORB";
        const char* ll = l.c_str();
        const char* ss = s.c_str();
        ASSERT_STREQ(ll, ss);

        l = BP::getLabel("///xxxa/vb/abc.def", "//b//dxdz/dzd/def.abc", hg);
        l = "a_b_det_MSER_desc_ORB";
        const char* lll = l.c_str();
        const char* sss = s.c_str();
        ASSERT_STREQ(lll, sss);
    }

}  // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}