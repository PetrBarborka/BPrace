//
// Created by Tyler Durden on 4/25/16.
//


#include <iostream>
#include "gtest/gtest.h"

//#include "../../main.cpp"

#include "utility.hpp"

//json = nlohmann::json;

namespace {

// The fixture for testing class Foo.
    class FrameworkTest : public ::testing::Test {
//    protected:
//        // You can remove any or all of the following functions if its body
//        // is empty.
//
//        FooTest() {
//            // You can do set-up work for each test here.
//        }
//
//        virtual ~FooTest() {
//            // You can do clean-up work that doesn't throw exceptions here.
//        }
//
//        // If the constructor and destructor are not enough for setting up
//        // and cleaning up each test, you can define the following methods:
//
//        virtual void SetUp() {
//            // Code here will be called immediately after the constructor (right
//            // before each test).
//        }
//
//        virtual void TearDown() {
//            // Code here will be called immediately after each test (right
//            // before the destructor).
//        }
//
//        // Objects declared here can be used by all tests in the test case for Foo.
    };

//// Tests that the Foo::Bar() method does Abc.
//    TEST_F(FooTest, MethodBarDoesAbc) {
//    const std::string input_filepath = "this/package/testdata/myinputfile.dat";
//    const std::string output_filepath = "this/package/testdata/myoutputfile.dat";
//    Foo f;
//    EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
//}


    // Tests that Foo does Xyz.
//    TEST_F(FrameworkTest, DoesXyz) {
//    // Exercises the Xyz feature of Foo.
//        EXPECT_EQ(0, 1);
//    }

//    TEST_F(FrameworkTest, IoTest) {
////        int argc = 7;
////        char *argv[] = {"dummypath", "--cjsn", "config.json", "--pjsn",
////                        "pics_minimal.json", "--out", "tst_outconf.json"};
//        int argc = 2;
//        char * argv[] = {"dummypath", "--help"};
//        main(argc, argv);
//    }

    TEST_F(FrameworkTest, parseArgs){
        // just pics
        int argc = 3;
        char *argv[100] = {"dummypath", "--pjsn", "pics_minimal.json"};
        BP::jsons_t conf_files = BP::parseArgs(argc, argv);
        ASSERT_FALSE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());
        // just config
        argc = 3;
        argv[1]  = "--cjsn"; argv[2]  = "config.json";
        conf_files = BP::parseArgs(argc, argv);
        ASSERT_TRUE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());
        // just out
        argc = 3;
        argv[1] = "--out"; argv[2] = "tst_outconf.json";
        conf_files = BP::parseArgs(argc, argv);
        ASSERT_TRUE(conf_files.pictures.empty());
        ASSERT_FALSE(conf_files.config.empty());
        ASSERT_FALSE(conf_files.output.empty());
        // all
        argc = 7;
        argv[1]  = "--pjsn"; argv[2]  = "pics_minimal.json";
        argv[3]  = "--cjsn"; argv[4]  = "config.json";
        argv[5] = "--out"; argv[6] = "tst_outconf.json";
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
            ASSERT_EQ(dm[i], std::to_string(i));
        }

        std::vector<std::string> dme = *(conf_files.config.find("detection_methods"));
        for (int i = 0; i<7;i++){
            ASSERT_EQ(dme[i], std::to_string(i));
        }

        ASSERT_EQ(*(conf_files.config.find("matchingThreshold")), "5");
        ASSERT_EQ(*(conf_files.config.find("maxPts")), "10000");
        ASSERT_EQ(*(conf_files.config.find("show")), "0");

        // output json contents ok?

        ASSERT_EQ(*(conf_files.output.find("csvpath")), "tstout/data.csv");
        ASSERT_EQ(*(conf_files.output.find("picspath")), "tstout/");

//        for(json::iterator it = conf_files.output.begin(); it != conf_files.output.end(); it++) {
//            std::cout << it.key() << ": " << it.value() << "\n";
//////            ASSERT_EQ(it.key(), "pair1");
//////            ASSERT_EQ(it.value()[0], "tstpics/pic11.png");
//////            ASSERT_EQ(it.value()[1], "tstpics/pic12.png");
//////            ASSERT_EQ(it.value()[2], "tstpics/H11to12");
//        }

    }

    TEST_F(FrameworkTest, computeHGS){
        /* Test Homography computation on minimal examples
         *
         */
        int argc = 7;
        char *argv[100] = {"dummypath", "--pjsn", "pics_minimal.json",
                                        "--cjsn", "config.json",
                                        "--out", "tst_outconf.json"};
        BP::jsons_t conf_files = BP::parseArgs(argc, argv);

        std::vector<BP::homography_t> hgs;

        BP::computeAllHGs(hgs, conf_files.pictures, conf_files.config, conf_files.output);

//        std::cout << hgs.size();

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
            ASSERT_EQ(h.show, 0);
            ASSERT_GT(h.time_homography, 0);
            ASSERT_GT(h.time_desc, 0);
            ASSERT_GT(h.time_det, 0);

//            for (int j = 0; j < )
        }
    }

    TEST_F(FrameworkTest, ptrtst) {
        float ff = 20.2;
        cv::Ptr<float> f = cv::makePtr<float>(ff);
        ASSERT_FLOAT_EQ(*f,20.2);
    }


}  // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}