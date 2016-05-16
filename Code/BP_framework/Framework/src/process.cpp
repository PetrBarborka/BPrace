//
// Created by Tyler Durden on 5/16/16.
//

#include "process.hpp"
#include "boost/algorithm/string.hpp"
#include "opencv2/highgui.hpp"

namespace BP {

    homography_t initialize(const jsons_t &js,
                            std::vector<std::vector<std::string>> &pic_pairs,
                            std::vector<std::string> &det_methods,
                            std::vector<std::string> &desc_methods,
                            std::vector<std::string> &ground_truths) {

        homography_t hg;
        parseConfig(hg, js.config, det_methods, desc_methods);

//         check for pics validity
        if (js.pictures.begin()++ == js.pictures.end()) {
            std::cout << "No pictures to test. --help for usage.\n";
            throw std::invalid_argument("no pics to test on");
        } else {
            for (json::const_iterator it = js.pictures.begin(); it != js.pictures.end(); it++) {
                // load ground truth
                if (it.value().size() > 2) {
                    ground_truths.push_back(it.value()[2]);
                }
                else {
                    std::cout << "Homography ground truth unavailable\n";
                }
                pic_pairs.push_back({it.value()[0], it.value()[1]});
            }
        }

        std::string pp = jsonGetValue<std::string>(js.output, "picspath");
        if (boost::iequals(pp, "")) {
            hg.save_pic = 0;
        } else {
            hg.save_pic = 1;
            hg.out_pic_path = pp;
        }

        std::string cp = jsonGetValue<std::string>(js.output, "csvpath");
        if (boost::iequals(cp, "")) {
            hg.save_csv = 0;
        } else {
            hg.save_csv = 1;
            hg.csv_path = cp;
        }

        return hg;
    }

    void detect(homography_t &hg) {
        std::clock_t begin = std::clock();

        Detection det1(hg.src1, hg.det_method, hg.maxPts);
        Detection det2(hg.src2, hg.det_method, hg.maxPts);
        hg.kpoints1 = det1.getKeypoints();
        hg.kpoints2 = det2.getKeypoints();

        // if description is SIFT, but detection is not SIFT, we need to unpack
        // and patch octave
        if (hg.det_method == "SIFT" & hg.desc_method != "SIFT") {
            patchSIFTOctaves(hg.kpoints1);
            patchSIFTOctaves(hg.kpoints2);
        }
        std::clock_t end = std::clock();
        hg.time_det = double(end - begin) / CLOCKS_PER_SEC;

        if (hg.kpoints1.empty()) {
            std::cout << "ERROR: no keypoints found on pic 1\n";
        }
        if (hg.kpoints2.empty()) {
            std::cout << "ERROR: no keypoints found on pic 2\n";
        }
    }

    void describe(homography_t &hg) {
        std::clock_t begin = std::clock();
        Description desc1(hg.src1, hg.kpoints1, hg.desc_method);
        Description desc2(hg.src2, hg.kpoints2, hg.desc_method);
        hg.descriptors1 = desc1.getDescriptors();
        hg.descriptors2 = desc2.getDescriptors();
        std::clock_t end = std::clock();
        hg.time_desc = double(end - begin) / CLOCKS_PER_SEC;

        if (hg.descriptors1.empty()) {
            std::cout << "ERROR: descriptors 1 empty\n";
        }
        if (hg.descriptors2.empty()) {
            std::cout << "ERROR: descriptors 2 empty\n";
        }
    }

    void match(homography_t &hg) {
        cv::Ptr<cv::DescriptorMatcher> matcher;

        if (hg.desc_method == "ORB" | hg.desc_method == "BRIEF") {
            // ORB, BRIEF - HAMMING distance for binary descriptors
            matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_HAMMING, 0));
        }
        else {
            // SIFT, SURF - L2 distance for non binary descriptors
            matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher::BFMatcher(cv::NORM_L2, 0));
        }

        matcher->match(hg.descriptors1, hg.descriptors2, hg.matches);

        if (hg.matches.empty()) {
            std::cout << "ERROR: got no matches!\n";
        }

        double min_dist = hg.matches[0].distance, max_dist = 0;
    }

    void computeHomography(homography_t &hg, std::string p1, std::string p2) {

        std::time_t begin = std::clock();
        Homography hmg(hg.descriptors1, hg.descriptors2,
                       hg.kpoints1, hg.kpoints2,
                       hg.desc_method, hg.homography_gt,
                       hg.matches);

        std::time_t end = std::clock();

        hg.time_homography = double(end - begin) / CLOCKS_PER_SEC;
        hg.homography = hmg.getHomography();
        hg.matches = hmg.getMatches();
        hg.good_matches = hmg.getGoodMatches();
        hg.mask = hmg.getMask();
        hg.hmg_distance = hmg.getHmgDistance();

        hg.label = getLabel(p1, p2, hg);
        hg.filename = hg.out_pic_path + hg.label + ".png";
        hg.csv_row = getCsvRow(hg, p1, p2);

        hg.outPic = getMatchesImg(hg);
    }

    void save(homography_t &hg) {
        if (hg.save_pic) {
            try {
                hg.filename = hg.out_pic_path + hg.label + ".png";
                cv::imwrite(hg.filename, hg.outPic);
            }
            catch (std::exception e) {
                std::cout << e.what();
            }
        }
    }

    void show(homography_t &hg) {
        if (hg.show_pic) {
            cv::imshow(hg.label, hg.outPic);
        }
    }

} //namespace BP