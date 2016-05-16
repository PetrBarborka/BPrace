//
// Created by Tyler Durden on 5/16/16.
//

#ifndef BP_PROCESS_H
#define BP_PROCESS_H


#include "homography.hpp"
#include "utility.hpp"

namespace BP {

    homography_t initialize(const jsons_t &c,
                            std::vector<std::vector<std::string>> &pic_pairs,
                            std::vector<std::string> &det_methods,
                            std::vector<std::string> &desc_methods,
                            std::vector<std::string> &ground_truths);
    void detect(homography_t &hg);
    void describe(homography_t &hg);
    void match(homography_t &hg);
    void computeHomography(homography_t &hg, std::string p1, std::string p2);
    void save(homography_t &hg);
    void show(homography_t &hg);

} //namespace BP


#endif //BP_PROCESS_H
