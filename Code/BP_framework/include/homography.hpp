//
// Created by Tyler Durden on 3/18/16.
//

#ifndef BP_HOMOGRAPHY_H
#define BP_HOMOGRAPHY_H

namespace BP {

    class Homography {
        private:
            // how many times the minimal distance is still a good match?
            float threshold;
            cv::Mat desc1;
            cv::Mat desc2;
            std::vector<cv::DMatch> matches, good_matches;
            std::vector<cv::KeyPoint> kpts1, kpts2;
            cv::Mat hmgr;
            void compute();
        public:
            Homography( cv::Mat desc1_in,
                        cv::Mat desc2_in,
                        std::vector<cv::KeyPoint> kpts1_in,
                        std::vector<cv::KeyPoint> kpts2_in,
                        float threshold_in);

            float getThreshold();
            std::vector<cv::KeyPoint> getDesc1();
            std::vector<cv::KeyPoint> getDesc2();
            std::vector<cv::DMatch> getMatches();
            std::vector<cv::DMatch> getGoodMatches();
            std::vector<cv::KeyPoint> getKpts1();
            std::vector<cv::KeyPoint> getKpts2();
            cv::Mat getHomography();

    };

}

#endif //BP_HOMOGRAPHY_H
