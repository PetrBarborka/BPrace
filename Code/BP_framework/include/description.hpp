//
// Created by Tyler Durden on 3/18/16.
//

#ifndef BP_DESCRIPTION_H
#define BP_DESCRIPTION_H

namespace BP {

    enum description_method { DESCRIPTION_BRIEF = 1,
                              DESCRIPTION_SIFT = 2,
                              DESCRIPTION_SURF = 3,
                              DESCRIPTION_ORB = 4};

    class Description {

    private:
        const cv::Mat &src;
        std::vector<cv::KeyPoint> keypoints;
        description_method method;
        cv::Mat descriptors;

        void describe();

    public:
        Description(cv::Mat src_in, std::vector<cv::KeyPoint> kpoints_in, description_method method_in);
        std::vector<cv::KeyPoint> getKeypoints();
        cv::Mat getDescriptors();
        description_method getMethod();
        cv::Mat getSrc();
    };

}

#endif //BP_DESCRIPTION_H
