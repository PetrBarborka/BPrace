//
// Created by Tyler Durden on 3/18/16.
//

#ifndef BP_DESCRIPTION_H
#define BP_DESCRIPTION_H

namespace BP {

    class Description {

    private:
        const cv::Mat &src;
        std::vector<cv::KeyPoint> keypoints;
        std::string method;
        cv::Mat descriptors;

        void describe();

    public:
        Description(cv::Mat src_in, std::vector<cv::KeyPoint> kpoints_in, std::string method_in);
        std::vector<cv::KeyPoint> getKeypoints();
        cv::Mat getDescriptors();
        std::string getMethod();
        cv::Mat getSrc();
    };

}

#endif //BP_DESCRIPTION_H
