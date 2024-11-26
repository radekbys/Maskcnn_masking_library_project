#pragma once
#include <opencv2/opencv.hpp>
#include <ranges>
#include <vector>
#include <iostream>

class ImageOperations
{
private:
    cv::Mat initial_image;
    int channels;
    int height;
    int width;

public:
    struct chw
    {
        int channels;
        int height;
        int width;
        std::vector<float> vec;
    };

    chw load_image(std::string image_path);
    chw load_image(cv::Mat image);
    std::vector<float> Get_chw_float_vector();
    void convert_to_0_1_range(std::vector<float> &vec, float max, float min);
    void Set_channels(int i);
    void convert_to_0_255_range_and_sharpen(std::vector<float> &vec, float max, float min, float sharpening_strength = 60);
    std::vector<float> convertToHWC(std::vector<float> &vec);
    cv::Mat convert_mask_to_proper_type(std::vector<float> &mask);
    std::vector<cv::Point> find_largest_contour(cv::Mat blurry_mask);
    cv::Mat calculate_final_mask(std::vector<cv::Point> contour);
    cv::Mat Get_initial_image();
};
