#pragma once

#include "ImageOperations.hpp"
#include "OrtOperations.hpp"

// Cross-platform export macro
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

class EXPORT Masking
{
public:
    static void mask_and_display(int label, std::string model_path, std::string image_path, int sharpening_strength);
    static void mask_and_display(int label, std::string model_path, cv::Mat image, int sharpening_strength);
    static cv::Mat final_mask(int label, std::string model_path, std::string image_path, int sharpening_strength);
    static cv::Mat final_mask(int label, std::string model_path, cv::Mat image, int sharpening_strength);
};
