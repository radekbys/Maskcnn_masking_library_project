#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <dlfcn.h>

#include "Masking.hpp"

int main(int argc, char *argv[])
{
#define LABEL 3
#define MODEL_PATH "model.onnx"
#define IMAGE_PATH "car.png"
#define SHARPENING_STRENGTH 150

    // // tested, works fine
    // ImageOperations imageOperator;
    // imageOperator.load_image("car.png");
    // Masking::mask_and_display(LABEL, MODEL_PATH, imageOperator.Get_initial_image(), SHARPENING_STRENGTH);

    return 0;
}

// // tested, works fine
// ImageOperations imageOperator;
// imageOperator.load_image("car.png");
// Masking::mask_and_display(LABEL, MODEL_PATH, imageOperator.Get_initial_image(), SHARPENING_STRENGTH);

// // tested, works fine
// Masking::mask_and_display(LABEL, MODEL_PATH, IMAGE_PATH, SHARPENING_STRENGTH);

// // tested, works fine
// ImageOperations imageOperator;
// imageOperator.load_image("car.png");
// Masking::mask_and_display(LABEL, MODEL_PATH, imageOperator.Get_initial_image(), SHARPENING_STRENGTH);

// // tested works fine
// ImageOperations image_operator;
// image_operator.load_image(IMAGE_PATH);
// cv::Mat final_mask = Masking::final_mask(LABEL, MODEL_PATH, IMAGE_PATH, SHARPENING_STRENGTH);
// cv::Mat inverted_mask;
// cv::bitwise_not(final_mask, inverted_mask);
// cv::Mat image = image_operator.Get_initial_image();
// cv::Mat finalImage;
// cv::bitwise_and(image, image, finalImage, inverted_mask);
// cv::imshow("Final Image", finalImage);
// cv::waitKey(0);

// // tested works fine
// ImageOperations image_operator;
// image_operator.load_image(IMAGE_PATH);
// cv::Mat final_mask = Masking::final_mask(LABEL, MODEL_PATH, image_operator.Get_initial_image(), SHARPENING_STRENGTH);
// cv::Mat inverted_mask;
// cv::bitwise_not(final_mask, inverted_mask);
// cv::Mat image = image_operator.Get_initial_image();
// cv::Mat finalImage;
// cv::bitwise_and(image, image, finalImage, inverted_mask);
// cv::imshow("Final Image", finalImage);
// cv::waitKey(0);
