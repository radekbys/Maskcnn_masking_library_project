#include "Masking.hpp"

void Masking::mask_and_display(int label, std::string model_path, std::string image_path, int sharpening_strength)
{
    ImageOperations imageOperator;
    OrtOperations ortOperator(model_path);

    // load image and prepare data
    ImageOperations::chw dimensions = imageOperator.load_image(image_path);
    std::vector<float> hwc_float_image = imageOperator.Get_chw_float_vector();
    float maximum = *std::max_element(hwc_float_image.begin(), hwc_float_image.end());
    float minimum = *std::min_element(hwc_float_image.begin(), hwc_float_image.end());
    imageOperator.convert_to_0_1_range(hwc_float_image, maximum, minimum);

    // run onnx model and get the output
    ortOperator.LoadImage(dimensions, hwc_float_image);
    ortOperator.inference();
    std::vector<float> initial_mask = ortOperator.get_initial_mask(label);

    // now working on mask with one channel
    imageOperator.Set_channels(1);

    // Get the initial mask
    imageOperator.convert_to_0_255_range_and_sharpen(initial_mask, maximum, minimum, sharpening_strength);
    initial_mask = imageOperator.convertToHWC(initial_mask);
    cv::Mat preprocessed_blurry_mask = imageOperator.convert_mask_to_proper_type(initial_mask);

    // Calculate the final mask
    std::vector<cv::Point> contour = imageOperator.find_largest_contour(preprocessed_blurry_mask);
    cv::Mat final_mask = imageOperator.calculate_final_mask(contour);

    // invert the mask and mask the image
    cv::Mat inverted_mask;
    cv::bitwise_not(final_mask, inverted_mask);
    cv::Mat image = imageOperator.Get_initial_image();
    cv::Mat finalImage;
    cv::bitwise_and(image, image, finalImage, inverted_mask);

    // Display the result
    cv::imshow("Final Image", finalImage);
    cv::waitKey(0);
}

void Masking::mask_and_display(int label, std::string model_path, cv::Mat image, int sharpening_strength)
{
    ImageOperations imageOperator;
    OrtOperations ortOperator(model_path);

    // load image and prepare data
    ImageOperations::chw dimensions = imageOperator.load_image(image);
    std::vector<float> hwc_float_image = imageOperator.Get_chw_float_vector();
    float maximum = *std::max_element(hwc_float_image.begin(), hwc_float_image.end());
    float minimum = *std::min_element(hwc_float_image.begin(), hwc_float_image.end());
    imageOperator.convert_to_0_1_range(hwc_float_image, maximum, minimum);

    // run onnx model and get the output
    ortOperator.LoadImage(dimensions, hwc_float_image);
    ortOperator.inference();
    std::vector<float> initial_mask = ortOperator.get_initial_mask(label);

    // now working on mask with one channel
    imageOperator.Set_channels(1);

    // Get the initial mask
    imageOperator.convert_to_0_255_range_and_sharpen(initial_mask, maximum, minimum, sharpening_strength);
    initial_mask = imageOperator.convertToHWC(initial_mask);
    cv::Mat preprocessed_blurry_mask = imageOperator.convert_mask_to_proper_type(initial_mask);

    // Calculate the final mask
    std::vector<cv::Point> contour = imageOperator.find_largest_contour(preprocessed_blurry_mask);
    cv::Mat final_mask = imageOperator.calculate_final_mask(contour);

    // invert the mask and mask the image
    cv::Mat inverted_mask;
    cv::bitwise_not(final_mask, inverted_mask);
    cv::Mat initial_image = imageOperator.Get_initial_image();
    cv::Mat finalImage;
    cv::bitwise_and(initial_image, initial_image, finalImage, inverted_mask);

    // Display the result
    cv::imshow("Final Image", finalImage);
    cv::waitKey(0);
}

cv::Mat Masking::final_mask(int label, std::string model_path, std::string image_path, int sharpening_strength)
{
    ImageOperations imageOperator;
    OrtOperations ortOperator(model_path);

    // load image and prepare data
    ImageOperations::chw dimensions = imageOperator.load_image(image_path);
    std::vector<float> hwc_float_image = imageOperator.Get_chw_float_vector();
    float maximum = *std::max_element(hwc_float_image.begin(), hwc_float_image.end());
    float minimum = *std::min_element(hwc_float_image.begin(), hwc_float_image.end());
    imageOperator.convert_to_0_1_range(hwc_float_image, maximum, minimum);

    // run onnx model and get the output
    ortOperator.LoadImage(dimensions, hwc_float_image);
    ortOperator.inference();
    std::vector<float> initial_mask = ortOperator.get_initial_mask(label);

    // now working on mask with one channel
    imageOperator.Set_channels(1);

    // Get the initial mask
    imageOperator.convert_to_0_255_range_and_sharpen(initial_mask, maximum, minimum, sharpening_strength);
    initial_mask = imageOperator.convertToHWC(initial_mask);
    cv::Mat preprocessed_blurry_mask = imageOperator.convert_mask_to_proper_type(initial_mask);

    // Calculate the final mask
    std::vector<cv::Point> contour = imageOperator.find_largest_contour(preprocessed_blurry_mask);
    cv::Mat final_mask = imageOperator.calculate_final_mask(contour);

    return final_mask;
}

cv::Mat Masking::final_mask(int label, std::string model_path, cv::Mat image, int sharpening_strength)
{
    ImageOperations imageOperator;
    OrtOperations ortOperator(model_path);

    // load image and prepare data
    ImageOperations::chw dimensions = imageOperator.load_image(image);
    std::vector<float> hwc_float_image = imageOperator.Get_chw_float_vector();
    float maximum = *std::max_element(hwc_float_image.begin(), hwc_float_image.end());
    float minimum = *std::min_element(hwc_float_image.begin(), hwc_float_image.end());
    imageOperator.convert_to_0_1_range(hwc_float_image, maximum, minimum);

    // run onnx model and get the output
    ortOperator.LoadImage(dimensions, hwc_float_image);
    ortOperator.inference();
    std::vector<float> initial_mask = ortOperator.get_initial_mask(label);

    // now working on mask with one channel
    imageOperator.Set_channels(1);

    // Get the initial mask
    imageOperator.convert_to_0_255_range_and_sharpen(initial_mask, maximum, minimum, sharpening_strength);
    initial_mask = imageOperator.convertToHWC(initial_mask);
    cv::Mat preprocessed_blurry_mask = imageOperator.convert_mask_to_proper_type(initial_mask);

    // Calculate the final mask
    std::vector<cv::Point> contour = imageOperator.find_largest_contour(preprocessed_blurry_mask);
    cv::Mat final_mask = imageOperator.calculate_final_mask(contour);

    return final_mask;
}