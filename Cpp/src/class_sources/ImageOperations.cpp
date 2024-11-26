#include "ImageOperations.hpp"

// this function loads an image saved under given filepath and converts it into a vector
// it also rearranges the format from hwc to cwh
ImageOperations::chw ImageOperations::load_image(std::string image_path)
{
    // load the image, if not possible end the program with code 1
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    image.convertTo(image, CV_8UC3);
    if (!image.data)
    {
        throw std::runtime_error("Cannot open the image, check image path!");
    }
    this->initial_image = image.clone();

    ImageOperations::chw data;
    data.channels = image.channels();
    data.height = image.rows;
    data.width = image.cols;

    this->channels = image.channels();
    this->height = image.rows;
    this->width = image.cols;

    return data;
}

ImageOperations::chw ImageOperations::load_image(cv::Mat image)
{
    if (!image.data)
    {
        throw std::runtime_error("Image .data is empty");
    }
    if (image.type() != CV_8UC3)
    {
        throw std::runtime_error("image is of wrong type, must be CV_8UC3, this does not have depth of 8");
    }
    if (image.channels() != 3)
    {
        throw std::runtime_error("image is of wrong type, must be CV_8UC3, this does not have 3 channels");
    }
    this->initial_image = image.clone();

    ImageOperations::chw data;
    data.channels = image.channels();
    data.height = image.rows;
    data.width = image.cols;

    this->channels = image.channels();
    this->height = image.rows;
    this->width = image.cols;

    return data;
}

std::vector<float> ImageOperations::Get_chw_float_vector()
{
    // convert data to float type
    cv::Mat image = this->initial_image.clone();
    image.convertTo(image, CV_32FC3);

    int channels = this->channels;
    int height = this->height;
    int width = this->width;

    std::vector<float> chw_data(height * width * channels);

    // load data into vector in chw order
    for (int channel = 0; channel < channels; channel++)
    {
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                chw_data[channel * height * width + row * width + column] = image.at<cv::Vec3f>(row, column)[channel];
            }
        }
    }
    // pack what is returned into the structure
    return chw_data;
}

// converts the input data to 0-1 range
void ImageOperations::convert_to_0_1_range(std::vector<float> &vec, float max, float min)
{
    std::ranges::transform(vec, vec.begin(), [max, min](float val)
                           {
        val = val - min;
        val = val / (max-min);
        return val; });
}

// set channels to 1 when working with mask
void ImageOperations::Set_channels(int i)
{
    this->channels = i;
}

// converts mask back to 0-255 range
void ImageOperations::convert_to_0_255_range_and_sharpen(std::vector<float> &vec, float max, float min, float sharpening_strength)
{
    std::ranges::transform(vec, vec.begin(), [max, min, sharpening_strength](float val)
                           {
        val = val * (max-min);
        val = val + min;
        val = val - sharpening_strength; // sharpening step
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        val = std::round(val);
        return val; });
}

std::vector<float> ImageOperations::convertToHWC(std::vector<float> &vec)
{
    std::vector<float> HWC_data(this->height * this->width * this->channels);

    // load data into vector in chw order
    for (int height = 0; height < this->height; height++)
    {
        for (int width = 0; width < this->width; width++)
        {
            for (int channels = 0; channels < this->channels; channels++)
            {
                HWC_data[height * this->width * this->channels + width * this->channels + channels] = vec[channels * this->height * this->width + height * this->width + width];
            }
        }
    }
    return HWC_data;
}

cv::Mat ImageOperations::convert_mask_to_proper_type(std::vector<float> &mask)
{
    cv::Mat new_mask(this->height, this->width, CV_32FC1, mask.data());
    new_mask.convertTo(new_mask, CV_8U, 1.0, 0);
    return new_mask;
}

// gets the contours of the mask, and returns the largest
std::vector<cv::Point> ImageOperations::find_largest_contour(cv::Mat blurry_mask)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(blurry_mask, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> largestContour = *std::max_element(contours.begin(), contours.end(),
                                                              [](const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2)
                                                              {
                                                                  return cv::contourArea(c1) < cv::contourArea(c2);
                                                              });

    return largestContour;
}

cv::Mat ImageOperations::calculate_final_mask(std::vector<cv::Point> contour)
{
    cv::Mat contourMask = cv::Mat::zeros(this->height, this->width, CV_8U);
    cv::drawContours(contourMask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, -1);
    return contourMask;
}

// returns initial image in a type that can be masked
cv::Mat ImageOperations::Get_initial_image()
{
    return this->initial_image;
}