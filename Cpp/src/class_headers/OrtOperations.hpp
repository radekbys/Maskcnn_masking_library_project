#pragma once
#include <onnxruntime_cxx_api.h>
#include <iostream>

#include "ImageOperations.hpp"

class OrtOperations
{
private:
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session;
    Ort::Value inputTensor;
    std::vector<float> input_raw;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    std::vector<Ort::Value> output_tensors;

    std::vector<const char *> get_input_names(Ort::Session *session);
    std::vector<const char *> get_output_names(Ort::Session *session);
    void deallocate_names(std::vector<const char *> &names);

public:
    OrtOperations(std::string model_filepath);
    void LoadImage(ImageOperations::chw dims, std::vector<float> data);
    void inference();
    std::vector<float> get_initial_mask(int label);
    ~OrtOperations();
};