#include "OrtOperations.hpp"

OrtOperations::OrtOperations(std::string model_filepath)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
      session(nullptr),
      runOptions()
{
    // open session

    this->session = Ort::Session(env, model_filepath.c_str(), Ort::SessionOptions{nullptr});
}

// gets input names from the session
std::vector<const char *> OrtOperations::get_input_names(Ort::Session *session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    int input_names_count = session->GetInputCount();
    std::vector<const char *> input_names;

    for (int i = 0; i < input_names_count; i++)
    {
        std::string input_name = session->GetInputNameAllocated(i, allocator).get();
        char *input_name_copy = new char[input_name.length() + 1];
        std::strcpy(input_name_copy, input_name.c_str());
        input_names.push_back(input_name_copy);
    }
    return input_names;
}

// gets output names from the session
std::vector<const char *> OrtOperations::get_output_names(Ort::Session *session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    int output_names_count = session->GetOutputCount();
    std::vector<const char *> output_names;

    for (int i = 0; i < output_names_count; i++)
    {
        std::string output_name = session->GetOutputNameAllocated(i, allocator).get();
        char *output_name_copy = new char[output_name.length() + 1];
        std::strcpy(output_name_copy, output_name.c_str());
        output_names.push_back(output_name_copy);
    }
    return output_names;
}

void OrtOperations::LoadImage(ImageOperations::chw dims, std::vector<float> data)
{
    // create input tensor
    std::array<int64_t, 4> input_shape = {1, dims.channels, dims.height, dims.width};
    this->input_raw = data;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    this->inputTensor = Ort::Value::CreateTensor<float>(memory_info, input_raw.data(), input_raw.size(), input_shape.data(), input_shape.size());

    // Prepare input and output names
    this->input_names = get_input_names(&session);
    this->output_names = get_output_names(&session);
}

void OrtOperations::inference()
{
    try
    {
        this->output_tensors = session.Run(this->runOptions, this->input_names.data(), &(this->inputTensor), this->input_names.size(), this->output_names.data(), this->output_names.size());
    }
    catch (Ort::Exception &e)
    {
        throw std::runtime_error(e.what());
    }
}

// get the initial mask from output tensor list
std::vector<float> OrtOperations::get_initial_mask(int label)
{
    int *raw_masks_labels = static_cast<int *>(this->output_tensors[1].GetTensorMutableRawData());
    std::vector<int64_t> labels_tensor_shape = this->output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int no_object_detected_flag = false;
    int index;
    for (int i = 0; i <= labels_tensor_shape[0]; i++)
    {
        if (i == labels_tensor_shape[0])
        {
            throw std::runtime_error("No object under given label has been detected");
        }

        if (raw_masks_labels[i] == label)
        {
            index = i;
            break;
        }
    }

    float *raw_masks = static_cast<float *>(this->output_tensors[3].GetTensorMutableRawData());
    std::vector<int64_t> raw_masks_tensor_shape = this->output_tensors[3].GetTensorTypeAndShapeInfo().GetShape();

    index = 0;

    int len = raw_masks_tensor_shape[1] * raw_masks_tensor_shape[2] * raw_masks_tensor_shape[3];
    float *start = raw_masks + (len * index);
    float *end = raw_masks + (len * index) + len;
    std::vector<float> vec(start, end);

    return vec;
}

// deallocate names vectors
void OrtOperations::deallocate_names(std::vector<const char *> &names)
{
    for (const char *name : names)
    {
        delete[] name;
    }
    names.clear();
}

OrtOperations::~OrtOperations()
{
    deallocate_names(this->input_names);
    deallocate_names(this->output_names);
}