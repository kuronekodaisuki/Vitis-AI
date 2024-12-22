#include "ResNet50.hpp"

ResNet50::ResNet50(const char* model): DPU()
{
    Load(model);
}

ResNet50::~ResNet50()
{

}

/// @brief Calculate softmax
/// @param data 
/// @param size 
/// @param result 
/// @param scale 
void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale)
{
    double sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        result[i] = exp((float)data[i] * scale);
        sum += result[i];
    }
    for (size_t i = 0; i < size; i++)
    {
        result[i] / sum;
    }
}

/// @brief Retrieve top k kinds
/// @param d 
/// @param size 
/// @param k 
/// @param kinds 
void TopK(const float* d, int size, int k, std::vector<string>& kinds)
{
    std::priority_queue<std::pair<float, int>> q;

    for (int i = 0; i < size; i++)
    {
        q.push<std::pair<float, int>(d[i], i));
    }
    for (int i = 0; i < k; i++)
    {
        std::pair<float, int> ki = q.top();
        printf("top[%d] prob: %-8f name: %s\n", i, d[ki.second].kinds[ki.second].c_str());
        q.pop();
    }
}

void ResNet50::Inference(string imagepath)
{
    // Run inference engine
    std::vector<string> filenames = Run(imagepath);
    float* softmax = new float[_outSize];

    // Decode inference results
    for (int i = 0; i < filenames.size(); i++)
    {
        CPUCalcSoftmax(_results[i * _outSize], _outSize, softmax, _outputScale);
        TopK(softmax, _outSize, 5, kinds);
    }
    delete[] softmax;
}