#include "DPU.hpp"

class ResNet50: public DPU
{
public:
    /// @brief Constructor
    /// @param model 
    ResNet50(const char* model);

    /// @brief Destructor
    ~ResNet50();

    /// @brief Infer
    /// @param imagepath 
    void Inference(string imagepath);

};