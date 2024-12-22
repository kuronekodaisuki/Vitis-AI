#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common/common.h"

class DPU
{
public :
    /// @brief Constructor
    DPU();

    /// @brief Destructor
    ~DPU();

    /// @brief Load model
    /// @param model_filename
    /// @return if succes return true
    bool Load(const char* model_filename);

    /// @brief read images from images_filepath
    /// @param images_filepath 
    /// @return returns image filenames
    std::vector<string> Run(const char* images_filepath);

protected :
    /// @brief Preprocess image files
    /// @param filepath 
    /// @param filenames 
    /// @return 
    int8_t* PreProcess(const char* filepath, std::vector<string> filenames);

protected :
    vart::Runner* _runner;
    std::vector<const xir::Tensor*> _inputTensors;
    std::vector<const xir::Tensor*> _outputTensors;
    TensorShape* _inputShapes;
    TensorShape* _outputShapes;
    std::vector<vart::TensorBuffer*> _inputBuffers;
    std::vector<vart::TensorBuffer*> _outputBuffers;
    std::vector<std::unique_ptr<vart::TensorBuffer> _inputs;
    std::vector<std::unique_ptr<vart::TensorBuffer> _outputs;
    int8_t* _inputBlob;
    int8_t* _results;
    int _outSize;
    float _outputScale;
};