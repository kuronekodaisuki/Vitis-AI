#ifndef DPU_INCLUDED
#define DPU_INCLUDED

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

using namespace std;

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
    
    int _width;
    int _height;
    int8_t* _results;
    
protected :


    /// @brief Preprocess image files
    /// @param filepath 
    /// @param filenames 
    /// @return 
    int8_t* PreProcess(string filepath, std::vector<string> filenames);

    /// @brief Preprocess image files
    /// @param image_filename
    /// @return 
    int8_t* PreProcess(string image_filename);

protected :
    int _inSize;
    int _outSize;
    float _inputScale;
    float _outputScale;
    int8_t* _inputBlob;

    std::unique_ptr<vart::Runner> _runner;
    TensorShape* _inputShapes;
    TensorShape* _outputShapes;
    std::vector<const xir::Tensor*> _inputTensors;
    std::vector<const xir::Tensor*> _outputTensors;
    std::vector<vart::TensorBuffer*> _inputBuffers;
    std::vector<vart::TensorBuffer*> _outputBuffers;
    std::vector<std::unique_ptr<vart::TensorBuffer>> _inputs;
    std::vector<std::unique_ptr<vart::TensorBuffer>> _outputs;
};
#endif // DPU_INCLUDED