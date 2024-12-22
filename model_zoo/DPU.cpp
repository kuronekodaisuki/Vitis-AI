#include <opencv2/opencv.hpp>
#include "DPU.hpp"

DPU::DPU(): 
    _runner(nullptr), _inputBlob(nullptr), _results(nullptr),
    _inputTensors(nullptr), _outputTensors(nullptr),
    _inputShapes(nullptr), _outputShapes(nullptr)
{

}

DPU::~DPU()
{
    // Destroy objects
    if (_inputBlob != nullptr)
        delete[] _inputBlob;
    if (_results != nullptr)
        delete[] _results;
}

bool DPU::Load(const char* model_filename)
{
    auto graph = xir::Graph::deserialize(model_filename);
    auto subgraph = get_dpu_subgraph(graph.get());
    
    // Create runner
    _runner = vart::Runner::create_runner(subgraph[0], "run");

    _inputTensors = _runner->get_input_tensors();
    _outputTensors = _runner->get_output_tensors();

    _inputShapes = new TensorShape[_inputTensors.size()];
    _outputShapes = new TensorShape[_outputTensors.size()];

    GraphInfo shapes = {_inputShapes, _outputShapes};

    getTensorShape(_runner.get(), &shapes, _inputTensors.size(), _outputTensors.size());
}

std::vector<string> DPU::Run(string images_filepath)
{
    std::vector<string> image_filenames;
    struct stat s;
    lstat(images_filepath.c_str(), &s);

    if (S_IS_DIR(s.st_mode))
    {
        // Enumerate image file names
        DIR* dir = opendir(images_filepath.c_str());
        if (dir != nullptr)
        {
            for (struct dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir))
            {
                if ( entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN)
                {
                    string name = entry->d_name;
                    string ext = name.substr(name.find_last(".") + 1);
                    switch (ext)
                    {
                        case "JPEG":
                        case "JPG":
                        case "jpeg":
                        case "jpg":
                        case "PNG":
                        case "png":
                        image_filenames.push_back(name);
                    }
                }
            }
            closedir(dir);
        }
    }
    else
    {
        image_filenames.push_back(images_filepath);
    }

    // PreProcess images
    PreProcess(images_filepath, image_filenames);

    auto in_dims = _inputTensors[0]->get_shape();
    auto out_dims = _outputTensors[0]->get_shape();

    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(_inputTensors[0]->get_name(), in_dims, xir::DataTyp{xir::DataType::XINT, 8u})));
    _inputs.push_back(std::make_unique<CpuFlatTensorBuffer)(blob, batchTensors.back().get());

    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(_outputTensors[0]->get_name(), out_dims, xir::DataTyp{xir::DataType::XINT, 8u})));


    // Run async
    std::pair<uint32_t, int> job = _runner->execute_async(_inputBuffers, _outputBuffers);

    // Wait job infinite
    _runner->wait(job.first, -1);

    // Retrieve output data

    return image_filenames;
}

int8_t* DPU::PreProcess(string filepath, std::vector<string> filenames)
{
    int width = _inShapes[0].width;
    int height = _inputShapes[0].height;
    int inSize = _inputShapes[0].size;
    _outSize = _outputShapes[0].size;

    float input_scale = get_input_scale(_inputTensors[0]);
    _outputScale = get_output_scale(_outputTensors[0]);
    float mean[3] = {104, 107, 123};

    _inputBlob = new int8_t[filenames.size() * inSize];
    _results = new int8_t[filenames.size() * _outSize];
    
    // Resize and convert to blob for each images
    for (size_t i = 0; i < filenames.size(); i++)
    {
        cv::Mat image = cv::imread(filepath + filenames[i]);
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height), 0, 0);
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; x < width; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    blob[i * size + h * width * 3 + w * 3 + c] = (int8_t)((resized.at<cv::Vec3b>(h, w)[c] - mean[c]) * input_scale);
                }
            }
        }
    }
    return _results;
}