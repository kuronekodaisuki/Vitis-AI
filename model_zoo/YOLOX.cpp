#include "YOLOX.h"


//#define INPUT_BLOB_NAME "images"
//#define OUTPUT_BLOB_NAME "output"

YOLOX::YOLOX(): _numClasses(80)
{
}

YOLOX::~YOLOX()
{

}

bool YOLOX::LoadModel(const char* filepath, uint width, uint height, uint channels, PRECISION precision)
{
    if (Load(filepath, width, height, channels, precision))
    {
        _grid_strides = generate_grids_and_stride();
        return true;
    }
    else
        return false;
}

/// <summary>
/// Inference
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
std::vector<Object> YOLOX::Detect(cv::Mat image)
{
    blobFromImage(image);
    doInference();

    postProcess(_width, _height, _width / (float)image.cols, _height / (float)image.rows);

    return _objects;
}

/// <summary>
/// Decode inference results
/// </summary>
/// <param name="prob"></param>
/// <param name="scale"></param>
/// <param name="image_w">Width of input image</param>
/// <param name="image_h">Height of input image</param>
void YOLOX::postProcess(const int width, const int height, float scaleX, float scaleY)
{
    _proposals.clear();
    generate_yolox_proposals(_bbox_confidential_threshold);

    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }

    std::vector<int> picked;
    nms_sorted_bboxes(_proposals, picked, _nms_threshold);

    size_t count = picked.size();

    _objects.resize(count);
    for (size_t i = 0; i < count; i++)
    {
        _objects[i] = _proposals[picked[i]];

        _objects[i].rect.x /= scaleX;
        _objects[i].rect.y /= scaleY;
        _objects[i].rect.width /= scaleX;
        _objects[i].rect.height /= scaleY;
    }
}

void YOLOX::DrawObjects(cv::Mat& image, const char* labels[], float threshold)
{
    for (const Object& object : _objects)
    {
	    if (object.prob < threshold)
		    continue;

        cv::Scalar color = cv::Scalar(255, 128, 128);
        float c_mean = (float)cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, object.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", labels[object.label], object.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = (int)object.rect.x;
        int y = (int)object.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);

    }
}

void YOLOX::DrawObjects(cv::Mat& image, const char* names[], const float colors[][3], float threshold)
{
    for (const Object& object : _objects)
    {
	    if (object.prob < threshold)
		    continue;

        cv::Scalar color = cv::Scalar(colors[object.label][0], colors[object.label][1], colors[object.label][2]);
        float c_mean = (float)cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, object.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", names[object.label], object.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = (int)object.rect.x;
        int y = (int)object.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);

    }
}


/// <summary>
/// Set thresholds
/// </summary>
/// <param name="bbox_conf_thres"></param>
/// <param name="nms_thres"></param>
void YOLOX::SetThresholds(float bbox_conf_thres, float nms_thres)
{
    _bbox_confidential_threshold = bbox_conf_thres;
    _nms_threshold = nms_thres;
}

/// <summary>
/// Convert image to tensor(1, channels, width, height)
/// </summary>
/// <param name="image"></param>
void YOLOX::blobFromImage(cv::Mat& image, bool bgr2rgb)
{
    cv::resize(image, _resized, cv::Size(_width, _height));
    if (bgr2rgb)
    {
        cv::cvtColor(_resized, _resized, cv::COLOR_BGR2RGB);
    }

    for (uint c = 0; c < _channels; c++)
    {
        for (uint h = 0; h < _height; h++)
        {
            for (uint w = 0; w < _width; w++)
            {
                _input[c * _width * _height + h * _width + w] = (float)_resized.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}


std::vector<YOLOX::GridAndStride> YOLOX::generate_grids_and_stride()
{
    std::vector<int> strides = { 8, 16, 32 };

    std::vector<GridAndStride> grid_strides;
    for (auto stride : strides)
    {
        int num_grid_y = _height / stride;
        int num_grid_x = _width / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back(GridAndStride(g0, g1, stride));
            }
        }
    }
    _numClasses = _output_shape[2] - 5;

    return grid_strides;
}

void YOLOX::generate_yolox_proposals(float prob_threshold)
{
    const size_t num_anchors = _grid_strides.size();

    for (size_t anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = _grid_strides[anchor_idx].grid0;
        const int grid1 = _grid_strides[anchor_idx].grid1;
        const int stride = _grid_strides[anchor_idx].stride;

        const size_t offset = anchor_idx * (_numClasses + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (_output[offset + 0] + grid0) * stride;
        float y_center = (_output[offset + 1] + grid1) * stride;
        float w = exp(_output[offset + 2]) * stride;
        float h = exp(_output[offset + 3]) * stride;
        float box_objectness = _output[offset + 4];

        float x0 = x_center - w / 2;
        float y0 = y_center - h / 2;

        cv::Rect_<float> bound(x0, y0, w, h);

        for (size_t class_idx = 0; class_idx < _numClasses; class_idx++)
        {
            float box_cls_score = _output[offset + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (prob_threshold < box_prob)
            {
                Object obj;
                obj.rect = bound;
                obj.label = (int)class_idx;
                obj.prob = box_prob;

                _proposals.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

void YOLOX::nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (size_t i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (nms_threshold < inter_area / union_area)
                keep = 0;
        }

        if (keep)
            picked.push_back((int)i);
    }
}



