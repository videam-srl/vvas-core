#pragma once
#include "vvas_dpupriv.hpp"

#include <vitis/ai/yolovx.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>

using namespace std;
using namespace cv;

class vvas_yolovx:public vvas_dpumodel
{

    int log_level = 0;
    std::unique_ptr < vitis::ai::YOLOvX > model;

public:

    vvas_yolovx (void *handle, const std::string & model_name,
                  bool need_preprocess);

    virtual int run (void *handle, std::vector<cv::Mat>& images,
                     VvasInferPrediction ** predictions);

    virtual int requiredwidth (void);
    virtual int requiredheight (void);
    virtual int supportedbatchsz (void);
    virtual int close (void);

    virtual ~vvas_yolovx ();
};