/*
 *
 * Copyright (C) 2022 Xilinx, Inc.
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * DOC: VVAS Dpu Infer Common APIs
 * This file contains common structures for inference classes.
 */

#pragma once

#ifndef __VVAS_DPUCOMMON_H__
#define __VVAS_DPUCOMMON_H__

/**
 * enum VvasClass - enum to define all supported model classes
 * @VVAS_XCLASS_YOLOV3: YOLOV3
 * @VVAS_XCLASS_CLASSIFICATION: CLASSIFICATION
 * @VVAS_XCLASS_FACEDETECT: FACEDETECT
 * @VVAS_XCLASS_VEHICLECLASSIFICATION: VEHICLECLASSIFICATION
 * @VVAS_XCLASS_SSD: SSD
 * @VVAS_XCLASS_REID: REID
 * @VVAS_XCLASS_REFINEDET: REFINEDET
 * @VVAS_XCLASS_TFSSD: TFSSD
 * @VVAS_XCLASS_YOLOV2: YOLOV2
 * @VVAS_XCLASS_SEGMENTATION: SEGMENTATION
 * @VVAS_XCLASS_PLATEDETECT: PLATEDETECT
 * @VVAS_XCLASS_PLATENUM: PLATENUM
 * @VVAS_XCLASS_POSEDETECT: POSEDETECT
 * @VVAS_XCLASS_BCC: BCC
 * @VVAS_XCLASS_EFFICIENTDETD2: EFFICIENTDETD2
 * @VVAS_XCLASS_FACEFEATURE: FACEFEATURE
 * @VVAS_XCLASS_FACELANDMARK:FACELANDMARK
 * @VVAS_XCLASS_ROADLINE: ROADLINE
 * @VVAS_XCLASS_ULTRAFAST: ULTRAFAST
 * @VVAS_XCLASS_RAWTENSOR: RAWTENSOR
 * @VVAS_XCLASS_NOTFOUND: UNKNOWN
 */
typedef enum {
  VVAS_XCLASS_YOLOV3,
  VVAS_XCLASS_FACEDETECT,
  VVAS_XCLASS_CLASSIFICATION,
  VVAS_XCLASS_VEHICLECLASSIFICATION,
  VVAS_XCLASS_SSD,
  VVAS_XCLASS_REID,
  VVAS_XCLASS_REFINEDET,
  VVAS_XCLASS_TFSSD,
  VVAS_XCLASS_YOLOV2,
  VVAS_XCLASS_SEGMENTATION,
  VVAS_XCLASS_PLATEDETECT,
  VVAS_XCLASS_PLATENUM,
  VVAS_XCLASS_POSEDETECT,
  VVAS_XCLASS_BCC,
  VVAS_XCLASS_EFFICIENTDETD2,
  VVAS_XCLASS_FACEFEATURE,
  VVAS_XCLASS_FACELANDMARK,
  VVAS_XCLASS_ROADLINE,
  VVAS_XCLASS_ULTRAFAST,
  VVAS_XCLASS_RAWTENSOR,
  VVAS_XCLASS_YOLOVX,

  VVAS_XCLASS_NOTFOUND
}VvasClass;

#endif
