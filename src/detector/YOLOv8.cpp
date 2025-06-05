#include "detector/YOLOv8.h"

#include "common/Logger.h"
#include "common/Assert.h"
#include <filesystem>
#include <fmt/format.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>

namespace fastdet::detector {
    
    YOLOv8::YOLOv8(const std::string& onnxPath,
        const std::string& enginePath,
        const std::vector<std::string>& labels,
        float probabilityThreshold,
        float nmsThreshold)
    : mInferenceEngine(nullptr),
      mLabels(labels),
      mProbabilityThreshold(probabilityThreshold),
      mNmsThreshold(nmsThreshold),
      mTopK(100) { 

        mInferenceEngine = fastdet::inference::EngineFactory::create(fastdet::inference::EngineType::TensorRT);
        FASTDET_ASSERT_MSG(mInferenceEngine != nullptr, "Failed to create inference engine");
        
        if (std::filesystem::exists(enginePath)) {
            FASTDET_LOG_INFO("Loading pre-built TensorRT engine from {}", enginePath);
            bool success = mInferenceEngine->load(enginePath, {0.F, 0.F, 0.F}, {1.F, 1.F, 1.F}, true);
            FASTDET_ASSERT_MSG(success, "Failed to load TensorRT engine from {}", enginePath);
        } else {
            FASTDET_LOG_INFO("Building TensorRT engine from ONNX model at {}", onnxPath);
            fastdet::inference::Options options{
                .optBatchSize = 1,
                .maxBatchSize = 1,
            };

            bool success = mInferenceEngine->build(onnxPath, options);
            FASTDET_ASSERT_MSG(success, "Failed to build TensorRT engine from ONNX model at {}", onnxPath);
        }
    }
    
    std::vector<Detection> YOLOv8::detect(const cv::Mat &image) {
        const auto input = preprocess(image);
        
        std::vector<std::vector<std::vector<float>>> output;
        auto success = mInferenceEngine->infer(input, output);
        FASTDET_ASSERT_MSG(success, "Unable to run inference.");
        
        std::vector<Detection> ret;
        const auto &numOutputs = mInferenceEngine->getOutputDims().size();
        
        if (numOutputs == 1) {
            std::vector<float> features;
            FASTDET_ASSERT_MSG(output.size() == 1 && output[0].size() == 1, "The feature vector has incorrect dimensions!");
            features = std::move(output[0][0]);
            ret = postprocess(features);
        } else {
            FASTDET_LOG_ERROR("Incorrect number of outputs: {}", numOutputs);
            FASTDET_ASSERT_MSG(false, "Incorrect number of outputs!");
        }
        
        return ret;
    }
    
    void YOLOv8::draw(cv::Mat &image, const std::vector<Detection> &detections, unsigned int scale) {
        const cv::Scalar bbox_color(0, 0, 255);
        const cv::Scalar label_color(255, 255, 255);
        
        for (const auto& detection : detections) {
            const auto &rect = detection.rect;
            
            cv::rectangle(image, rect, bbox_color * 255, scale + 1);
            
            char text[256];
            sprintf(text, "%s %.1f%%", mLabels[detection.label].c_str(), detection.probability * 100);
            
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);
            
            int x = detection.rect.x;
            int y = detection.rect.y + 1;
            
            cv::Scalar txt_bk_color = bbox_color * 0.7 * 255;
            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);
            cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, label_color, scale);
        }
    }
    
    std::vector<std::vector<cv::cuda::GpuMat>> YOLOv8::preprocess(const cv::Mat &image) { 
        const auto &inputDims = mInferenceEngine->getInputDims();
        
        cv::cuda::GpuMat gpuImage;
        gpuImage.upload(image);
        
        cv::cuda::GpuMat rgbMat;
        cv::cuda::cvtColor(gpuImage, rgbMat, cv::COLOR_BGR2RGB);
        
        auto resized = rgbMat;
        
        if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
            float r = std::min(inputDims[0].d[2] / (rgbMat.cols * 1.0f), inputDims[0].d[1] / (rgbMat.rows * 1.0f));
            
            int unpad_w = r * rgbMat.cols;
            int unpad_h = r * rgbMat.rows;
            
            cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
            cv::cuda::resize(rgbMat, re, re.size());
            resized = cv::cuda::GpuMat(inputDims[0].d[1], inputDims[0].d[2], CV_8UC3, cv::Scalar(0, 0, 0)); 
            re.copyTo(resized(cv::Rect(0, 0, re.cols, re.rows)));
        }
        
        std::vector<cv::cuda::GpuMat> input{std::move(resized)};
        std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};
        
        mImgHeight = rgbMat.rows;
        mImgWidth = rgbMat.cols;
        mRatio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));
        
        return inputs;
    }

    std::vector<Detection> YOLOv8::postprocess(std::vector<float> &features) {
        const auto &outputDims = mInferenceEngine->getOutputDims();
        const int numChannels = outputDims[0].d[1];
        const int numAnchors = outputDims[0].d[2];
        const int numClasses = static_cast<int>(mLabels.size());
        
        FASTDET_ASSERT_MSG(static_cast<size_t>(numAnchors * numChannels) == features.size(), "Output size mismatch: expected {}, got {}", numAnchors * numChannels, features.size());
                          
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> labels;
        std::vector<int> indices;

        cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, features.data());
        output = output.t();

        for (int i = 0; i < numAnchors; i++) {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
            float score = *maxSPtr;
            
            if (score > mProbabilityThreshold) {
                float x = *bboxesPtr++;
                float y = *bboxesPtr++;
                float w = *bboxesPtr++;
                float h = *bboxesPtr;
                
                float x0 = std::clamp((x - 0.5f * w) * mRatio, 0.f, mImgWidth);
                float y0 = std::clamp((y - 0.5f * h) * mRatio, 0.f, mImgHeight);
                float x1 = std::clamp((x + 0.5f * w) * mRatio, 0.f, mImgWidth);
                float y1 = std::clamp((y + 0.5f * h) * mRatio, 0.f, mImgHeight);
                
                int label = static_cast<int>(maxSPtr - scoresPtr);
                
                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;
                
                bboxes.push_back(bbox);
                labels.push_back(label);
                scores.push_back(score);
            }
        }
        
        cv::dnn::NMSBoxesBatched(bboxes, scores, labels, mProbabilityThreshold, mNmsThreshold, indices);
        
        std::vector<Detection> detections;

        int cnt = 0;
        for (int idx : indices) {
            if (cnt >= mTopK) {
                break;
            }

            Detection det{};
            det.probability = scores[idx];
            det.label = labels[idx];
            det.rect = bboxes[idx];
            detections.push_back(det);

            cnt += 1;
        }

        return detections;
    }
}