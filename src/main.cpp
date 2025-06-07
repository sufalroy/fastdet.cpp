#include "common/Logger.h"
#include "common/Assert.h"
#include "detector/YOLOv8.h"

#include <memory>
#include <string>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

auto main(int argc, char *argv[]) -> int {
    try {
        FASTDET_LOG_INFO("Starting YOLOv8 Object Detection");
        FASTDET_LOG_INFO("TensorRT version: {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        int deviceCount = 0;
        cudaError_t const error = cudaGetDeviceCount(&deviceCount);
        FASTDET_ASSERT_MSG(error == cudaSuccess, "CUDA error: {}", cudaGetErrorString(error));
        FASTDET_ASSERT_MSG(deviceCount > 0, "No CUDA devices found");
        FASTDET_LOG_INFO("Found {} CUDA device(s)", deviceCount);

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp properties{};
            cudaGetDeviceProperties(&properties, i);
            FASTDET_LOG_INFO("Device {}: {}, Compute: {}.{}, Memory: {:.2f} GB, Multiprocessors: {}",
                             i, properties.name, properties.major, properties.minor,
                             static_cast<float>(properties.totalGlobalMem) / (1024 * 1024 * 1024),
                             properties.multiProcessorCount);
        }

        std::string onnxPath = (argc > 1)
                                   ? argv[1]
                                   : "C:/Laboratory/cpp-workspace/fastdet.cpp/models/yolo11s.onnx";
        std::string imagePath = (argc > 2)
                                    ? argv[2]
                                    : "C:/Laboratory/cpp-workspace/fastdet.cpp/inputs/parking.jpg";
        std::string outputPath = (argc > 3)
                                     ? argv[3]
                                     : "C:/Laboratory/cpp-workspace/fastdet.cpp/outputs/detection.jpg";

        FASTDET_LOG_INFO("ONNX model: {}", onnxPath);
        FASTDET_LOG_INFO("Input image: {}", imagePath);
        FASTDET_LOG_INFO("Output image: {}", outputPath);

        std::vector<std::string> labels = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        std::string enginePath = "./engines/yolo11s_fp16_b1_w-1.engine";

        FASTDET_LOG_INFO("Creating YOLOv8 detector");
        float probabilityThreshold = 0.5f;
        float nmsThreshold = 0.45f;

        auto detector = std::make_unique<fastdet::detector::YOLOv8>(
            onnxPath,
            enginePath,
            labels,
            probabilityThreshold,
            nmsThreshold
        );
        FASTDET_LOG_INFO("YOLOv8 detector created successfully");

        FASTDET_LOG_INFO("Loading image from: {}", imagePath);
        cv::Mat image = cv::imread(imagePath);
        FASTDET_ASSERT_MSG(!image.empty(), "Failed to load image: {}", imagePath);
        FASTDET_LOG_INFO("Image loaded successfully - Size: {}x{}", image.cols, image.rows);

        FASTDET_LOG_INFO("Running object detection...");
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<fastdet::detector::Detection> detections = detector->detect(image);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        FASTDET_LOG_INFO("Detection completed in {} ms", duration.count());

        detector->draw(image, detections, 1);
        FASTDET_LOG_INFO("Detections drawn on image");

        bool saved = cv::imwrite(outputPath, image);
        FASTDET_ASSERT_MSG(saved, "Failed to save result image to: {}", outputPath);
        FASTDET_LOG_INFO("Result image saved to: {}", outputPath);

        if (getenv("DISPLAY") != nullptr) {
            FASTDET_LOG_INFO("Displaying result image (press any key to close)");
            cv::namedWindow("YOLOv8 Detection Results", cv::WINDOW_AUTOSIZE);
            cv::imshow("YOLOv8 Detection Results", image);
            cv::waitKey(0);
            cv::destroyAllWindows();
        } else {
            FASTDET_LOG_INFO("No display available - result saved to file only");
        }

        FASTDET_LOG_INFO("Object detection completed successfully");
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        FASTDET_LOG_FATAL("Object detection failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
