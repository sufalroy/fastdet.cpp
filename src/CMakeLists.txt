add_executable(runtime main.cpp
        common/logging.cpp
        inference/TensorRTEngine.cpp
        inference/EngineFactory.cpp)

target_include_directories(runtime
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TensorRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(runtime
        PUBLIC
        fmt::fmt
        spdlog::spdlog
        PRIVATE
        TensorRT::NvInfer
        TensorRT::OnnxParser
        TensorRT::Plugin
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
)

fastdet_set_project_warnings(runtime ON "" "" "" "")
fastdet_enable_sanitizers(runtime ON OFF OFF OFF OFF)