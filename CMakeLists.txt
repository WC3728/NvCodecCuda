cmake_minimum_required(VERSION 2.6)
project(pro)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/opt/intel/openvino_2022/opencv/cmake")

set(CUDA_DIR     "/usr/local/cuda")
set(CUDNN_DIR    "/usr/local/include/cudnn")
set(TENSORRT_DIR "/usr/local/include/TensorRT")
set(NVDEC_DIR    "/home/bckj/桌面/NvCodecCuda/Video_Codec_SDK_12.1.14")
set(FFMPEG_DIR   "/usr/local/ffmpeg")
set(PROTOBUF_DIR "/usr/local/include/protobuf-3.11.4")

find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/inc
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
    ${NVDEC_DIR}/Interface
    ${FFMPEG_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
    ${NVDEC_DIR}/Lib/linux/stubs/x86_64
    ${FFMPEG_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugins SHARED ${cuda_srcs})

add_library(nvCodecYoloV5 SHARED ${cpp_srcs})
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
install(TARGETS plugins nvCodecYoloV5 LIBRARY DESTINATION lib)
install(FILES
${PROJECT_SOURCE_DIR}/src/inc/nvCodecYoloV5.h
DESTINATION include)

add_executable(pro ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(pro nvinfer nvinfer_plugin nvinfer_builder_resource)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro nvcuvid nvidia-encode)
target_link_libraries(pro protobuf pthread plugins)
target_link_libraries(pro avcodec avformat avresample swscale avutil)
target_link_libraries(pro ${OpenCV_LIBS})

add_custom_target(
    yolo
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo
)

add_custom_target(
    demuxer
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro demuxer
)

add_custom_target(
    hard_decode
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro hard_decode
)
