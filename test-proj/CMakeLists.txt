cmake_minimum_required(VERSION 3.6)

project(test)
set(CMAKE_CXX_STANDARD 11)
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/opt/intel/openvino_2022.3.0/opencv")
set(CUDA_DIR     "/usr/local/cuda")
set(CUDNN_DIR    "/usr/local/include/cudnn")
set(TENSORRT_DIR "/usr/local/include/TensorRT")
set(NVDEC_DIR    "/home/bckj/桌面/NvCodecCuda/Video_Codec_SDK_12.1.14")
set(FFMPEG_DIR   "/usr/local/ffmpeg")
set(PROTOBUF_DIR "/usr/local/include/protobuf-3.11.4")


find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")

include_directories(main PUBLIC
${PROJECT_SOURCE_DIR}/../install/include
)

link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
    ${NVDEC_DIR}/Lib/linux/stubs/x86_64
    ${FFMPEG_DIR}/lib
    ${PROJECT_SOURCE_DIR}/../install/lib
    ${OpenCV_DIR}/lib
)

add_executable(main main.cpp)

target_link_libraries(main nvCodecYoloV5 plugins)
target_link_libraries(main nvinfer nvinfer_plugin nvinfer_builder_resource)
target_link_libraries(main cuda cublas cudart cudnn)
target_link_libraries(main nvcuvid nvidia-encode)
target_link_libraries(main protobuf pthread)
target_link_libraries(main avcodec avformat avresample swscale avutil)
target_link_libraries(main opencv_imgproc opencv_core opencv_dnn opencv_imgcodecs opencv_video opencv_videoio opencv_highgui)


add_custom_target(
    yolo
    DEPENDS main
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./main yolo
)

add_custom_target(
    demuxer
    DEPENDS main
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./main demuxer
)

add_custom_target(
    hard_decode
    DEPENDS main
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./main hard_decode
)
