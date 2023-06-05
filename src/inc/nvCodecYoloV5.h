#ifndef _NVCODEC_YOLOV5_H_
#define _NVCODEC_YOLOV5_H_
#include <string>


namespace cuvid {
    /**
     * @brief 视频解码
     * 
     * @param uri 
     */
    void AppHardDecode(const std::string& uri);

    /**
     * @brief 视频编码
     * 
     * @param uri 
     */
    void AppDemuxer(const std::string& uri);

    /**
     * @brief 目标检测 + 视频硬解码
     * 
     * @param modelFile 模型文件名称, 如yolov5s, 注意这里不包括文件后缀名称.
     * @param uri  视频流uri, 可以是视频文件, 也可以是支持各种协议的uri路径
     * @param code  默认为true, 表示使用硬件解码, false 表示使用 CPU进行解码(软解码)
     */
    void AppYoloV5(const std::string& modelFile, const std::string& uri, bool code = true);
}



#endif // !_NVCODEC_YOLOV5_H_