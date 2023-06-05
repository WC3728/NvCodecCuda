#include "nvCodecYoloV5.h"
#include "ffhdd/cuvid_decoder.hpp"
#include "ffhdd/ffmpeg_demuxer.hpp"
#include "ffhdd/nalu.hpp"
#include "common/ilogger.hpp"
#include "opencv2/opencv.hpp"
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include "app_yolo/yolo.hpp"

static const char* labels[] = {
    "slots"
};

using namespace cuvid;


namespace functionals {

    static std::shared_ptr<Yolo::Infer> get_yolo(Yolo::Type type, TRT::Mode mode, const std::string& model, int device_id){
        auto mode_name = TRT::mode_string(mode);
        TRT::set_device(device_id);

        auto int8process = [=](int current, int count, const std::vector<std::string>&files, std::shared_ptr<TRT::Tensor>& tensor){
            INFO("INT8 %d / %d", current, count);
            for(int i = 0; i < files.size(); ++i){
                auto image = cv::imread(files[i]);
                Yolo::image_to_tensor(image, tensor, type, i);
            }
        };

        const char* name = model.c_str();
        INFO("================== TEST %s %s %s ===================",
            Yolo::type_name(type), mode_name, name);

        std::string onnx_file = iLogger::format("%s.onnx", name);
        std::string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
        int test_batch_size = 16;

        if (not iLogger::exists(model_file)){
            TRT::compile(
                mode,
                test_batch_size,
                onnx_file,
                model_file,
                {},
                int8process,
                "inference"
            );

        }

        return Yolo::create_infer(
            model_file,
            type,
            device_id,
            0.25f,
            0.45f,
            Yolo::NMSMethod::FastGPU,
            1024,
            false
        );
    }
    void render(
        std::vector<std::shared_future<Yolo::BoxArray>>& arrays, 
        const std::string& name, const std::string& uri){

        iLogger::rmtree(name);
        iLogger::mkdir(name);

        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::VideoCapture capture(uri);
        cv::Mat image;

        if(!capture.isOpened()){
            INFOE("Open Video is Failed.");
            return;
        }
        int iframe = 0;

        while (capture.read(image) && iframe < arrays.size()){
            auto objs = arrays[iframe].get();
            for (auto& obj : objs){
                uint8_t b, g, r;
                std::tie(b, g, r) = iLogger::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
                auto Name = labels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            cv::imshow(name, image);
        }


        capture.release();
        cv::destroyWindow(name);

    
    }
    
    
    static void hard_decode(const std::string& modelFile, const std::string& uri){
        auto name = "hard";
        int yolo_device_id = 0;
        auto yolo = get_yolo(
            Yolo::Type::V5,
            TRT::Mode::FP32,
            modelFile,
            yolo_device_id
        );
        if (yolo == nullptr){
            INFOE("YOLO create Failed.");
            return;
        }


        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri);
        if(demuxer == nullptr){
            INFOE("demuxer create Failed.");
            return;
        }

        int decoder_device_id = 0;
        auto decoder = FFHDDecoder::create_cuvid_decoder(
            true,
            FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, decoder_device_id
        );
        if (decoder == nullptr){
            INFOE("decoder create Failed.");
            return;
        }


        uint8_t* packet_data = nullptr;
        int packet_size = 0;
        int64_t pts = 0;
        
        demuxer->get_extra_data(&packet_data, &packet_size);
        decoder->decode(packet_data, packet_size);

        for (int i = 0; i < 10; ++i)
            yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();
        
        std::vector<std::shared_future<Yolo::BoxArray>> all_boxes;
        auto tic = iLogger::timestamp_now_float();
        do {
            demuxer->demux(&packet_data, &packet_size, &pts);
            int nddecoded_frame = decoder->decode(packet_data, packet_size, pts);
            for (int i =  0; i < nddecoded_frame; ++i){
                unsigned int frame_index = 0;
                Yolo::Image image(
                    decoder->get_frame(&pts, &frame_index),
                    decoder->get_width(), decoder->get_height(),
                    decoder_device_id,
                    decoder->get_stream()
                );

                all_boxes.emplace_back(yolo->commit(image));
            }
            
        } while (packet_size > 0);


        auto toc = iLogger::timestamp_now_float();
        INFO("%s decode and inference time: %.2f ms", name, toc - tic);
        render(all_boxes, name, uri);

    }


    static void soft_decode(const std::string& modelFile, const std::string& uri){
        auto name = "soft";
        int yolo_device_id = 0;
        auto yolo = get_yolo(Yolo::Type::V5, TRT::Mode::FP32, modelFile, yolo_device_id);
        if(yolo == nullptr){
            INFOE("Yolo create failed");
            return;
        }

        cv::VideoCapture capture(uri);
        cv::Mat image;

        if(!capture.isOpened()){
            INFOE("Open video failed.");
            return;
        }


        for (int i = 0; i < 10; ++i)    
            yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();
        
        std::vector<std::shared_future<Yolo::BoxArray>> all_boxes;

        auto tic = iLogger::timestamp_now_float();

        while (capture.read(image)){
            all_boxes.emplace_back(yolo->commit(image));
        }
        
        auto toc = iLogger::timestamp_now_float();
        INFO("soft decode and inference time: %.2f ms", toc - tic);

        functionals::render(all_boxes, name, uri);

    }


}

void cuvid::AppYoloV5(const std::string& modelFile, const std::string& uri, bool code){
    using namespace functionals;
    
    if (code)
        hard_decode(modelFile, uri);
    else
        soft_decode(modelFile, uri);
    
}

void cuvid::AppHardDecode(const std::string& uri){
    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, false);

    if(demuxer == nullptr){
        INFOE("demuxer create failed");
        return;
    }

    auto decoder = FFHDDecoder::create_cuvid_decoder(
        false, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0
    );

    if (decoder == nullptr){
        INFOE("decoder create failed");
        return;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    demuxer->get_extra_data(&packet_data, &packet_size);

#if true
    iLogger::rmtree("imgs");
    iLogger::mkdir("imgs");
#endif
    do {
        demuxer->demux(&packet_data, &packet_size, &pts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
        for (int i = 0; i < ndecoded_frame; ++i){
            unsigned int frame_index = 0;
            cv::Mat image(
                decoder->get_height() * 1.5, decoder->get_width(),
                CV_8U, decoder->get_frame(&pts, &frame_index)
            );

            cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);
            frame_index += 1;
            INFO("write imgs/img_%05d.jpg  %dx%d", frame_index, image.cols, image.rows);
            cv::imwrite(cv::format("imgs/img_%05d.jpg", frame_index), image);

        }
    } while (packet_size > 0);

}

void cuvid::AppDemuxer(const std::string& uri){
    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri);
    if (demuxer == nullptr){
        INFOE("demuxer create failed");
        return;
    }

    INFO("demuxer create done.");

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;
    demuxer->get_extra_data(&packet_data, &packet_size);
    
    std::vector<uint8_t> extra_data(packet_size + 3);
    memcpy(extra_data.data() + 3, packet_data, packet_size);

    int ipacket = 0;
    auto frame_type = NALU::format_nalu_type(
        NALU::find_all_nalu_info(extra_data.data(), packet_size, 0));
    INFO("Extra Data size: %d, type: %s", packet_size, frame_type.c_str());

    do {
        demuxer->demux(&packet_data, &packet_size, &pts);
        if (packet_size > 0){
            frame_type = NALU::format_nalu_frame_type(NALU::find_all_nalu_info(packet_data, packet_size, 0));
            INFO("Packet %d NALU sizeL %d, pts = %lld, type = %s",
                ipacket,
                packet_size,
                pts,
                frame_type.c_str()
            );
        }
    } while (packet_size > 0);
}


