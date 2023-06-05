#include "nvCodecYoloV5.h"
#include <stdio.h>
#include <string.h>
using namespace std;
using namespace cuvid;

int main(int argc, char** argv){
    
    std::string uri = "NvCodecCuda/workspace/exp/circles.mp4";
    const char* method = "yolo";
    std::string modelFile = "yolov5s";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "demuxer") == 0){
        cuvid::AppDemuxer(uri);
    }else if(strcmp(method, "hard_decode") == 0){
        AppHardDecode(uri);
    }else if(strcmp(method, "yolo") == 0){
        AppYoloV5(modelFile, uri, true);
    }else{
        printf("Unknow method: %s\n", method);
        printf(
            "Help: \n"
            "    ./pro method[demuxer]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro fall\n"
        );
    }
    return 0;
}