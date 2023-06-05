# 硬件解码配合TensorRT
- 配置tensorRT一样的环境
- 增加NVDEC和ffmpeg的配置
- `make yolo -j64`
    - Yolo和硬件解码直接对接
- `make demuxer -j64`
    - 仅仅解封装得到h264的包，并分析是什么帧
- `make hard_decode -j64`
    - 硬件解码测试
- 软解码和硬解码，分别消耗cpu和gpu资源。在多路，大分辨率下体现明显
- 硬件解码和推理可以允许跨显卡
- 理解并善于利用的时候，他才可能发挥最大的效果


### 提示
- 在执行命令 `make yolo -j64`, 如果出现错误信息`(Unable to load library: libnvinfer_builder_resource.so.8.5.3)`, 尝试检测CMakeLists.txt 的link文件是否正确是不能生效的, 正确的做法是执行以下命令
```shell
cd /usr/local/include/TensorRT/targets/x86_64-linux-gnu/lib # 这是TensorRT 的安装目录
sudo cp libnvinfer_builder_resource.so.8.5.3 /usr/lib
```
