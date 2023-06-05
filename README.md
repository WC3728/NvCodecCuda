## 介绍

基于视频目标检测的硬件解码教程

## 软件架构

1. Ubuntu 20.04
2. Cmake 3.16
3. gcc/g++  9.4
4. cuda 11.4
5. TensorRT 8.5.3
6. cudnn 8.9.0
7. protobuf 3.11.4
8. FFmpeg 4.0
9. Video_Codec_SDK 12.1.14
10. OpenCV-4.x

## 安装教程

- cuda 安装

  - 使用`nvidia-smi`查看计算机驱动版本， 如果没有安装驱动，则按照提示输入命令安装
  - 根据版本选择对应的cuda版本进行安装[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive) 
  - 点击选择后进入下面页面, 根据计算机的配置进行选择, 选择后根据下面的命令行在计算机上进行安装
  - `如果使用sudo sh cuda_xxx ` 执行失败， 依次执行下面的命令

  ```shell
  sudo su
  bash cuda_xxx
  ```

  - 安装完成后需要进行下面的配置

  ```shell
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  ```


- TensorRT 下载

  - 下载链接:[TensorRT - Get Started | NVIDIA Developer](https://developer.nvidia.com/tensorrt-getting-started) , 链接有时候进不去，出现这种情况想其他办法。

  ```shell
  tar -zxvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
  mv TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6 TensorRT
  mv TensorRT /usr/local/include/
  ```

  

- cudnn 下载

  - 下载链接: [CUDA Deep Neural Network (cuDNN) | NVIDIA Developer](https://developer.nvidia.com/cudnn) ，同样链接有可能进不去

  ```shell
  tar -xf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
  mv cudnn-linux-x86_64-8.9.0.131_cuda11-archive cudnn
  mv cudnn /usr/local/include/
  ```

  

- protobuf 下载

  - 下载链接: [protobuf-3.11.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.11.4) 
  - 选择 protobuf-cpp-3.11.4.tar.gz 进行下载

  ```shell
  tar -zxvf protobuf-cpp-3.11.4.zip
  cd protobuf-cpp-3.11.4
  
  cmake . -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/include/protobuf-3.11.4
  make -j8
  sudo make install
  ```

  

- FFmpeg 安装

  - 下载链接：[FFmpeg4.0](https://github.com/FFmpeg/FFmpeg/tree/release/4.0) 
  - 以zip 压缩文件进行下载
  - 下载命令

  ```shell
  sudo apt-get install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
  
  unzip ffmpeg
  cd ffmpeg
  # 参数根据续需求进行配置
  # 在后面的版本中 enable-cuda-sdk -> enable-cuda-nvcc
  # 如果提示算力的错误, 即是出现了sm_xxx之类
  # 需要对configure 配置文件进行修改为(一般RTX30000系列的显卡做如下修改):
  # nvccflags_default="-gencode arch=compute_75,code=sm_75 -O2"
  
  ./configure --prefix=/usr/local/include/ffmpeg --enable-avresample --enable-nonfree --enable-cuda-sdk --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
  
  make -j8
  sudo make install
  
  code ~/.bashrc
  export FFMPEG_ROOT=/usr/local/include/ffmpeg
  export PATH=$FFMPEG_ROOT/bin:$PATH
  export LD_LIBRARY_PATH=$FFMPEG_ROOT/lib:$LD_LIBRARY_PATH
  source ~/.bashrc
  
  cd ..
  ffmpeg -h 
  # successfully
  ```

  

- Video_Codec_SDK 安装

  - 进入nvidia官方提供的下载界面: [Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)


- OpenCV-4.x 安装

```shell
# opencv4.2.0 https://github.com/opencv/opencv/tree/4.2.0
# opencv_contrib4.2.0  https://github.com/opencv/opencv_contrib/tree/4.2.0
# 使用zip形式进行下载
unzip opencv-4.2.0.zip
unzip opencv_contrib-4.2.0.zip
rm opencv-4.2.0.zip
rm opencv_contrib-4.2.0.zip


sudo apt-get install build-essential libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev


cd opencv4.2.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local/include/opencv4.2.0 \  
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules \  
	-D WITH_CUDA=ON \
	-D ENABLE_FAST_MATH=ON \
	-D CUDA_FAST_MATH=ON \
	-D WITH_CUBLAS=ON \
	-D WITH_NVCUVID=ON \
	-D WITH_TBB=ON \
	-D WITH_OPENMP=ON \
	-D WITH_OPENGL=ON ..

make -j8
sudo make install
```


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



## 运行工程
1. 拉镜像
```shell
git clone https://github.com/WC3728/NvCodecCuda.git
cd NvCodecCuda
```
2.  配置CMakeLists.txt 文件, 需要修改为刚刚安装的第三方包的根目录
```CMake
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")
# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/opt/intel/openvino_2022/opencv/cmake")
set(CUDA_DIR     "/usr/local/cuda")
set(CUDNN_DIR    "/usr/local/include/cudnn")
set(TENSORRT_DIR "/usr/local/include/TensorRT")
set(NVDEC_DIR    "/usr/local/include/NvCodecCuda/Video_Codec_SDK_12.1.14")
set(FFMPEG_DIR   "/usr/local/ffmpeg")
set(PROTOBUF_DIR "/usr/local/include/protobuf-3.11.4")

```
3. 编译项目
```shell
mkdir build && cd build && cmake ..

make -j8

# 这一步会导出两个动态库, 用于测试工程项目
make install

```
4. 运行示例程序
```shell
cd ../workspace

# 1. 示例1
./pro yolo

# 2. 示例2
./pro demuxer

# 3. 示例3
./pro hard_decoder

```



## 测试导出的动态库（封装）

```shell
cd test-proj

# 1.同样需要修改test-proj 目录下CMakeLists.txt 的几个目录后, 运行下面的命令

# 2. 修改main.cpp 下面几项代码
std::string uri = "exp/circles.mp4";  // 更改为视频文件地址

mkdir build && cd build && cmake ..

make -j4

# 运行示例程序
cd ../../workspace


./main yolo 

./main hard_decode

./main demuxer

# 能够得到相同的输出结果

```




### 提示
- 在执行命令 `make yolo -j8`, 如果出现错误信息`(Unable to load library: libnvinfer_builder_resource.so.8.5.3)`, 尝试检测CMakeLists.txt 的link文件是否正确是不能生效的, 正确的做法是执行以下命令
```shell
cd /usr/local/include/TensorRT/targets/x86_64-linux-gnu/lib # 这是TensorRT 的安装目录
sudo cp libnvinfer_builder_resource.so.8.5.3 /usr/lib
```








