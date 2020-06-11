# ROS_NCNN #

This is a ROS package for NCNN, a high-performance neural network inference framework *- by Tencent -* optimized for mobile platforms:

- ARM NEON assembly level optimization
- Sophisticated memory management and data structure design, very low memory footprint
- Supports multi-core parallel computing acceleration
- Supports GPU acceleration via the next-generation low-overhead Vulkan API
- The overall library size is less than 700K, and can be easily reduced to less than 300K
- Extensible model design, supports 8bit quantization and half-precision floating point storage
- Can import caffe/pytorch/mxnet/onnx models



## Setting up ##

### Library ###

- [Build for NVIDIA Jetson](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-nvidia-jetson)
- [Build for Linux x86](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86)
- [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)
- [Build for MacOSX](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macosx)
- [Build for Raspberry Pi 3](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-raspberry-pi-3)
- [Build for ARM Cortex-A family with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling)
- [Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
- [Build for iOS on MacOSX with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macosx-with-xcode)
- [Build for iOS on Linux with cctools-port](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-linux-with-cctools-port)
- [Build for Hisilicon platform with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-hisilicon-platform-with-cross-compiling)



## ROS package ##

* Clone this repository into your catkin workspace.
* Initialize and update submodule `ncnn-assets` *( this is a collection of some popular models )*
* Compile the workspace.
* CMake script is going to autodetect whether the **ncnn library** is built with **Vulkan** or not. _( All nodes will utilize the GPU if Vulkan is enabled. )_


```xml
<node name="yolact_node" pkg="ros_ncnn" type="yolact_node" output="screen">
  <param name="display_output" value="$(arg display_output)"/>
  <remap from="/camera/image_raw" to="$(arg camera_topic)"/>
  <!-- Select discrete GPU, in any other case the node jumps to the first discrete GPU. -->
  <param name="gpu_device" value="0"/>
  <!-- Number of CPU threads to use, uses all available if not provided. -->
  <param name="num_threads" value="8"/>
</node>
```

### Yolact on ROS_NCNN ###
![](doc/yolact.png)

### RetinaFace on ROS_NCNN ###
![](doc/retinaface.png)

### Faster RCNN ###
Don't forget to uncompress `ZF_faster_rcnn_final.bin.zip` in assets directory first.
![](doc/rcnn.png)

## :construction:  ##

* General model loader node
* Message generation ( eg. faceObject etc... )
* Dynamic reconfiguration
