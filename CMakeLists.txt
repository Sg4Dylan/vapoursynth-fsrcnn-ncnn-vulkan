project(vs_fsrcnn_vulkan)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(OpenCV "D:/ProgramData/opencv/build")
include_directories(${OpenCV}/include)
link_directories(${OpenCV}/x64/vc15/lib)

set(Vulkan "D:/ProgramData/VulkanSDK/1.1.126.0")
include_directories(${Vulkan}/Include)
link_directories(${Vulkan}/Lib)

set(NCNN_Build "D:/Workspace/ncnn-master/build-vs2017/install")
include_directories(${NCNN_Build}/include/ncnn)
link_directories(${NCNN_Build}/lib)

set(VapourSynthSDK "D:/Program Files/VapourSynth/sdk")
include_directories(${VapourSynthSDK}/include/vapoursynth)
link_directories(${VapourSynthSDK}/lib64)

add_library(vs_fsrcnn_vulkan SHARED main.c fsrcnn.cpp)
target_link_libraries(vs_fsrcnn_vulkan vapoursynth)
target_link_libraries(vs_fsrcnn_vulkan vsscript)
target_link_libraries(vs_fsrcnn_vulkan opencv_world412)
target_link_libraries(vs_fsrcnn_vulkan vulkan-1)
target_link_libraries(vs_fsrcnn_vulkan ncnn)
