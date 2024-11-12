cmake_minimum_required(VERSION 3.1)
project(vs_fsrcnn_vulkan)

include(GNUInstallDirs)

set(CMAKE_BUILD_TYPE Release)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)
find_package(ncnn REQUIRED)

add_library(vs_fsrcnn_vulkan SHARED main.c fsrcnn.cpp)
target_link_libraries(vs_fsrcnn_vulkan ${OpenCV_LIBS} ${VULKAN_LIBRARIES} ${NCNN_LIBRARIES})

install(TARGETS vs_fsrcnn_vulkan LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/vapoursynth)
