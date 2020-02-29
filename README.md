
# Vapoursynth FSRCNN Vulkan

ncnn implementation of fsrcnn converter for vapoursynth.

## Build for Windows x64

install Visual Studio Community 2019

> Start → Programs → Visual Studio 2019 → 64 Native Tools Command Prompt for VS 2019

build vs-fsrcnn-vulkan

> mkdir build  
> cd build  
> cmake -G"NMake Makefiles" ..  
> nmake

## Usage

> clip = mvf.ToRGB(clip, depth=32, sample=1)  #Convert to RGB32  
> clip = core.fsrcnn.Filter(clip)

## Reference

 - [VapourSynth-Waifu2x-caffe](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Waifu2x-caffe/)
 - [FSRCNN-PyTorch](https://github.com/yjn870/FSRCNN-pytorch)
