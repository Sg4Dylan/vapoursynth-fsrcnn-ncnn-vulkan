#include <stdio.h>
#include <wchar.h>

// VapourSynth Library
#include "VapourSynth.h"
#include "VSHelper.h"

// Image decode and encode
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>

// NCNN Library
#include "platform.h"
#include "net.h"
#include "gpu.h"

// FSRCNN Model
#include "fsrcnn.id.h"
#include "fsrcnn.mem.h"

// Export Function
#include "fsrcnn.h"

ncnn::Net fsrcnn;

void fsrcnn_process(cv::Mat&, cv::Mat&, int, int);

void init_ncnn() {
	ncnn::create_gpu_instance();
	fsrcnn.set_vulkan_device(0);
	fsrcnn.opt.use_vulkan_compute = true;
	fsrcnn.load_param(fsrcnn_param_bin);
	fsrcnn.load_model(fsrcnn_bin);
}

void destroy_ncnn() {
	fsrcnn.clear();
	ncnn::destroy_gpu_instance();
}

void filter(const VSFrameRef* src, VSFrameRef* dst, const VSVideoInfo* vi, const VSAPI* vsapi) {

	const int srcwidth = vsapi->getFrameWidth(src, 0);
	const int srcheight = vsapi->getFrameHeight(src, 0);
	const int dstwidth = vsapi->getFrameWidth(dst, 0);
	const int dstheight = vsapi->getFrameHeight(dst, 0);
	const float* srcpR = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0));
	const float* srcpG = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1));
	const float* srcpB = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2));
	float* VS_RESTRICT dstpR = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));
	float* VS_RESTRICT dstpG = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1));
	float* VS_RESTRICT dstpB = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2));

	cv::Mat original_image;
	{
		cv::Mat src_r(srcheight, srcwidth, CV_32FC1, (void*)srcpR);
		cv::Mat src_g(srcheight, srcwidth, CV_32FC1, (void*)srcpG);
		cv::Mat src_b(srcheight, srcwidth, CV_32FC1, (void*)srcpB);
		cv::Mat src_raw[3] = { src_r, src_g, src_b };
		cv::merge(src_raw, 3, original_image);
	}

	cv::Mat src_y;
	cv::Mat img_ycrcb;
	{
		cv::cvtColor(original_image, img_ycrcb, cv::COLOR_RGB2YCrCb);
		cv::Mat img_src_chan[3];
		cv::split(img_ycrcb, img_src_chan);
		src_y = img_src_chan[0];
		src_y = src_y.clone();
	}

	cv::Mat dst_y(dstheight, dstwidth, CV_32FC1, cv::Scalar(0));
	fsrcnn_process(src_y, dst_y, srcwidth, srcheight);

	src_y.release();

	cv::Mat target_image;
	{
		cv::Mat img_dst_rgb;
		cv::Mat img_dst_ycrcb;
		cv::resize(original_image, img_dst_rgb, cv::Size(dstwidth, dstheight), 0, 0, cv::INTER_CUBIC);
		original_image.release();
		cv::cvtColor(img_dst_rgb, img_dst_ycrcb, cv::COLOR_RGB2YCrCb);
		img_dst_rgb.release();
		cv::Mat img_dst_chan[3];
		cv::split(img_dst_ycrcb, img_dst_chan);
		img_dst_chan[0] = dst_y;
		cv::Mat merged_image;
		cv::merge(img_dst_chan, 3, merged_image);
		cv::cvtColor(merged_image, target_image, cv::COLOR_YCrCb2RGB);
	}

	dst_y.release();

	{
		cv::Mat dst_chan[3];
		cv::split(target_image, dst_chan);
		auto stride = dstwidth * 4;
		for (int i = 0; i < dstheight; i++) {
			memcpy((uint8_t*)dstpR + stride * i, dst_chan[0].data + stride * i, stride);
			memcpy((uint8_t*)dstpG + stride * i, dst_chan[1].data + stride * i, stride);
			memcpy((uint8_t*)dstpB + stride * i, dst_chan[2].data + stride * i, stride);
		}
	}

	target_image.release();
}

int new_from_gray(float* gray, int w, int h, ncnn::Mat& m, ncnn::Allocator* allocator)
{
	m.create(w, h, 1, 4u, allocator);
	if (m.empty())
		return -100;
	float* ptr = m;
	int size = w * h;
	int remain = size;
	for (; remain > 0; remain--)
	{
		*ptr = *gray;
		gray++;
		ptr++;
	}
	return 0;
}

void new_to_gray(const ncnn::Mat& m, float* gray)
{
	const float* ptr = m;
	int size = m.w * m.h;
	int remain = size;
	for (; remain > 0; remain--)
	{
		*gray = min(max(*ptr, 0), 255);
		gray++;
		ptr++;
	}
}

void fsrcnn_process(cv::Mat& img_y, cv::Mat& out_y, int w, int h) {
	ncnn::VkMat in_gpu;
	ncnn::VkMat out_gpu;
	ncnn::Mat inimage = in_gpu.mapped();
	ncnn::Mat outimage = out_gpu.mapped();

	new_from_gray((float *)img_y.data, w, h, inimage, (ncnn::Allocator*)0);

	{
		ncnn::Extractor ex = fsrcnn.create_extractor();
		ex.set_light_mode(true);
		ex.input(fsrcnn_param_id::LAYER_image, inimage);
		ex.extract(fsrcnn_param_id::LAYER_output, outimage);
	}

	new_to_gray(outimage, (float*)out_y.data);
}
