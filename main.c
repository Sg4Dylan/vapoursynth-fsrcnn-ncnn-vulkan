// VapourSynth Library
#include "VapourSynth.h"
#include "VSHelper.h"

// FSRCNN
#include "fsrcnn.h"
#include "stdio.h"

typedef struct {
	VSNodeRef* node;
	VSVideoInfo vi;
	int scale;
	int srcInterleaved, dstInterleaved;
} FilterData;

static void VS_CC filterInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi) {
	FilterData* d = (FilterData*)*instanceData;
	vsapi->setVideoInfo(&d->vi, 1, node);
	fprintf(stderr, "[FSRCNN-Vulkan] NCNN Init...\n");
	init_ncnn();
}

static const VSFrameRef* VS_CC filterGetFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
	FilterData* d = (FilterData*)*instanceData;
	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, d->node, frameCtx);
	}
	else if (activationReason == arAllFramesReady) {
		const VSFrameRef* src = vsapi->getFrameFilter(n, d->node, frameCtx);
		if (d->vi.format->colorFamily == cmRGB && d->vi.format->sampleType == 1 && d->vi.format->bytesPerSample == 4) {
			VSFrameRef* dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);
			filter(src, dst, &d->vi, vsapi);
			vsapi->freeFrame(src);
			return dst;
		}
		else {
			return src;
		}
	}

	return 0;
}

static void VS_CC filterFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
	fprintf(stderr, "[FSRCNN-Vulkan] NCNN Destory...\n");
	destroy_ncnn();
	FilterData* d = (FilterData*)instanceData;
	vsapi->freeNode(d->node);
	free(d);
}

static void VS_CC filterCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
	FilterData d;
	FilterData* data;
	fprintf(stderr, "[FSRCNN-Vulkan] Filter create.\n");

	d.node = vsapi->propGetNode(in, "clip", 0, 0);
	d.vi = *vsapi->getVideoInfo(d.node);
	if (d.vi.format->colorFamily == cmRGB && d.vi.format->sampleType == 1 && d.vi.format->bytesPerSample == 4) {
		d.vi.width *= 2;
		d.vi.height *= 2;
	}

	data = malloc(sizeof(d));
	*data = d;

	vsapi->createFilter(in, out, "FSRCNN-Vulkan", filterInit, filterGetFrame, filterFree, fmParallel, 0, data, core);

}


//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
	configFunc("works.acg.vulkan.fsrcnn", "fsrcnn", "VapourSynth FSRCNN Vulkan", VAPOURSYNTH_API_VERSION, 1, plugin);
	registerFunc("Filter", "clip:clip;", filterCreate, 0, plugin);
}
