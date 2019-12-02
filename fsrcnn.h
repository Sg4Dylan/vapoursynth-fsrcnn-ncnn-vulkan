#ifdef __cplusplus
extern "C" {
#endif
void init_ncnn();
void destroy_ncnn();
void filter(const VSFrameRef *, VSFrameRef*, const VSVideoInfo*, const VSAPI*);
#ifdef __cplusplus
}
#endif