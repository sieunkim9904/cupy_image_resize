/* Disable comment when profiling with nsys
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
*/

/* Comment this extern"C" {} when profiling with nsys*/
extern "C"{
// return the index of pixel/thread
__device__ long long get_idx(const int n, 
                       const int h, const int H,
                       const int w, const int W, 
                       const int c, const int C)
{
    return (n * H * W * C) + (h * W * C) + (w * C) + c;
}

__device__ double lerp1d(int a, int b, float w)
{
    return fma(w, (float)b, fma(-w,(float)a,(float)a));
}

__device__ float lerp2d(int f00, int f01, int f10, int f11,
                        float centroid_h, float centroid_w )
{
    centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
    centroid_h = (1 + lroundf(centroid_h) - centroid_h)/2;
    
    double r0, r1, r;
    r0 = lerp1d(f00,f01,centroid_w);
    r1 = lerp1d(f10,f11,centroid_w);

    r = lerp1d(r0, r1, centroid_h); //+ 0.00001
    return r;
}

__global__ void GPU_validation(void)
{
    printf("GPU has been activated \n");
}

__global__ void cuResize(const unsigned char* src_img, unsigned char* dst_img,
                         const int src_h, const int src_w,
                         const int tmp_h, const int tmp_w,
                         const int dst_h, const int dst_w,
                         const float scale_h, const float scale_w,
                         const int stride)
{
    // Define shared memory
    // the extern keyword indicates the size of the shared memory is decided dynamically
    // The size is delcared when calling the kernel as a parameter
    __shared__ extern unsigned char tmp_img[];

    int n = blockIdx.y; // batch number
    int C = gridDim.z;  // channel 
    int c = blockIdx.z; // channel number
    // get thread index
    long long idx = get_idx(n, blockIdx.x, gridDim.x, threadIdx.x, blockDim.x, c, C);
    
    // some overhead threads in each image process
    // when thread idx in one image exceed one image size return;
    if (idx % (blockDim.x * gridDim.x * C) >= dst_h * dst_w * C) {
        return;
    }

    // coordinate of one image, not idx of batch image
    int img_coor = idx % (dst_h * dst_w * C);
    int h = img_coor / (dst_w * C);
    int w = img_coor % (dst_w * C) / C;

    // Calcualte the index of filtered image
    float centroid_h = scale_h * (h + 0.5);
    float centroid_w = scale_w * (w + 0.5);
    int tmp_h_idx = lroundf(centroid_h) - 1;
    int tmp_w_idx = lroundf(centroid_w) - 1;

    // boundary check for filtered image
    if (tmp_h_idx < 0) tmp_h_idx = 0;
    else if (tmp_h_idx == tmp_h - 1) tmp_h_idx -= 1;

    if (tmp_w_idx < 0) tmp_w_idx = 0;
    else if (tmp_w_idx == tmp_w - 1) tmp_w_idx -= 1;

    // Load the filtered image to shared memory
    for(int i = threadIdx.x; i / (tmp_w * 2) == 0; i += blockDim.x) {
        long long src_idx = get_idx(n, (tmp_h_idx + (i / tmp_w)) * stride, src_h, (i % tmp_w) * stride, src_w, c, C);
        tmp_img[i] = (unsigned char)src_img[src_idx];
    }

    // Load the filtered image to shared memory using data tiling
    // if (w == 0) {
    //     long long src_idx = get_idx(n, tmp_h_idx * stride, src_h, w, src_w, c, C);
    //     int i = 0;
    //     for (; i < tmp_w; i++) {
    //         tmp_img[i] = (unsigned char)src_img[src_idx];
    //         src_idx += (stride * C);
    //     }

    //     src_idx = get_idx(n, (tmp_h_idx + 1) * stride, src_h, w, src_w, c, C);
    //     for(; i < tmp_w * 2; i++) {
    //         tmp_img[i] = (unsigned char)src_img[src_idx];
    //         src_idx += (stride * C);
    //     }
    // }

    __syncthreads();

    int result = lerp2d(tmp_img[tmp_w_idx],
                        tmp_img[tmp_w_idx + 1],
                        tmp_img[tmp_w_idx + tmp_w],
                        tmp_img[tmp_w_idx + tmp_w + 1],
                        centroid_h, centroid_w);

    // Calculate the index of output image
    long long dst_idx = get_idx(n, h, dst_h, w, dst_w, c, C);
    dst_img[dst_idx] = (unsigned char)result;
}

__global__ void cuResize_free(const unsigned char* src_img, unsigned char* dst_img,
                              const int src_h, const int src_w,
                              const int tmp_h, const int tmp_w,
                              const int dst_h, const int dst_w,
                              const float scale_h, const float scale_w,
                              const int stride)
{
    // Define shared memory
    // the extern keyword indicates the size of the shared memory is decided dynamically
    // The size is delcared when calling the kernel as a parameter
    __shared__ extern unsigned char tmp_img[];

    int batch = blockIdx.y; // batch number
    int C = gridDim.z;      // channel 
    int c = blockIdx.z;     // channel number
    int num = blockIdx.x;   // block number
    int num_row_per_block = dst_h / gridDim.x;

    // calculate the height range for the block on the output image
    int h_start = num * num_row_per_block;
    int h_end = h_start + (num_row_per_block - 1);

    // calcualte the height range for the block on the filtered image on shared memory
    int tmp_h_start = lroundf(scale_h * (h_start + 0.5)) - 1;
    int tmp_h_end = lroundf(scale_h * (h_end + 0.5)) - 1;

    // boundary check for filtered image
    if (tmp_h_start < 0) 
        tmp_h_start = 0;
    if (tmp_h_end == tmp_h - 1) 
        tmp_h_end -= 1;

    // number of rows on the filtered image this block has to process
    int num_tmp_rows = tmp_h_end - tmp_h_start + 2;

    // Load the filtered image to shared memory
    for(int i = threadIdx.x; i < (tmp_w * num_tmp_rows); i += blockDim.x) {
        long long src_idx = get_idx(batch, (tmp_h_start + (i / tmp_w)) * stride, src_h, (i % tmp_w) * stride, src_w, c, C);
        tmp_img[i] = (unsigned char)src_img[src_idx];
    }

    __syncthreads();

    for(int i = threadIdx.x; i < (dst_w * num_row_per_block); i += blockDim.x) {
        int h = h_start + (i / dst_w);
        int w = i % dst_w;

        // tmp_h_start is subtracted so that the hight is referenced correctly to the shared memory index
        float centroid_h = scale_h * (h + 0.5) - tmp_h_start;
        float centroid_w = scale_w * (w + 0.5);
        int tmp_h_idx = lroundf(centroid_h) - 1;
        int tmp_w_idx = lroundf(centroid_w) - 1;

        // boundary check for filtered image
        if (tmp_h_idx < 0) tmp_h_idx = 0;
        else if (tmp_h_idx == tmp_h - 1) tmp_h_idx -= 1;

        if (tmp_w_idx < 0) tmp_w_idx = 0;
        else if (tmp_w_idx == tmp_w - 1) tmp_w_idx -= 1;

        // Execute bilinear interpolation to determin output pixel value
        int result = lerp2d(tmp_img[(tmp_h_idx * tmp_w) + tmp_w_idx],
                            tmp_img[(tmp_h_idx * tmp_w) + tmp_w_idx + 1],
                            tmp_img[((tmp_h_idx + 1) * tmp_w) + tmp_w_idx],
                            tmp_img[((tmp_h_idx + 1) * tmp_w) + tmp_w_idx + 1],
                            centroid_h, centroid_w);

        long long dst_idx = get_idx(batch, h, dst_h, w, dst_w, c, C);
        dst_img[dst_idx] = (unsigned char)result;
    }
}

/* Disable comment when profiling using nsys
int main(){
    int batch = 1;
    int stride = 2;

    int SRC_HEIGHT = 1080;
    int SRC_WIDTH = 1920;
    int SRC_SIZE = batch * SRC_HEIGHT * SRC_WIDTH * 3;

    int DST_HEIGHT = 640;
    int DST_WIDTH = 640;
    int DST_SIZE = batch * DST_HEIGHT * DST_WIDTH * 3;

    int TMP_HEIGHT = ((SRC_HEIGHT - 1) / stride) + 1;
    int TMP_WIDTH = ((SRC_WIDTH - 1) / stride) + 1;

    // cudaStream_t stream1, stream2, stream3, stream4 ;
    cudaStream_t stream1;
    cudaStreamCreate (&stream1);
    
    dim3 dimBlock(640, 1, 1); // maximum threads: 1024
    dim3 dimGrid(640, batch, 3);
    
    unsigned char host_src[SRC_SIZE];
    unsigned char host_dst[DST_SIZE];

    // init src image
    for(int i = 0; i < SRC_SIZE; i++){
        host_src[i] = i % 256;
    }

    float scale_h = (float)TMP_HEIGHT / DST_HEIGHT;
    float scale_w = (float)TMP_WIDTH / DST_WIDTH;

    unsigned char *device_src, *device_dst;
	cudaMalloc((unsigned char **)&device_src, SRC_SIZE* sizeof(unsigned char));
    cudaMalloc((unsigned char **)&device_dst, DST_SIZE* sizeof(unsigned char));
    
	cudaMemcpy(device_src , host_src , SRC_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Allocate CUDA events for time estimation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GPU_validation<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    printf("shared_mem: %lu\n", TMP_WIDTH * 2 * sizeof(unsigned char));

    cuResize<<<dimGrid, dimBlock, TMP_WIDTH * 2 * sizeof(unsigned char), stream1>>>(device_src, device_dst,
                                                                                    SRC_HEIGHT, SRC_WIDTH,
                                                                                    TMP_HEIGHT, TMP_WIDTH,
                                                                                    DST_HEIGHT, DST_WIDTH,
                                                                                    scale_h, scale_w,
                                                                                    stride);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaMemcpy(host_dst, device_dst, DST_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Total time for kernel: %f\n", msecTotal);

	cudaFree(device_src);
	cudaFree(device_dst);

    return 0;
}
*/
}