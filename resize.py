import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
from line_profiler import LineProfiler
import glob

profile = LineProfiler()

bl_Normalize = 0
bl_Trans = 1
pagelock = 1

STRIDE = 2

module = SourceModule("""
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

__global__ void Transpose(unsigned char *odata, const unsigned char *idata,
                            int H, int W)
{
    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long long idx = n * blockDim.x * gridDim.x * C + 
               threadIdx.x * gridDim.x * C +
               blockIdx.x * C+
               c;
    int img_coor = idx % (H*W*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C); // dst idx 
    int w = img_coor % (W*C)/C; // dst idx

    long long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx];
}

__global__ void Transpose_and_normalise(float *odata, const unsigned char *idata,
                            int H, int W)
{
    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long long idx = n * blockDim.x * gridDim.x * C + 
               threadIdx.x * gridDim.x * C +
               blockIdx.x * C+
               c;
    int img_coor = idx % (H*W*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C); // dst idx 
    int w = img_coor % (W*C)/C; // dst idx

    long long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx]/255.0;
}

__global__ void cuResize(const unsigned char* src_img, unsigned char* dst_img,
                         const int src_h, const int src_w,
                         const int tmp_h, const int tmp_w,
                         const int dst_h, const int dst_w,
                         const float scale_h, const float scale_w,
                         const int stride)
{
    // Define shared memory
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

    // // Load the filtered image to shared memory
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
    __shared__ extern unsigned char tmp_img[];

    int batch = blockIdx.y; // batch number
    int C = gridDim.z;      // channel 
    int c = blockIdx.z;     // channel number
    int num = blockIdx.x;   // block number
    int num_row_per_block = dst_h / gridDim.x;

    int h_start = num * num_row_per_block;
    int h_end = h_start + (num_row_per_block - 1);

    int tmp_h_start = lroundf(scale_h * (h_start + 0.5)) - 1;
    int tmp_h_end = lroundf(scale_h * (h_end + 0.5)) - 1;

    // boundary check for filtered image
    if (tmp_h_start < 0) 
        tmp_h_start = 0;
    if (tmp_h_end == tmp_h - 1) 
        tmp_h_end -= 1;

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

        float centroid_h = scale_h * (h + 0.5) - tmp_h_start;
        float centroid_w = scale_w * (w + 0.5);
        int tmp_h_idx = lroundf(centroid_h) - 1;
        int tmp_w_idx = lroundf(centroid_w) - 1;

        // boundary check for filtered image
        if (tmp_h_idx < 0) tmp_h_idx = 0;
        else if (tmp_h_idx == tmp_h - 1) tmp_h_idx -= 1;

        if (tmp_w_idx < 0) tmp_w_idx = 0;
        else if (tmp_w_idx == tmp_w - 1) tmp_w_idx -= 1;

        int result = lerp2d(tmp_img[(tmp_h_idx * tmp_w) + tmp_w_idx],
                            tmp_img[(tmp_h_idx * tmp_w) + tmp_w_idx + 1],
                            tmp_img[((tmp_h_idx + 1) * tmp_w) + tmp_w_idx],
                            tmp_img[((tmp_h_idx + 1) * tmp_w) + tmp_w_idx + 1],
                            centroid_h, centroid_w);
        // int result = 0;

        long long dst_idx = get_idx(batch, h, dst_h, w, dst_w, c, C);
        dst_img[dst_idx] = (unsigned char)result;
    }
}
""")

cuResizeKer = module.get_function("cuResize")
cuResizeKer_free = module.get_function("cuResize_free")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")

@profile
def gpu_resize(input_img: np.ndarray, shape=(640,640)):
    """
    Resize the batch image to (608,608) 
    and Convert NHWC to NCHW
    pass the gpu array to normalize the pixel ( divide by 255)

    Application oriented

    input_img : batch input, format: NHWC , recommend RGB. *same as the NN input format 
                input must be 3 channel, kernel set ChannelDim as 3.
    out : batch resized array, format: NCHW , same as intput channel
    """
    # ========= Init Params =========
    stream = cuda.Stream()

    # convert to array
    batch, src_h, src_w, channel = input_img.shape
    assert channel == 3
    dst_h, dst_w = shape[0], shape[1]
    DST_SIZE = dst_h* dst_w* 3
    # Mem Allocation
    # input memory
    
    if pagelock: #  = = = = = = Pagelock memory = = = = = = 
        inp = {"host":cuda.pagelocked_zeros(shape=(batch,src_h,src_w,channel),
                                            dtype=np.uint8,
                                            mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        # inp = {"host":cuda.pagelocked_empty_like(input_img,
                                                #  mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        # print(inp["host"].shape,input_img.shape)
        inp["host"][:,:src_h,:src_w,:] = input_img
    else: #  = = = = = = Global memory = = = = = = 
        inp = {"host":input_img}

    inp["device"] = cuda.mem_alloc(inp["host"].nbytes)
    cuda.memcpy_htod_async(inp["device"], inp["host"],stream)




    # output data
    if pagelock: #  = = = = = = Pagelock emory = = = = = = 
        out = {"host":cuda.pagelocked_zeros(shape=(batch,dst_h,dst_w,channel), 
                                        dtype=np.uint8,
                                        mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
    else: #  = = = = = = Global memory = = = = = = 
        out = {"host":np.zeros(shape=(batch,dst_h,dst_w,channel), dtype=np.uint8)}  # N H W C
    
    out["device"] = cuda.mem_alloc(out["host"].nbytes)
    cuda.memcpy_htod_async(out["device"], out["host"],stream)

    import time
    time.sleep(5)
    
    #Transpose (and Normalize)
    if bl_Normalize or bl_Trans:
        if bl_Normalize:
            if pagelock:
                trans = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
                                                      dtype=np.float32,
                                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N C H W
            else:
                trans = {"host":np.zeros(shape=(batch,channel,dst_h,dst_w), dtype=np.float32)}  # N C H W
        else:
            if pagelock:
                trans = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
                                                      dtype=np.uint8,
                                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
            else:
                trans = {"host":np.zeros(shape=(batch,channel,dst_h,dst_w), dtype=np.uint8)}  # N C H W

        trans["device"] = cuda.mem_alloc(trans["host"].nbytes)
        cuda.memcpy_htod_async(trans["device"], trans["host"],stream)

    # Calculate filtered image size to be stored in shared memory
    tmp_h = ((src_h - 1) // STRIDE) + 1
    tmp_w = ((src_w - 1) // STRIDE) + 1

    # init resize , store kernel in cache
    cuResizeKer_free(inp["device"], out["device"], 
               np.int32(src_h), np.int32(src_w),
               np.int32(tmp_h), np.int32(tmp_w),
               np.int32(dst_h), np.int32(dst_w),
               np.float32(tmp_h/dst_h), np.float32(tmp_w/dst_w),
               np.int32(STRIDE),
               block=(1024, 1, 1),
               grid=(16, batch, 3),
               shared=(tmp_w * 35),
               stream=stream)
    
    # ========= Testing =========

    for _ in range(1):
        cuResizeKer_free(inp["device"], out["device"], 
                    np.int32(src_h), np.int32(src_w),
                    np.int32(tmp_h), np.int32(tmp_w),
                    np.int32(dst_h), np.int32(dst_w),
                    np.float32(tmp_h/dst_h), np.float32(tmp_w/dst_w),
                    np.int32(STRIDE),
                    block=(1024, 1, 1),
                    grid=(16, batch, 3),
                    shared=(tmp_w * 35))

    # ========= Copy out result =========

    if bl_Normalize:
        TransNorKer(trans["device"],out["device"],
                    block=(32, 32, 1),
                    grid=(19,19,3*batch))
        cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)
        stream.synchronize()
        return trans["host"]
    elif bl_Trans:
        TransposeKer(trans["device"],out["device"],
                    np.int32(dst_h), np.int32(dst_w),
                    block=(1024, 1, 1),
                    grid=(int(DST_SIZE/3//1024)+1,batch,3))
        cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)
        stream.synchronize()
        return trans["host"]
    else:
        cuda.memcpy_dtoh_async(out["host"], out["device"],stream)
        stream.synchronize()
        return out["host"]

if __name__ == "__main__":
    batch = []
    
    i = 0
    for jpg in sorted(glob.glob("./test_data/1080_1920/*.jpg")):
        img = cv2.imread(jpg)
        batch.append(img)
        i += 1
        if (i==50):
            break
    
    batch = np.ascontiguousarray(batch)
    print("batch size: ", batch.shape)
    
    output_img = gpu_resize(batch,shape = (640,640))
    if bl_Normalize or bl_Trans:
        output_img = np.transpose(output_img,[0,2,3,1])
        
    for i in range(output_img.shape[0]):
        img = output_img[i]
        cv2.imwrite("../resize_img/media_{}_resize_tiling_stride1.jpg".format(i), img)
