import cupy as cp
import numpy as np

# Call the .cu CUDA file using RawModule of cupy
with open('cuResize.cu', 'r') as reader:
    module = cp.RawModule(code=reader.read())
    
cuResizeKer = module.get_function("cuResize_free")

# Define stride for image filtering
STRIDE = 2

def cuda_resize(inputs: cp.ndarray, # src: (N,H,W,C)
                shape: tuple, # (dst_h, dst_w)
                output: cp.ndarray=None): # dst: (N,H,W,C)
    
    out_dtype = cp.uint8
    
    N, src_h, src_w, C = inputs.shape
    assert C == 3 # resize kernel only accept 3 channel tensors.
    
    # Calculate filtered image size to be stored in shared memory
    tmp_h = ((src_h - 1) // STRIDE) + 1
    tmp_w = ((src_w - 1) // STRIDE) + 1
    # Define output size
    dst_h, dst_w = shape
    DST_SIZE = dst_h * dst_w * C
    
    # define kernel configs
    block = (1024, 1, 1)
    grid = (16, N, C)
    
    if(dst_w > 1024):
        block = (1024, 1, 1)
        grid = (int(DST_SIZE / 3 // 1024) + 1, N, 3)
    
    if output:
        assert output.dtype == out_dtype
        assert output.shape[1] == dst_h
        assert output.shape[2] == dst_w
    else:
        out_shape = (N, dst_h, dst_w, C)
        output = cp.empty(tuple(out_shape), dtype = out_dtype)
    
    # Run CUDA kernel
    # __call__ (self, grid, block, args=(), shared_mem=0, stream=None)
    gpu_times = []
    for _ in range(0, 10):
        # Create cuda event to measure the time taken to run the resize kernel
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        # Start record
        start_gpu.record()
        with cp.cuda.stream.Stream() as stream:
            cuResizeKer(grid, block,
                        (inputs, output,
                        cp.int32(src_h), cp.int32(src_w),
                        cp.int32(tmp_h), cp.int32(tmp_w),
                        cp.int32(dst_h), cp.int32(dst_w),
                        cp.float32(tmp_h/dst_h), cp.float32(tmp_w/dst_w),
                        cp.int32(STRIDE)),
                        shared_mem=(tmp_w * inputs.dtype.itemsize * 35) # Determine the size of shared memory to be used for this kernel
            )
            # check for memory errors
            try:
                stream.synchronize()
            except:
                print("Encounter CUDA Memory Error")
        # End Record
        end_gpu.record()
        end_gpu.synchronize()
        gpu_times.append(cp.cuda.get_elapsed_time(start_gpu, end_gpu))
    # Take the average of recorded time
    gpu_avg_time = np.average(gpu_times)
    print(f"{gpu_avg_time:.6f} s")
    
    return output
