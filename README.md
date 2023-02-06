# Image Resize Project

This project's objective is to increase the efficiency and speed of resizing images using NVIDIA GPU and CUDA. The main concept is to enable shared memory () to store image data from the global memory so that data access from threads are acheived faster. From the image data stored in the shared memory, the output pixel values are calculated using bilinear interpolation algorithm. 

## Environment
* Ubuntu 20.04.4 LTS
* GPU NVIDIA GeForce RTX 3090
* GPU Architecture: Ampere
* CUDA Version: 11.7
* CUDA Driver Version: 515.76
* Docker
* Python 3.8.10

## Build & Run
```bash
docker build -t cuda_example .
docker run --rm -it --privileged --runtime nvidia -v $PWD:/py -w /py cuda_example bash
python3 osnet.py
```

Command for creating profiling results
```bash
# inside docker image
# profiling using nsight system profiler
# the result file (should be named report_.nsys-rep) can be opened and analyzed using nsight system
$ sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
$ nvcc cuResize.cu -o cuResize.o
$ nsys profile ./cuResize.o

# profiling using nsight compile profiler
# the result file (metrics.ncu-rep in this case) can be opened and analyzed using nsight compute
$ ncu -o metrics /bin/python3 resize.py
```

## Methodology
The biggest difference between the original code and this project, is the utilization of shared memory. The image below illustrates the original methodology. 

(Original method image insert)

Originally, the grid and block sizes are determined so that each thread covers 1 pixel in the output image. According to the scale between the input image size and desired output size, the position of the output pixel on the input image will be calculated. Depending on the position, the closest 4 pixels index is found. Using these 4 pixels and bilinear interpolation algorithm, the pixel value for the output pixel is calculated. Since the kernel fetches the 4 pixel values of the input image from the global memory, this process can be carried out faster using shared memory.

(New method image insert)

This image depicts the new method. Although shared memory is much faster than global memory, the size of it is limited (theoretically GeForce RTX 3090 has 128kB of memory for both shared memory and L1 cache together, but through some experiments, the suggested shared memory limit size is ~= 46kB). Therefore, not all of the input image can be stored in the shared memory. For efficiency, the input image is filtered and the filtered data is stored in the shared memory. The filtering size can be determined through the `STRIDE` variable in [`_cuda_util.py`](_cuda_util.py). For the current project, the STRIDE is set to 1.

(STRIDE value examples images)

After all threads updated the filtered image in the shared memory, the output pixel value is calculated using the closest 4 pixel in the *filtered image* using bilinear interpolation.

More details of the code will be provided below.

### Minor changes
Originally, padding was enabled for the output image when the input image ratio(ex. 9:16) is different to the output image ratio (ex. 1:1). However, now padding is not implemented and the output image is flexibily fitted into the output image size in full (no black paddings).

## Detailed description of code files

[`osnet.py`](osnet.py): This file contains the main function. 

[`_cuda_util.py`](_cuda_util.py): The `STRIDE` varible is declared in this file. The cuda_resize() function calculates the filtered image size, kernel configurations (grid and block size), and calles the kernels.

[`cuResize.cu`](cuResize.cu): The kernel code is implemented. When using nsys for profiling, remember to comment and uncomment some lines (described in the code as comments). 

[`resize.py`](resize.py): This python file is used for profiling using nsight compile. 

### Kernel description for cuResize_free function in cuResize.cu
For this project and this kernel, it is assumed that the input image size is [500, 1080, 1920, 3], which means 500 images with 1080 height and 1920 width of RGB(3 channels). To maximize the usage of shared memory, each image of the output is divided into 16 parts (each part will consist of 40(= 640 / 16) lines of pixel). Hence, the grid size of the kernel is determined to be (16, batch, 3). Since each block of threads share the same shared memory, and each block now has to cover a part of the output image ( 40 lines * 640 pixels in a line = total number of pixels to be figured out in one block). Each threads in the block (1024 in total) cooperate to store the filtered image section to the shared image using the for loop. Then again using the for loop, each thread execute bilinear interpolation to calculate the pixel value of the output image. 
