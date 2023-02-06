FROM nvcr.io/nvidia/tensorrt:22.07-py3

RUN apt update && apt install -y libgl1-mesa-glx
RUN pip3 install cupy-cuda11x \
                 requests \
                 opencv-python \
                 line_profiler
