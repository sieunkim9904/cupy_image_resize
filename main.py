import numpy as np
import cupy as cp
import cv2
import glob

from _cuda_util import cuda_resize

class OSNet():
    def __init__(self) -> None:
        super().__init__()
        # Define resize shape for output
        self.resize_shape = (640, 640)
    
    def preprocess(self, inputs: list) -> list:
        """
        np.ndarray : (N, H, W, C)
        cp.ndarray : (N, H, W, C)
        list: [ (H, W, C), .... ]
        """
        def _chkList_same_size(lst):
            if len(lst) < 0 :
                return True
            
            return all(ele == lst[0] for ele in lst)

        if _chkList_same_size([input_.shape for input_ in inputs]):
            resized_img = self._preprocess(inputs)
            
            return resized_img
        else: 
            output_array = []
            
            for img in inputs:
                if len(img.shape) == 3 :
                    img = np.expand_dims(img, axis=0)
                resized_img = self._preprocess(img)
                output_array.append(resized_img)
            
            output_array = cp.concatenate(output_array,axis=0)
            output_array = cp.ascontiguousarray(output_array)
            
            return output_array

    def _preprocess(self, input_array: list) -> list:
        """
        resize + scaling + transpose
        in:  (N,H,W,C) , [0,255] , uint8
        out: (N,C,H,W) , [0,1] , float32
        """
        # Create empty cupy array 
        input_array_gpu = cp.empty(shape=input_array.shape, dtype=input_array.dtype)
        
        # Copy host(cpu) data to device(gpu) data
        # For kind= 0: HtoH, 1: HtoD, 2: DtoH, 3: DtoD, 4: unified virtual addressing
        if isinstance(input_array, cp.ndarray): # DtoD 
            cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                   src = int(input_array.data), # src_ptr
                                   size=input_array.nbytes,
                                   kind=3)
        elif isinstance(input_array, np.ndarray):
            cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                   src = input_array.ctypes.data, # src_ptr
                                   size=input_array.nbytes,
                                   kind=1)
        
        # Call cuda_resize function
        output_array = cuda_resize(input_array_gpu,
                                   self.resize_shape) # N,W,H,C

        output_array = cp.transpose(output_array, [0,3,1,2]) # N,C,H,W
        if output_array.dtype != cp.float32:
            output_array = output_array.astype(cp.float32)
        output_array = cp.ascontiguousarray(output_array)
        
        return output_array
    
    def inference(self, input_array: cp.ndarray) -> list:
        # in: RGB <NHWC> raw image batch  , out: <NCHW> resized <N,3,640,640>
        pre_output = self.preprocess(input_array)
        
        return pre_output
    

if __name__ == "__main__":
    # init main object
    engine = OSNet()
    
    # read image and prepare input array
    batch = []
    
    for jpg in sorted(glob.glob("./test_data/1080_1920/*.jpg")):
        img = cv2.imread(jpg)
        batch.append(img)
    
    batch = np.ascontiguousarray(batch)
    print("batch size: ", batch.shape)
    
    # copy to gpu device
    in_arr = cp.array(batch)
    # implement bilinear interpolation with CUDA kernel
    filtered_img = engine.inference(in_arr)
    # copy back to cpu host
    filtered_img = cp.asnumpy(filtered_img)
    
    # output resized output images
    for i in range(filtered_img.shape[0]):
        img = filtered_img[i]
        img = np.transpose(img, [1,2,0])
        cv2.imwrite("./resize_img/media_{}_resize_stride1.jpg".format(i), img)