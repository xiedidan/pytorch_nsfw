import sys, os
import random

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
from PIL import Image
import numpy as np

class ModelData(object):
    MODEL_PATH = "caffe_model/resnet_50_1by2_nsfw.caffemodel"
    DEPLOY_PATH = "caffe_model/deploy.prototxt"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

def build_engine_caffe(model_file, deploy_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
        config.max_workspace_size = 1 << 30 # 1 GiB
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        
        return builder.build_engine(network, config)

def load_normalized_test_case(test_image, pagelocked_buffer):
    def normalize_image(image):
        c, h, w = ModelData.INPUT_SHAPE
        
        # pre-processing from original caffe pipeline
        resized_img = np.asarray(image.resize((w, h), Image.BILINEAR))
        nor_img = resized_img - np.array([104, 117, 123])
        bgr_img = nor_img[:,:,::-1]
        
        return bgr_img.transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image).convert('RGB')))
    return test_image

def test(test_path='./test'):
    with build_engine_caffe(ModelData.MODEL_PATH, ModelData.DEPLOY_PATH) as engine:
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        
        with engine.create_execution_context() as context:
            files = sorted(os.listdir(test_path))
            
            for test_file in files:
                test_image = os.path.join(test_path, test_file)
                
                test_case = load_normalized_test_case(test_image, h_input)
                do_inference(context, h_input, d_input, h_output, d_output, stream)

                pred = np.argmax(h_output)
                print('test file: {}, predict - label: {}, nsfw score: {:.4f}'.format(test_file, pred, h_output[1]))

if __name__ == '__main__':
    test()
