# This sample uses a Caffe ResNet50 Model to create a TensorRT Inference Engine
import sys, os
import random

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
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
TRT_LOGGER = trt.Logger(trt.Logger.DEBUG)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

# The Caffe path is used for Caffe2 models.
def build_engine_caffe(model_file, deploy_file):
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        config.max_workspace_size = 1 << 30 # 1 GiB
        # Load the Caffe model and parse it in order to populate the TensorRT network.
        # This function returns an object that we can query to find tensors by name.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        # For Caffe, we need to manually mark the output of the network.
        # Since we know the name of the output tensor, we can find it in model_tensors.
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        
        return builder.build_engine(network, config)

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        resized_img = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE))
        
        return resized_img.ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image).convert('RGB')))
    return test_image

def test(test_path='./test'):
    # Build a TensorRT engine.
    with build_engine_caffe(ModelData.MODEL_PATH, ModelData.DEPLOY_PATH) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            files = sorted(os.listdir(test_path))
            
            for test_file in files:
                test_image = os.path.join(test_path, test_file)
                
                # Load a normalized test case into the host input page-locked buffer.
                test_case = load_normalized_test_case(test_image, h_input)
                # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
                # probability that the image corresponds to that label
                do_inference(context, h_input, d_input, h_output, d_output, stream)

                # We use the highest probability as our prediction. Its index corresponds to the predicted label.
                pred = np.argmax(h_output)
                print('test file: {}, predict - label: {}, scores: {:.4f}'.format(test_file, pred, h_output[pred]))

if __name__ == '__main__':
    test()
