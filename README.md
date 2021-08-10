# pytorch_nsfw
convert yahoo [open_nsfw](https://github.com/yahoo/open_nsfw) caffe model to pytorch and trt models

## TensorRT

```caffe_infer.py``` loads original caffe model with TensorRT directly. There are model loader and test code.  
Create a ```test``` folder and place images in it, then run ```python caffe_infer.py``` for test.  
