
# ## Model Support in This Demo
# 
 * SqueezeNet
 * VGG
 * ResNet
 * MobileNet
# 
# ## Prerequisites
# 
 * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
 * ONNX - `pip3 install onnx`
 * MXNet - `pip3 install mxnet --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
 * numpy - `pip3 install numpy`
 * matplotlib - `pip3 install matplotlib`
# 

# * Run the script: `python3 imagenet_inference.py`

