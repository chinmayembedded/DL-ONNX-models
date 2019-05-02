
 
# ## Installation
# 

 * Download the model from - (https://github.com/onnx/models/tree/master/models/face_recognition/ArcFace) and name it resnet100.onnx 
 * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
 * ONNX - `pip install onnx`
 * MXNet - `pip install mxnet --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
 * numpy - `pip install numpy`
 * matplotlib - `pip install matplotlib`
 * OpenCV - `pip install opencv-python`
 * Scikit-learn - `pip install scikit-learn`
 * EasyDict - `pip install easydict`
 * Scikit-image - `pip install scikit-image`
# 
# Also the following scripts and folders (included in the repo) must be present in the same folder as this notebook:
 * `mtcnn_detector.py` (Performs face detection as a part of preprocessing)
 * `helper.py` (helper script for face detection)

# * Run the script: `python3 arcface_inference.py`

