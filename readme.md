# Video detection
This project aims to detect objects in a video stream then generate data that can be consumed by various clients e.g. SPA apps, chat bots etc.

It comprises of a python script that uses pretrained tensorflow models for out of the box inference.

## Conda Setup
Run `conda create --name <env> --file package-list-linux-64.txt` to setup a conda environment

## Handler Service Setup
It is possible to transmit the results of a detection to a detection handling service for more processing - e.g. visualizing boxes, database storage.

To (re)generate the client, server grpc code, follow the steps below.

* Generate the code - `python -m grpc_tools.protoc -I proto --grpc_python_out=proto/generated/ --python_out=proto/generated proto/detection_handler.proto `

## Running
- Download or clone the [tensorflow core repo](https://github.com/tensorflow/tensorflow). This will make available several models that can be used to run inferences.
- add the generated python code to the python path

   `export PYTHONPATH=.:[qualified/path/to/]proto/generated/`
- start the sample standard output detection handler

  `python samples/stdout_detection_handler.py`
- Attempt to detect objects in a video. This example uses the ssd_mobilenet_v1_coco_2017_11_17 model in the downloaded tensorflow repo

  `python detect_video_stream.py ~/Videos/train-passenger-foot-stuck.mp4 ~/tensorflow-models-repo/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb ~/tensorflow-models-repo/research/object_detection/data/mscoco_complete_label_map.pbtxt --cutoff 70 --handler_port=50002`
  
## Related Projects
- https://github.com/kunadawa/object-detection-event-web-server
- https://github.com/kunadawa/object-detection-react-app


## Resources
- <https://towardsdatascience.com/detecting-pikachu-in-videos-using-tensorflow-object-detection-cd872ac42c1d>
- <https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb>
- https://www.semantics3.com/blog/a-simplified-guide-to-grpc-in-python-6c4e25f0c506/
- other credits are given in code and commit comments

## Licensing
MIT

## Contributors
Eric Njogu
