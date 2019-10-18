# Video detection
This project aims to detect objects in a video stream then generate data that can be consumed by various clients e.g. SPA apps, chat bots etc.

It comprises of a python script that uses pretrained tensorflow models for out of the box inference.

## Handler Service Setup
It is possible to transmit the results of a detection to a detection handling service for more processing - e.g. visualizing boxes, database storage.

To (re)generate the client, server grpc code, follow the steps below.

* Download or clone the [tensorflow core repo](https://github.com/tensorflow/tensorflow)
* In this project's `proto` dir, create a soft link named `tensorflow` to the `[tensorflow-repo]/tensorflow` dir
* cd to `proto/generated`
* Generate the code - `python -m grpc_tools.protoc -I ../ --grpc_python_out=. --python_out=. ../detection_handler.proto`

## Resources
- <https://towardsdatascience.com/detecting-pikachu-in-videos-using-tensorflow-object-detection-cd872ac42c1d>
- <https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb>
- https://www.semantics3.com/blog/a-simplified-guide-to-grpc-in-python-6c4e25f0c506/
- other credits are given in code and commit comments

## Licensing
MIT

## Contributors
Eric Njogu
