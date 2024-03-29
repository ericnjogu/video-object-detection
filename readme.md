# Video detection
This project aims to detect objects in a video stream then generate data that can be consumed by various clients e.g. SPA apps, chat bots etc.

It comprises of a python script that uses pretrained tensorflow models for out of the box inference.

## Envisioned Architecture
![Architectural diagram](video-object-detection-arch.jpg "Architectural diagram")
The system is conceived as micro services which will communicate via publish/subscribe queues.

A **source** could be a video file, a network stream, a directory of images.
The service that reads the source is also aware of the instance name from the configuration. The source creates a message with a unique ID.

The **source** pushes the message into a channel where services that communicate with **model services (models)** receive messages.

At the time of pushing the message, a shared counter, say counter <msg-id>-in, with identified by the message ID is created with the total number of **model services**
 that received the message for processing.

After each **model service** produces some detection results, those results are placed on a publish/subscribe channel where all related results are aggregated.
The aggregation can be handled by via another shared counter, say counter <msg-id>-out which is incremented by each model service upon receiving the message from the **source**.

Once the two counters are equal, the message is placed on another broadcast channel where other services take the message and deliver it appropriately.

## Setup
A conda environment is created first and when activated, additional pip packages are installed.
 - Run `conda env create -f env.yaml` to setup a conda environment
 - activate environment `source activate object_detection_mini`
 - install pip packages `pip install -r requirements.txt  --no-deps`

## Running with Tensorflow Serving (Docker)
- [setup](https://www.tensorflow.org/tfx/serving/docker) and start the tensor flow serving docker container

   First time run:

   `sudo docker run --name ssd_mobilenet_v1_coco -p 8501:8501 -p 8500:8500 --mount type=bind,source=/home/mugo/downloaded-tensorflow-models/ssd_mobilenet_v1_coco_2017_11_17/,target=/models/ssd_mobilenet_v1_coco -e MODEL_NAME=ssd_mobilenet_v1_coco -t tensorflow/serving`

   Every other run:

   `sudo docker start -a ssd_mobilenet_v1_coco`

   Stopping:

   `sudo docker stop ssd_mobilenet_v1_coco`

- Run video detection script

 `bash run_with_env.sh python detect_video_stream_tf_serving.py ~/Videos/train-passenger-foot-stuck.mp4  ~/tensorflow-models-repo/research/object_detection/data/mscoco_complete_label_map.pbtxt 8500 ssd_mobilenet_v1_coco predictions --cutoff 70`

## Testing
Individual tests can be run like this:

`bash run_with_env.sh pytest detect_video_stream_utils_test.py --disable-warnings --log-cli-level=DEBUG`

## Related Projects
- https://github.com/kunadawa/object-detection-event-web-server
- https://github.com/kunadawa/object-detection-react-app
- https://github.com/ericnjogu/object-detection-protos


## Resources
- <https://towardsdatascience.com/detecting-pikachu-in-videos-using-tensorflow-object-detection-cd872ac42c1d>
- <https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb>
- https://www.semantics3.com/blog/a-simplified-guide-to-grpc-in-python-6c4e25f0c506/
- https://medium.com/innovation-machine/deploying-object-detection-model-with-tensorflow-serving-part-3-6a3d59c1e7c0
- https://medium.com/@yuu.ishikawa/how-to-show-signatures-of-tensorflow-saved-model-5ac56cf1960f
- other credits are given in code and commit comments.


## Licensing
MIT

## Contributors
Eric Njogu
