# imports
import logging
import argparse
import sys
import json
import cv2
import numpy as np
import object_detection.utils.label_map_util as label_utils
from datetime import datetime as dt
import grpc
import google.protobuf.json_format as json_format
import tensorflow as tf

from proto.generated import detection_handler_pb2_grpc, detection_handler_pb2
import video_object_detection as obj_detect
import detect_video_stream_utils
from proto.generated.tensorflow_serving.apis import predict_pb2, model_pb2, prediction_service_pb2_grpc
import tempfile

CUT_OFF_SCORE = 90.0
SAMPLE_RATE = 5
HANDLER_PORT = 50051


def detect_video_stream(args):
    """ detect objects in video stream """
    # setup grpc comms to detection handler
    handler_port = detect_video_stream_utils.determine_handler_port(args.handler_port, HANDLER_PORT)
    url = f'localhost:{handler_port}'
    logging.debug(f'connecting to handler at {url}')
    detection_handler_channel = grpc.insecure_channel(url)
    detection_handler_stub = detection_handler_pb2_grpc.DetectionHandlerStub(detection_handler_channel)

    # setup grpc comms to tensorflow serving
    tensorflow_serving_port = args.tensorflow_serving_port
    url = f'localhost:{tensorflow_serving_port}'
    logging.debug(f'connecting to tensorflow serving at {url}')
    tensorflow_serving_channel = grpc.insecure_channel(url)
    tensorflow_serving_stub = prediction_service_pb2_grpc.PredictionServiceStub(tensorflow_serving_channel)

    # generate dict from labels
    category_index = label_utils.create_category_index_from_labelmap(args.path_to_label_map, use_display_name=True)
    # logging.debug(f"category_index: {category_index}")
    # TODO validate args
    # determine sample rate
    sample_rate = detect_video_stream_utils.determine_samplerate(args.samplerate, SAMPLE_RATE)
    cap = detect_video_stream_utils.determine_source(args, cv2.VideoCapture)
    float_map = {'frame_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'frame_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH)}
    total_frame_count = 0
    prediction_frame_count = 0
    start_time = dt.now().timestamp()
    cut_off_score = detect_video_stream_utils.determine_cut_off_score(args, default_cut_off=CUT_OFF_SCORE)
    logging.debug(f"using a cut off score of {cut_off_score}")
    model_name = args.model_name
    # loop over frames in video
    # adapted from https://github.com/juandes/pikachu-detection/blob/master/detection_video.py
    while cap.isOpened():
        frame_returned, frame = cap.read()
        # only consider frames that are a multiple of the sample rate
        if frame is not None and total_frame_count % sample_rate == 0:
            frame_bgr2rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_to_array = np.expand_dims(frame_bgr2rgb, axis=0)
            # run inference
            prediction_request = predict_pb2.PredictRequest(
                        model_spec=model_pb2.ModelSpec(name=model_name),
                        inputs={'inputs': tf.compat.v1.make_tensor_proto(img_to_array)})
            prediction_response =  tensorflow_serving_stub.Predict(prediction_request, 10.0)
            output_dict = prediction_response.outputs
            output_dict = detect_video_stream_utils.filter_detection_output_tf_serving(output_dict, cut_off_score)
            if len(output_dict['detection_boxes']) > 0:
                #logging.debug(f'filtered output: {output_dict}')
                detection_boxes = detection_handler_pb2.float_array(numbers=output_dict['detection_boxes'].ravel(),
                                                                    shape=output_dict['detection_boxes'].shape)
                filtered_category_index = detect_video_stream_utils.class_names_from_index(
                    output_dict['detection_classes'], category_index)
                source = detect_video_stream_utils.determine_source_name(args.source)
                instance_name = detect_video_stream_utils.determine_instance_name(args.instance_name)
                # TODO - if someone reruns the same static source (video file), using the same model
                #  (which could be provided via instance name), we expect the same id for each frame
                #  for live streams (cameras, network sources), detect_video_stream_utils.determine_source()
                #  could be changed to append the start timestamp to the source
                request_id = detect_video_stream_utils.create_detection_request_id\
                    (instance_name, source, total_frame_count)
                string_map = {'id': request_id}
                message = detection_handler_pb2.handle_detection_request(
                    start_timestamp=start_time,
                    detection_classes=output_dict['detection_classes'],
                    detection_scores=output_dict['detection_scores'],
                    detection_boxes=detection_boxes,
                    instance_name=instance_name,
                    frame=detection_handler_pb2.float_array(numbers=frame.ravel(), shape=frame.shape),
                    frame_count=total_frame_count,
                    source=source,
                    float_map=float_map,
                    category_index=filtered_category_index,
                    string_map=string_map)
                response = detection_handler_stub.handle_detection(message)
                logging.debug(f"just finished frame: {total_frame_count}")
                prediction_frame_count += 1
        total_frame_count += 1
        if not frame_returned:
            break
    cap.release()
    logging.info(f" predictions/total frames : {prediction_frame_count}/{total_frame_count}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="detect objects in video")
    # credit for adding required arg - https://stackoverflow.com/a/24181138/315385
    parser.add_argument("source",
                        help="- for standard input, path to file or a numeral that represents the webcam device number")
    parser.add_argument("path_to_label_map", help="path to label map")
    parser.add_argument("tensorflow_serving_port", help="the grpc port to request prediction results from")
    parser.add_argument("model_name", help="the model name")
    parser.add_argument("--cutoff", help="cut off detection score (%%), a value between 1 and 100")
    parser.add_argument("--dryrun", help="echo a params as json object, don't process anything", action="store_true")
    parser.add_argument("--samplerate", help="how often to retrieve video frames for object detection")
    parser.add_argument("--instance_name", help="a descriptive name for this detection instance e.g. hostname")
    parser.add_argument("--handler_port", help="the port to grpc detection results to")
    args = parser.parse_args()
    detect_video_stream(args)
