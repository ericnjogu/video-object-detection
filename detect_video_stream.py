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

from proto.generated import detection_handler_pb2_grpc, detection_handler_pb2
import video_object_detection as obj_detect
import detect_video_stream_utils

CUT_OFF_SCORE = 90.0
SAMPLE_RATE = 5
HANDLER_PORT = 50051

def detect_video_stream(args):
    """ detect objects in video stream """
    # setup grpc comms
    handler_port = detect_video_stream_utils.determine_handler_port(args.handler_port, HANDLER_PORT)
    url = f'localhost:{handler_port}'
    logging.debug(f'connecting to handle at {url}')
    channel = grpc.insecure_channel(url)
    stub = detection_handler_pb2_grpc.DetectionHandlerStub(channel)
    # __dict__ trick from https://stackoverflow.com/a/3768975/315385
    if args.dryrun:
        sys.stdout.writelines(json.dumps(args.__dict__))
        return
    # generate dict from labels
    category_index = label_utils.create_category_index_from_labelmap(args.path_to_label_map, use_display_name=True)
    # logging.debug(f"category_index: {category_index}")
    # TODO validate args
    detection_graph = obj_detect.load_frozen_model_into_memory(args.path_to_frozen_graph)
    # determine sample rate
    sample_rate = detect_video_stream_utils.determine_samplerate(args.samplerate, SAMPLE_RATE)
    cap = detect_video_stream_utils.determine_source(args, cv2.VideoCapture)
    float_map = {'frame_height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'frame_width':cap.get(cv2.CAP_PROP_FRAME_WIDTH)}
    frame_count = 0
    start_time = dt.now().timestamp()
    cut_off_score = detect_video_stream_utils.determine_cut_off_score(args, default_cut_off=CUT_OFF_SCORE)
    logging.debug(f"using a cut off score of {cut_off_score}")
    # loop over frames in video
    # adapted from https://github.com/juandes/pikachu-detection/blob/master/detection_video.py
    while cap.isOpened():
        ret, frame = cap.read()
        # only consider frames that are a multiple of the sample rate
        if frame is not None and frame_count % sample_rate == 0:
            frame_bgr2rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_to_array = np.expand_dims(frame_bgr2rgb, axis=0)
            # run inference
            output_dict = obj_detect.run_inference_for_single_image(img_to_array, detection_graph)
            # filter for cut off score
            output_dict = detect_video_stream_utils.filter_detection_output(output_dict, cut_off_score)
            if len(output_dict['detection_boxes']) > 0:
                # convert to numpy array so that we can flatten, retrieve shape
                output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
                detection_boxes = detection_handler_pb2.float_array(numbers=output_dict['detection_boxes'].ravel(), shape=output_dict['detection_boxes'].shape)
                filtered_category_index = detect_video_stream_utils.class_names_from_index(output_dict['detection_classes'], category_index)
                message = detection_handler_pb2.handle_detection_request(
                            start_timestamp = start_time,
                            detection_classes = output_dict['detection_classes'],
                            detection_scores = output_dict['detection_scores'],
                            detection_boxes = detection_boxes,
                            instance_name = detect_video_stream_utils.determine_instance_name(args.instance_name),
                            frame = detection_handler_pb2.float_array(numbers=frame.ravel(), shape=frame.shape),
                            frame_count = frame_count,
                            source = detect_video_stream_utils.determine_source_name(args.source),
                            float_map=float_map,
                            category_index=filtered_category_index)
                response = stub.handle_detection(message)
                logging.debug(f"detection handler response is {response.status}")
            else:
                logging.debug("no score was above cut-off, skipping")
            logging.debug(f"just finished frame: {frame_count}")
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="detect objects in video")
    # credit for adding required arg - https://stackoverflow.com/a/24181138/315385
    parser.add_argument("source", help="- for standard input, path to file or a numeral that represents the webcam device number")
    parser.add_argument("path_to_frozen_graph", help="path to frozen model graph")
    parser.add_argument("path_to_label_map", help="path to label map")
    parser.add_argument("--cutoff", help="cut off detection score (%%), a value between 1 and 100")
    parser.add_argument("--dryrun", help="echo a params as json object, don't process anything", action="store_true")
    parser.add_argument("--classes",
            help="space separated list of object classes to detect as specified in label mapping")
    parser.add_argument("--samplerate", help="how often to retrieve video frames for object detection")
    parser.add_argument("--instance_name", help="a descriptive name for this detection instance e.g. hostname")
    parser.add_argument("--handler_port", help="the port to grpc detection results to")
    args = parser.parse_args()
    detect_video_stream(args)
