# imports
import logging
import os
import argparse
import sys
import json
import cv2
import numpy as np
import tempfile
from datetime import datetime as dt
import platform
import grpc

from proto.generated import detection_handler_pb2_grpc, detection_handler_pb2
import object_detection as obj_detect

CUT_OFF_SCORE = 90.0
SAMPLE_RATE = 5
HANDLER_PORT = 50051

# TODO - move the determine* methods to a util module
# introduce a determine_port method

def determine_instance_name(instance_name):
    """
    parameters:
        instance_name: the value provided via an optional arg --name. It is used to describe this detection instance

        returns a descriptive string e.g. 'erics-laptop'
    """
    if instance_name:
        return instance_name
    else:
        return platform.uname().node

def determine_source_name(src):
    """
    parameters:
        src: The source attribute determine which source the video is to be read, can be any of the following values
                '-' (hyphen): standard input
                digit: webcam
                URL: file path

        returns a descriptive string e.g. 'device 0'
    """
    if src == '-':
        return "standard input"
    elif src.isnumeric():
        return f"device {src}"
    elif os.path.exists(src):
        return src

def filter_detection_output(detection_output_dict, cut_off_score):
    """
    drop all detections from the dict whose score is less than the cut_off_score

    args:
    detection_output_dict - A dict returned frrom running obj_detect.run_inference_for_single_image()
    cut_off_score - the minimum score to retain detections which is the percentage divided by 100. e.g. 30% will be passed in as 0.3

    return - the filtered dict
    """
    result = {}
    # create a lambda function that returns an iterable of True/False values using map() on scores
    score_retain_status_iter = lambda : map(lambda score: score >= cut_off_score, detection_output_dict['detection_scores'])
    # logging.debug(f"true/false values of matching scores : {list(score_retain_status_iter())}")
    # use the true/false values to filter out detection classes in the corresponding positions
    iterator  = score_retain_status_iter()
    result['detection_classes'] = list(filter(lambda x: next(iterator), detection_output_dict['detection_classes']))
    # use the true/false values to filter out detection boxes in the corresponding positions
    iterator  = score_retain_status_iter()
    result['detection_boxes'] = list(filter(lambda x: next(iterator), detection_output_dict['detection_boxes']))
    # use the true/false values to filter out detection scores in the corresponding positions
    iterator  = score_retain_status_iter()
    result['detection_scores'] = list(filter(lambda x: next(iterator), detection_output_dict['detection_scores']))

    return result

def determine_cut_off_score(args):
    """
    check for cut_off_score in args, if absent return default

    Args:
    args: a namespace object from argparse.ArgumentParser.parse_args()

    returns - the cut_off_score in args, if absent return default
    """
    try:
        if args.cutoff:
            return float(args.cutoff) / 100
        else:
            return CUT_OFF_SCORE
    except AttributeError:
        return CUT_OFF_SCORE

def determine_samplerate(args):
    try:
        """ check for sample rate in args, if absent return default """
        return args.samplerate
    except AttributeError:
        return SAMPLE_RATE

def determine_source(args, video_reader):
    """
    parameters:
        args: a namespace object from argparse.ArgumentParser.parse_args().
            The source attribute determine which source the video is to be read load_frozen_model_into_memory and can be any of the following values
                '-' (hyphen): standard input
                digit: webcam
                URL: file path
        video_reader: the class/function to use to read video from file or camera, useful for mocking

        returns a file object
    """
    if args.source == '-':
        return video_reader(sys.stdin)
    elif args.source.isnumeric():
        return video_reader(args.source)
    elif os.path.exists(args.source):
        return video_reader(args.source)

def detect_video_stream(args):
    """ detect objects in video stream """
    # setup grpc comms
    channel = grpc.insecure_channel(f'localhost:{HANDLER_PORT}')
    stub = detection_handler_pb2_grpc.DetectionHandlerStub(channel)
    # __dict__ trick from https://stackoverflow.com/a/3768975/315385
    if args.dryrun:
        sys.stdout.writelines(json.dumps(args.__dict__))
        return
    # TODO validate args
    detection_graph = obj_detect.load_frozen_model_into_memory(args.path_to_frozen_graph)
    # determine sample rate
    sample_rate = determine_samplerate(args.samplerate)
    cap = determine_source(args, cv2.VideoCapture)
    frame_count = 0
    start_time = dt.now().timestamp()
    cut_off_score = determine_cut_off_score(args)
    logging.debug(f"using a cut off score of {cut_off_score}")
    # loop over frames in video
    # adapted from https://github.com/juandes/pikachu-detection/blob/master/detection_video.py
    while cap.isOpened():
        ret, frame = cap.read()
        # only consider frames that are a multiple of the sample rate
        if frame_count % sample_rate == 0:
            frame_bgr2rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_to_array = np.expand_dims(frame_bgr2rgb, axis=0)
            # run inference
            output_dict = obj_detect.run_inference_for_single_image(img_to_array, detection_graph)
            # filter for cut off score
            output_dict = filter_detection_output(output_dict, cut_off_score)
            # TODO
            # convert the output dict entries into float_array - could handle the one dim separately in comprehension
            # investigate how we can call float_array on each element in a numpy array - at a certain level without deep (for arrays only)
            if len(output_dict['detection_boxes']) > 0:
                output_dict['detection_boxes'] = [detection_handler_pb2.float_array(numbers=array) for array in output_dict['detection_boxes']]
                message = detection_handler_pb2.handle_detection_request(
                            start_timestamp = start_time,
                            detection_classes = output_dict['detection_classes'],
                            detection_scores = output_dict['detection_scores'],
                            detection_boxes = output_dict['detection_boxes'],
                            instance_name = determine_instance_name(args.instance_name),
                            frame = [detection_handler_pb2.float_array(numbers=array) for array in frame[0]],
                            frame_count = frame_count,
                            source = determine_source_name(args.source))
                response = stub.handle_detection(message)
                logging.debug(f"detection handler response is {response.status}")
            else:
                logging.debug("no score was above cut-off, skipping")
            logging.debug(f"just finished frame: {frame_count}")
        frame_count += 1

    video_out.release()
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
    args = parser.parse_args()
    detect_video_stream(args)
