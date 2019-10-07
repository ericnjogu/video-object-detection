# imports
import logging
import os
import argparse
import sys
import json
import cv2
import object_detection.object_detection as obj_detect
import numpy as np
import tempfile
import object_detection.utils.visualization_utils as vis_utils
import object_detection.utils.label_map_util as label_utils
from datetime import datetime as dt
import platform

CUT_OFF_SCORE = 90.0
SAMPLE_RATE = 5

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
    sample_rate = determine_samplerate(args.samplerate)
    # logging.debug(f"using sample_rate {sample_rate}")
    # loop over frames in video
    # adapted from https://github.com/juandes/pikachu-detection/blob/master/detection_video.py
    cap = determine_source(args, cv2.VideoCapture)
    # TODO accept a switch that will change from video output to text output

    # START - move to visualization service

    # output_video_file_path = tempfile.NamedTemporaryFile(suffix='.avi', delete=False).name
    # logging.info(f'writing output video to {output_video_file_path}')

    # video_out = cv2.VideoWriter(output_video_file_path,
    #                             fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    #                             apiPreference=cv2.CAP_ANY,
    #                             fps=10,
    #                             frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    # END - move to visualization service

    frame_count = 0
    start_time = dt.now().timestamp()
    cut_off_score = determine_cut_off_score(args)
    logging.debug(f"using a cut off score of {cut_off_score}")
    while cap.isOpened():
        ret, frame = cap.read()
        ##logging.debug(f"read status is {ret}")
        # only consider frames that are a multiple of the sample rate
        if frame_count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_to_array = np.expand_dims(frame, axis=0)
            # image = object_detection.load_image_into_numpy_array(img_to_array)
            # run inference
            output_dict = obj_detect.run_inference_for_single_image(img_to_array, detection_graph)
            # TODO filter for classes, cut off score
            output_dict = filter_detection_output(output_dict, cut_off_score)
            # TODO implement with switch from args - write output dict to stdout
            #
            # OR write image to video out
            # credit - https://github.com/juandes/pikachu-detection/blob/master/detection_video.py

            # START - move to visualization service - and change to image output

            # vis_utils.visualize_boxes_and_labels_on_image_array(
            #     frame,
            #     output_dict['detection_boxes'],
            #     output_dict['detection_classes'],
            #     output_dict['detection_scores'],
            #     category_index,
            #     instance_masks=output_dict.get('detection_masks'),
            #     use_normalized_coordinates=True,
            #     line_thickness=10)
            # output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # video_out.write(output_rgb)

            # END move to visualization service
            if len(output_dict['detection_boxes']) > 0:
                sys.stdout.writelines(str({'start_time': start_time,'output_dict':output_dict,
                    'name': determine_instance_name(args.instance_name), 'frame': frame,
                    'frame_count':frame_count, 'source':determine_source_name(args.source)}))
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
