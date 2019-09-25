# imports
import logging
import os
import argparse
import sys
import json
import cv2
import object_detection
# assemble inputs
# run inference
# echo output as json file - explore whether we will output numpy array or create a temp file with frame
CUT_OFF_SCORE = 90
SAMPLE_RATE = 5

def determine_samplerate(input_from_args):
    # TODO
    pass

def determine_source(input_from_args):
    # TODO
    pass

def detect_video_stream(args):
    """ detect objects in video stream """
    # __dict__ trick from https://stackoverflow.com/a/3768975/315385
    if args.dryrun:
        sys.stdout.writelines(json.dumps(args.__dict__))
        return
    # TODO validate args
    detection_graph = object_detection.load_frozen_model_into_memory(args.path_to_frozen_graph)
    # determine sample rate
    sample_rate = determine_samplerate(args.samplerate)
    # loop over frames in video
    # adapted from https://github.com/juandes/pikachu-detection/blob/master/detection_video.py
    cap = cv2.VideoCapture(determine_source(args.source))
    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        # only consider frames that are a multiple of the sample rate
        if frame_count % sample_rate == 0:
            image = object_detection.load_image_into_numpy_array(frame)
            # run inference
            output_dict = object_detection.run_inference_for_single_image(image, detection_graph)
            # TODO filter for classes, cut off score
            # write JSON to stdout
            sys.stdout.write(json.dumps(output_dict.__dict__()))
        frame_count += 1


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="detect objects in video")
    # credit for adding required arg - https://stackoverflow.com/a/24181138/315385
    parser.add_argument("source", help="- for standard input, path to file or a numeral that represents the webcam device number")
    parser.add_argument("path_to_frozen_graph", help="path to frozen model graph")
    parser.add_argument("path_to_label_map", help="path to label map")
    parser.add_argument("--cutoff", help="cut off detection score (%)")
    parser.add_argument("--dryrun", help="echo a params as json object, don't process anything", action="store_true")
    parser.add_argument("--classes",
            help="space separated list of object classes to detect as specified in label mapping")
    parser.add_argument("--samplerate", help="how often to retrieve video frames for object detection")
    args = parser.parse_args()
    detect_video_stream(args)
