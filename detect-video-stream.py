# imports
import logging
import os
import argparse
import sys
import json
# assemble inputs
# run inference
# echo output as json file - explore whether we will output numpy array or create a temp file with frame
CUT_OFF_SCORE = 90
SAMPLE_RATE = 5

###
def detect_video_stream(args):
    """ detect objects in video stream """
    # __dict__ trick from https://stackoverflow.com/a/3768975/315385
    if args.dryrun:
        sys.stdout.writelines(json.dumps(args.__dict__))

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="detect objects in video")
    # credit for adding required arg - https://stackoverflow.com/a/24181138/315385
    parser.add_argument("source", help="- for standard input or a numeral that represents the webcam device number")
    parser.add_argument("path_to_frozen_graph", help="path to frozen model graph")
    parser.add_argument("--cutoff", help="cut off detection score (%)")
    parser.add_argument("--dryrun", help="echo a params as json object, don't process anything", action="store_true")
    parser.add_argument("--classes",
            help="space separated list of object classes to detect as specified in label mapping")
    parser.add_argument("--samplerate", help="how often to retrieve video frames for object detection")
    args = parser.parse_args()
    detect_video_stream(args)
