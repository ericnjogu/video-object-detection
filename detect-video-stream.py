# imports
import logging
import os
import argparse
# assemble inputs
# run inference
# echo output as json file - explore whether we will output numpy array or create a temp file with frame

###
def detect_video_stream(source='-'):
    """ detect objects in video stream """
    # credit for retrieving current file path - http://www.karoltomala.com/blog/?p=622
    logging.debug(f"in {os.path.abspath(__file__)}: detect_video_stream()")
    pass

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="detect objects in video")
    # credit for adding required arg - https://stackoverflow.com/a/24181138/315385
    parser.add_argument("source", help="- for standard input or a numeral that represents the webcam device number")
    args = parser.parse_args()
    detect_video_stream(args.source)
