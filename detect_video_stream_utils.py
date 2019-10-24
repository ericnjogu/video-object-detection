import os
import platform
import sys

def determine_handler_port(args, default_handler_port):
    #TODO - consider reusing determine_samplerate
    pass

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

def determine_cut_off_score(args, default_cut_off):
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
            return default_cut_off
    except AttributeError:
        return default_cut_off

def determine_samplerate(args, default_sample_rate):
    try:
        """ check for sample rate in args, if absent return default """
        return args.samplerate
    except AttributeError:
        return default_sample_rate

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
