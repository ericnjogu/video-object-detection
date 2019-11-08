import os
import platform
import sys


def determine_input_arg(arg_val, default_arg_val):
    """ if arg_val exists, use it, else return default_arg_val """
    if arg_val:
        return arg_val
    else:
        return default_arg_val


def determine_handler_port(handler_port_arg, default_handler_port):
    """ if handler_port_arg exists, use it, else return default_handler_port """
    return determine_input_arg(handler_port_arg, default_handler_port)


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


def determine_samplerate(sample_rate_arg, default_sample_rate):
    """ if sample_rate_arg exists, use it, else return default_sample_rate """
    return determine_input_arg(sample_rate_arg, default_sample_rate)


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


def class_names_from_index(classes, category_index):
    """
    :param classes: a list of detected classes e.g. [1, 1]
    :param category_index: the category index dict e.g. {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'pedestrian'}}
    :return: a dict of {class_id:class_name} e.g. {1:'car'}
    """
    return {k: category_index[k]['name'] for k in classes}

