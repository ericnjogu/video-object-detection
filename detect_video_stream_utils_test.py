import subprocess
import logging
import sys
import json
import detect_video_stream_tf_libs as detect_video_stream, detect_video_stream_utils
import unittest.mock as mock
import tempfile
import pytest
import platform
from datetime import datetime
from proto.generated.tensorflow_serving.apis import predict_pb2, model_pb2
import numpy
import tensorflow as tf
#from proto.generated.tensorflow.core.framework import tensor_pb2


@pytest.fixture(scope='function')
def setup_logging():
    logging.getLogger().setLevel(logging.DEBUG)


def test_required_args():
    """ running file without path to label map parameter should show error and return non-zero status"""
    result = subprocess.run(
        ["python", "./detect_video_stream_tf_libs.py", "-", "/tmp/graph.pb", "/path/to/label-map.txt", "--dryrun"])
    assert result.returncode == 0, "there should be no errors, all positional args are present"


def test_optional_args():
    """ test that optional args are correctly received """
    result = subprocess.run(
        ["python", "./detect_video_stream_tf_libs.py", "-", "/tmp/graph.ext", "/path/to/label-map.txt", "--cutoff", "88",
         "--dryrun", "--classes", "1 5 8 34", "--samplerate", "10", "--instance_name", "acer-ubuntu-18"],
        stdout=subprocess.PIPE)
    assert result.stderr is None
    assert result.returncode == 0, "there should be no error return code"
    args = json.loads(result.stdout)
    assert args['source'] == '-', "source differs"
    assert args['path_to_frozen_graph'] == "/tmp/graph.ext", "path to frozen graph differs"
    assert args['cutoff'] == "88", "cutoff score differs"
    assert args['classes'] == "1 5 8 34", "classes differ"
    assert args['samplerate'] == "10", "samplerates differs"
    assert args['instance_name'] == "acer-ubuntu-18", "name differs"


def test_determine_samplerate_no_input():
    sample_rate = detect_video_stream_utils.determine_samplerate(None, detect_video_stream.SAMPLE_RATE)
    assert sample_rate == detect_video_stream.SAMPLE_RATE, "when sample rate is not specified, use default"


def test_determine_samplerate_with_input():
    samplerate = 15
    sample_rate_result = detect_video_stream_utils.determine_samplerate(samplerate, detect_video_stream.SAMPLE_RATE)
    assert sample_rate_result == samplerate, "when sample rate is specified, use it"


def test_determine_source_hyphen():
    args = mock.Mock()
    args.source = '-'

    video_reader = mock.Mock().callable()
    src = detect_video_stream_utils.determine_source(args, video_reader)

    video_reader.assert_called_once()
    video_reader.assert_called_with(sys.stdin)


def test_determine_source_webcam_device_number():
    args = mock.Mock()
    args.source = '2'
    video_reader = mock.Mock().callable()
    src = detect_video_stream_utils.determine_source(args, video_reader)
    video_reader.assert_called_once()
    video_reader.assert_called_with('2')


def test_determine_source_url():
    args = mock.Mock()
    url = tempfile.NamedTemporaryFile(delete=False).name
    args.source = url
    video_reader = mock.Mock().callable()
    src = detect_video_stream_utils.determine_source(args, video_reader)
    video_reader.assert_called_once()
    video_reader.assert_called_with(url)


def test_determine_cut_off_score_absent_in_args():
    # simulate args not having the optional arg
    args = {}
    cut_off_score = detect_video_stream_utils.determine_cut_off_score(args,
                                                                      default_cut_off=detect_video_stream.CUT_OFF_SCORE)
    assert cut_off_score == detect_video_stream.CUT_OFF_SCORE, "cut off score differs from default value"


def test_determine_cut_off_score_none():
    # Attribute error is not being thrown when the optional arg is not present, instead None is returned
    args = mock.Mock()
    args.cutoff = None
    cut_off_score = detect_video_stream_utils.determine_cut_off_score(args,
                                                                      default_cut_off=detect_video_stream.CUT_OFF_SCORE)
    assert cut_off_score == detect_video_stream.CUT_OFF_SCORE, "cut off score differs from default value"


def test_determine_cut_off_score_present_in_args():
    args = mock.Mock()
    args.cutoff = '51'
    cut_off_score = detect_video_stream_utils.determine_cut_off_score(args,
                                                                      default_cut_off=detect_video_stream.CUT_OFF_SCORE)
    assert cut_off_score == .51, "cut off score differs from provided value"


def test_filter_detection_output_from_file(setup_logging):
    # needed to properly create the numpy arrays in the file being read below during eval()
    import numpy as np
    # code to read dict from file adapted from https://stackoverflow.com/a/11027069/315385
    with open('samples/output_dict_01.txt', 'r') as f:
        text = f.read()
        output_dict = eval(text)
        result = detect_video_stream_utils.filter_detection_output(output_dict, .14)
        assert result is not None
        # logging.debug(f"test_filter_detection_output: {result}")
        assert len(result['detection_scores']) == 1
        assert pytest.approx(result['detection_scores'][0]) == 0.14765409
        assert len(result['detection_classes']) == 1
        assert result['detection_classes'][0] == 16


def test_filter_detection_output_from_dict(setup_logging):
    output_dict = {'detection_scores': [.23, .66, .85], 'detection_classes': [11, 2, 31],
                   'detection_boxes': [[0.5740724, 0.28274727, 0.6627937, 0.40734732],
                                       [0.5740724, 0.28274727, 0.6627937, 0.40734732],
                                       [0.56495595, 0.25473273, 0.6740638, 0.43713987]]}
    result = detect_video_stream_utils.filter_detection_output(output_dict, .8)
    assert result is not None
    # logging.debug(f"test_filter_detection_output: {result}")
    assert len(result['detection_scores']) == 1
    assert result['detection_scores'][0] == 0.85
    assert len(result['detection_classes']) == 1
    assert result['detection_classes'][0] == 31


def test_determine_source_name_hyphen():
    assert "standard input" == detect_video_stream_utils.determine_source_name('-')


def test_determine_source_name_webcam_device_number():
    assert "device 2" == detect_video_stream_utils.determine_source_name('2')


def test_determine_source_name_url():
    url = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    assert url == detect_video_stream_utils.determine_source_name(url)


def test_determine_instance_name_not_provided():
    assert platform.uname().node == detect_video_stream_utils.determine_instance_name(None)


def test_determine_instance_name_provided():
    name = "backyard"
    assert name == detect_video_stream_utils.determine_instance_name(name)


def test_determine_handler_port_not_provided():
    handler_port = detect_video_stream_utils.determine_handler_port(None, detect_video_stream.HANDLER_PORT)
    assert handler_port == detect_video_stream.HANDLER_PORT, "when handler port is not specified, use default"


def test_determine_handler_port_with_input():
    handler_port = 50000
    handler_port_result = detect_video_stream_utils.determine_handler_port(handler_port,
                                                                           detect_video_stream.HANDLER_PORT)
    assert handler_port_result == 50000, "when handler port is specified, use it"


def test_class_names_from_index_01():
    classes = [1, 1]
    category_index = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'pedestrian'}}
    result = detect_video_stream_utils.class_names_from_index(classes, category_index)
    assert result == {1: 'car'}


def test_class_names_from_index_02():
    classes = [1, 1, 2, 2, 1]
    category_index = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'pedestrian'}}
    result = detect_video_stream_utils.class_names_from_index(classes, category_index)
    assert result == {1: 'car', 2: 'pedestrian'}


def test_create_detection_request_id(setup_logging):
    req_id = detect_video_stream_utils.create_detection_request_id("localhost", "webcam", 958, datetime.now().timestamp())
    assert isinstance(req_id, str)
    assert len(req_id) > 0
    logging.debug(f"id is {req_id}")


def test_create_prediction_request():
    array = numpy.array([[1,2,3],[4,5,6]])
    request = predict_pb2.PredictRequest(
                model_spec=model_pb2.ModelSpec(name='good model'),
                inputs={'image_tensor:0': tf.make_tensor_proto(array)})
