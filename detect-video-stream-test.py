import subprocess
import logging
import sys
import json
import detect_video_stream
import unittest.mock as mock
import tempfile
import ast
import numpy as np
import pytest

@pytest.fixture(scope='function')
def setup_logging():
    logging.getLogger().setLevel(logging.DEBUG)

def test_required_args():
    """ running file without path to label map parameter should show error and return non-zero status"""
    result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.pb", "/path/to/label-map.txt", "--dryrun"])
    assert result.returncode == 0, "there should be no errors, all positional args are present"

def test_optional_args():
    """ test that optional args are correctly received """
    result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.ext", "/path/to/label-map.txt", "--cutoff", "88",
                    "--dryrun", "--classes", "1 5 8 34", "--samplerate", "10", "--name", "acer-ubuntu-18"], stdout=subprocess.PIPE)
    assert result.stderr is None
    assert result.returncode == 0, "there should be no error return code"
    args = json.loads(result.stdout)
    assert args['source'] == '-', "source differs"
    assert args['path_to_frozen_graph'] == "/tmp/graph.ext", "path to frozen graph differs"
    assert args['cutoff'] == "88", "cutoff score differs"
    assert args['classes'] == "1 5 8 34", "classes differ"
    assert args['samplerate'] == "10", "samplerates differs"
    assert args['name'] == "acer-ubuntu-18", "name differs"

def test_determine_samplerate_no_input():
    sample_rate = detect_video_stream.determine_samplerate({})
    assert sample_rate == detect_video_stream.SAMPLE_RATE, "when sample rate is not specified, use default"

def test_determine_samplerate_with_input():
    args = mock.Mock()
    args.samplerate = 15
    sample_rate = detect_video_stream.determine_samplerate(args)
    assert sample_rate == 15, "when sample rate is specified, use it"

def test_determine_source_hyphen():
    args = mock.Mock()
    args.source = '-'

    video_reader = mock.Mock().callable()
    src = detect_video_stream.determine_source(args, video_reader)

    video_reader.assert_called_once()
    video_reader.assert_called_with(sys.stdin)

def test_determine_source_webcam_device_number():
    args = mock.Mock()
    args.source = '2'
    video_reader = mock.Mock().callable()
    src = detect_video_stream.determine_source(args, video_reader)
    video_reader.assert_called_once()
    video_reader.assert_called_with('2')

def test_determine_source_url():
    args = mock.Mock()
    url = tempfile.NamedTemporaryFile(delete=False).name
    args.source = url
    video_reader = mock.Mock().callable()
    src = detect_video_stream.determine_source(args, video_reader)
    video_reader.assert_called_once()
    video_reader.assert_called_with(url)

def test_determine_cut_off_score_absent_in_args():
    # simulate args not having the optional arg
    args = {}
    cut_off_score = detect_video_stream.determine_cut_off_score(args)
    assert cut_off_score == detect_video_stream.CUT_OFF_SCORE, "cut off score differs from default value"

def test_determine_cut_off_score_present_in_args():
    args = mock.Mock()
    args.cutoff = 51
    cut_off_score = detect_video_stream.determine_cut_off_score(args)
    assert cut_off_score == 51, "cut off score differs from provided value"

def test_filter_detection_output_from_file(setup_logging):
    # code to read dict from file adapted from https://stackoverflow.com/a/11027069/315385
    with open('samples/output_dict_01.txt', 'r') as f:
        text = f.read()
        output_dict = eval(text)
        result = detect_video_stream.filter_detection_output(output_dict, .14)
        assert result is not None
        # logging.debug(f"test_filter_detection_output: {result}")
        assert len(result['detection_scores']) == 1
        assert pytest.approx(result['detection_scores'][0]) == 0.14765409
        assert len(result['detection_classes']) == 1
        assert result['detection_classes'][0] == 16

def test_filter_detection_output_from_dict(setup_logging):
    output_dict = {'detection_scores':[.23, .66, .85], 'detection_classes':[11, 2, 31], 'detection_boxes': [[0.5740724 , 0.28274727, 0.6627937 , 0.40734732],
           [0.5740724 , 0.28274727, 0.6627937 , 0.40734732],
           [0.56495595, 0.25473273, 0.6740638 , 0.43713987]]}
    result = detect_video_stream.filter_detection_output(output_dict, .8)
    assert result is not None
    #logging.debug(f"test_filter_detection_output: {result}")
    assert len(result['detection_scores']) == 1
    assert result['detection_scores'][0] == 0.85
    assert len(result['detection_classes']) == 1
    assert result['detection_classes'][0] == 31

def test_determine_source_name_hyphen():
    assert "standard input" == detect_video_stream.determine_source_name('-')

def test_determine_source_name_webcam_device_number():
    assert "device 2" == detect_video_stream.determine_source_name('2')

def test_determine_source_name_url():
    url = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    assert url == detect_video_stream.determine_source_name(url)
