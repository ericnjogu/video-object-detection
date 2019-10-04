import subprocess
import logging
import sys
import json
import detect_video_stream
import unittest.mock as mock
import tempfile

def test_required_args():
    """ running file without path to label map parameter should show error and return non-zero status"""
    result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.pb", "/path/to/label-map.txt", "--dryrun"])
    assert result.returncode == 0, "there should be no errors, all positional args are present"

def test_optional_args():
    """ test that optional args are correctly received """
    result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.ext", "/path/to/label-map.txt", "--cutoff", "88",
                    "--dryrun", "--classes", "1 5 8 34", "--samplerate", "10"], capture_output=True)
    assert len(result.stderr) == 0
    assert result.returncode == 0, "there should be no error return code"
    args = json.loads(result.stdout)
    assert args['source'] == '-', "source differs"
    assert args['path_to_frozen_graph'] == "/tmp/graph.ext", "path to frozen graph differs"
    assert args['cutoff'] == "88", "cutoff score differs"
    assert args['classes'] == "1 5 8 34", "classes differ"
    assert args['samplerate'] == "10", "samplerates differs"

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
