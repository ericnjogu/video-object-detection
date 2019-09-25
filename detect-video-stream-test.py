import unittest
import subprocess
import logging
import sys
import json
import detect_video_stream
import unittest.mock as mock
import tempfile

class ClassName(unittest.TestCase):
    """testing detect_video_stream.py"""

    def test_required_args(self):
        """ running file without path to label map parameter should show error and return non-zero status"""
        result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.pb", "/path/to/label-map.txt", "--dryrun"])
        self.assertEqual(result.returncode, 0, "there should be no errors, all positional args are present")

    def test_optional_args(self):
        """ test that optional args are correctly received """
        result = subprocess.run(["python", "./detect_video_stream.py", "-", "/tmp/graph.ext", "/path/to/label-map.txt", "--cutoff", "88",
                        "--dryrun", "--classes", "1 5 8 34", "--samplerate", "10"], capture_output=True)
        assert len(result.stderr) == 0
        self.assertEqual(result.returncode, 0, "there should be no error return code")
        args = json.loads(result.stdout)
        self.assertEqual(args['source'], '-', "source differs")
        self.assertEqual(args['path_to_frozen_graph'], "/tmp/graph.ext", "path to frozen graph differs")
        self.assertEqual(args['cutoff'], "88", "cutoff score differs")
        self.assertEqual(args['classes'], "1 5 8 34", "classes differ")
        self.assertEqual(args['samplerate'], "10", "samplerates differs")

    def test_determine_samplerate_no_input(self):
        sample_rate = detect_video_stream.determine_samplerate({})
        assert sample_rate == detect_video_stream.SAMPLE_RATE, "when sample rate is not specified, use default"

    def test_determine_samplerate_with_input(self):
        args = mock.Mock()
        args.samplerate = 15
        sample_rate = detect_video_stream.determine_samplerate(args)
        assert sample_rate == 15, "when sample rate is specified, use it"

    def test_determine_source_hyphen(self):
        args = mock.Mock()
        args.source = '-'

        video_reader = mock.Mock().callable()
        src = detect_video_stream.determine_source(args, video_reader)

        video_reader.assert_called_once()
        video_reader.assert_called_with(sys.stdin)

    def test_determine_source_webcam_device_number(self):
        args = mock.Mock()
        args.source = '2'
        video_reader = mock.Mock().callable()
        src = detect_video_stream.determine_source(args, video_reader)
        video_reader.assert_called_once()
        video_reader.assert_called_with('2')

    def test_determine_source_url(self):
        args = mock.Mock()
        url = tempfile.NamedTemporaryFile(delete=False).name
        args.source = url
        video_reader = mock.Mock().callable()
        src = detect_video_stream.determine_source(args, video_reader)
        video_reader.assert_called_once()
        video_reader.assert_called_with(url)

if __name__ == "__main__":
    unittest.main()
