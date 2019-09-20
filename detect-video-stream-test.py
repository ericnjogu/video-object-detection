import unittest
import subprocess
import logging
import sys
import json

class ClassName(unittest.TestCase):
    """testing detect-video-stream.py"""

    def test_required_args(self):
        """ running file without path to label map parameter should show error and return non-zero status"""
        result = subprocess.run(["python", "./detect-video-stream.py", "-", "/tmp/graph.pb", "/path/to/label-map.txt"])
        self.assertEqual(result.returncode, 0, "there should be no errors, all positional args are present")

    def test_optional_args(self):
        """ test that optional args are correctly received """
        result = subprocess.run(["python", "./detect-video-stream.py", "-", "/tmp/graph.ext", "/path/to/label-map.txt", "--cutoff", "88",
                        "--dryrun", "--classes", "1 5 8 34", "--samplerate", "10"], capture_output=True)
        assert len(result.stderr) == 0
        self.assertEqual(result.returncode, 0, "there should be no error return code")
        args = json.loads(result.stdout)
        self.assertEqual(args['source'], '-', "source differs")
        self.assertEqual(args['path_to_frozen_graph'], "/tmp/graph.ext", "path to frozen graph differs")
        self.assertEqual(args['cutoff'], "88", "cutoff score differs")
        self.assertEqual(args['classes'], "1 5 8 34", "classes differ")
        self.assertEqual(args['samplerate'], "10", "samplerates differs")

if __name__ == "__main__":
    unittest.main()
