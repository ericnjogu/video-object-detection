import unittest
import subprocess
import logging
import sys
import json

class ClassName(unittest.TestCase):
    """testing detect-video-stream.py"""

    def test_source_arg(self):
        """ running file without source parameter should show error and return non-zero status"""
        try:
            result = subprocess.run(["python", "./detect-video-stream.py"])
            self.assertNotEqual(result.returncode, 0, "there should have been an error due to missing source arg")
        except:
            logging.error(sys.exc_info()[0])

    def test_frozen_graph_arg(self):
        """ running file without frozen graph arg parameter should show error and return non-zero status"""
        result = subprocess.run(["python", "./detect-video-stream.py", "-"])
        self.assertNotEqual(result.returncode, 0, "there should have been an error due to missing frozen arg")

    def test_optional_args(self):
        """ test that optional args are correctly received """
        result = subprocess.run(["python", "./detect-video-stream.py", "-", "/tmp/graph.ext", "--cutoff", "88",
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
