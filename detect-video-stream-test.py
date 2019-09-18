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
                        "--dryrun", "--classes", "1 5 8 34"], capture_output=True)
        self.assertEqual(result.returncode, 0, "there should be no error return code")
        args = json.loads(result.stdout)
        self.assertEquals(args['source'], '-', "source differs")
        self.assertEquals(args['path_to_frozen_graph'], "/tmp/graph.ext", "path to frozen graph differs")
        self.assertEquals(args['cutoff'], "88", "cutoff score differs")
        self.assertEquals(args['classes'], "1 5 8 34", "classes differ")

if __name__ == "__main__":
    unittest.main()
