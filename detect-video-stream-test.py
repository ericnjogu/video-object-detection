import unittest
import subprocess
import logging
import sys

class ClassName(unittest.TestCase):
    """testing detect-video-stream.py"""

    def test_source_arg(self):
        """ running file without source parameter should show error and return non-zero status"""
        try:
            result = subprocess.run(["python", "./detect-video-stream.py"])
            self.assertNotEqual(result.returncode, 0, "there should have been an error due to missing source arg")
        except:
            logging.error(sys.exc_info()[0])

if __name__ == "__main__":
    unittest.main()
