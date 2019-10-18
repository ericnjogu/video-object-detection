import pytest
import grpc
import datetime
import numpy

from proto.generated import detection_handler_pb2

def test_create_handle_detection_request():
    #TODO this same data could be made available using a fixture to test the server implementation
    msg = detection_handler_pb2.handle_detection_request(
                start_timestamp = datetime.datetime.now().timestamp(),
                detection_output = {'detection_scores': detection_handler_pb2.float_array(numbers=[5.0])},
                instance_name = "testing",
                frame = [detection_handler_pb2.float_array(numbers=[5.0, 2.1, 3.3]),
                         detection_handler_pb2.float_array(numbers=[8.4, 7.9, 5.2])],
                frame_count = 1619,
                source = "steam")


def test_create_handle_detection_response():
    response = detection_handler_pb2.handle_detection_response(status=True)
