import pytest
import grpc
import datetime
import numpy
import logging

from proto.generated import detection_handler_pb2

# to enable log statements on CLI run with python -m pytest samples/stdout_detection_handler-test.py --log-cli-level=DEBUG

def test_create_handle_detection_request():
    # TODO this same data could be made available using a fixture to test the server implementation
    frame_array1 = [5, 2, 3]
    frame_array1_nd = numpy.array(frame_array1)
    # not sure why the frame from opencv is nested this way
    frame = [[frame_array1_nd, numpy.array([8.4, 7.9, 5.2])]]
    detection_output = {'detection_scores': [0.54], 'detection_classes':[8],
                        'detection_boxes':[numpy.array([0.36190858, 0.11737314, 0.94603133, 0.3205647]),
                                            numpy.array([0.345639  , 0.69829893, 0.38075703, 0.7310691 ])]}
    detection_boxes = [detection_handler_pb2.float_array(numbers=array) for array in detection_output['detection_boxes']]
    logging.debug(f"detection_boxes: {detection_boxes}")
    msg = detection_handler_pb2.handle_detection_request(
                start_timestamp = datetime.datetime.now().timestamp(),
                detection_scores = detection_output['detection_scores'],
                detection_classes = detection_output['detection_classes'],
                detection_boxes = detection_boxes,
                instance_name = "testing",
                frame = [detection_handler_pb2.float_array(numbers=array) for array in frame[0]],
                frame_count = 1619,
                source = "steam")
    assert msg.frame[0].numbers == frame_array1
    assert len(msg.detection_boxes) == 2


def test_create_handle_detection_response():
    response = detection_handler_pb2.handle_detection_response(status=True)
