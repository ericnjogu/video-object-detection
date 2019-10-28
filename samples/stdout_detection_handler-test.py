import pytest
import grpc
import datetime
import numpy
import logging

from proto.generated import detection_handler_pb2

# to enable log statements on CLI run with python -m pytest samples/stdout_detection_handler-test.py --log-cli-level=DEBUG

def test_create_handle_detection_request(create_handle_detection_request):
    """
    Tests that the input data is serialized okay into the protobuf types without errors
    Receives a fixture function defined in conftest.py which is loaded by pytest
    """
    msg, string_map, float_map = create_handle_detection_request
    assert len(msg.frame.numbers) == 9
    assert msg.frame.shape == [1, 3, 3]

    assert len(msg.detection_boxes.numbers) == 8
    assert msg.detection_boxes.shape == [2, 4]
    ndarray = numpy.array(msg.detection_boxes.numbers).reshape(msg.detection_boxes.shape)
    logging.debug(ndarray)

    assert msg.string_map == string_map
    for k, v in msg.float_map.items():
        assert v == pytest.approx(float_map[k])



def test_create_handle_detection_response():
    response = detection_handler_pb2.handle_detection_response(status=True)
