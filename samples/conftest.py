import pytest
import numpy
from proto.generated import detection_handler_pb2
import datetime


@pytest.fixture
def create_handle_detection_request():
    """
    returns a detection_handler_pb2.handle_detection_request object if the input data is successfully serialized into protobuf types
    """
    frame_array1 = [5, 2, 3]
    # not sure why the frame from opencv is nested this way
    frame = numpy.array([[frame_array1, [8.4, 7.9, 5.2], [59,  64,  64]]])
    detection_output = {'detection_scores': [0.54], 'detection_classes':[8],
                        'detection_boxes':numpy.array([[0.36190858, 0.11737314, 0.94603133, 0.3205647]])}
    detection_boxes = detection_handler_pb2.float_array(numbers=detection_output['detection_boxes'].ravel(),
                                                        shape=detection_output['detection_boxes'].shape)
    string_map={'color':'blue', 'music':'classical'}
    float_map={'weight':56.9, 'height':85.4}
    category_index = {8: {'name':"elephant"}}
    return detection_handler_pb2.handle_detection_request(
                start_timestamp = datetime.datetime.now().timestamp(),
                detection_scores = detection_output['detection_scores'],
                detection_classes = detection_output['detection_classes'],
                detection_boxes = detection_boxes,
                instance_name = "testing",
                frame = detection_handler_pb2.float_array(numbers=frame.ravel(), shape=frame.shape),
                frame_count = 1619,
                source = "steam",
                string_map=string_map,
                float_map=float_map,
                category_index={k:v['name'] for k, v in category_index.items()}), string_map, float_map, category_index
