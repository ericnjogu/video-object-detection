import grpc
import sys
import time
from concurrent import futures
import logging

from proto.generated import detection_handler_pb2_grpc, detection_handler_pb2

class StdoutDetectionHandler(detection_handler_pb2_grpc.DetectionHandlerServicer):
    def handle_detection(self, request, context):
      """
      handle a detection output
      """
      sys.stdout.writelines(str(request))
      return detection_handler_pb2.handle_detection_response(status=True)

# credit - https://www.semantics3.com/blog/a-simplified-guide-to-grpc-in-python-6c4e25f0c506/
# create server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
# add implementing class to server
detection_handler_pb2_grpc.add_DetectionHandlerServicer_to_server(StdoutDetectionHandler(), server);
# listen
port  = 50051
logging.getLogger().setLevel(logging.DEBUG)
logging.info(f'starting server on port {port}')
server.add_insecure_port(f'[::]:{port}')
server.start()
# sleep loop
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
