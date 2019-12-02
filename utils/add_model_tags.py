import video_object_detection as obj_detect
import tensorflow as tf
import sys
from datetime import datetime
import os

graph_path = sys.argv[1]
graph = obj_detect.load_frozen_model_into_memory(graph_path)
path = os.path.join(os.path.dirname(graph_path), f"model-{datetime.now().isoformat().replace(':', '-')}")
builder = tf.saved_model.builder.SavedModelBuilder(path)
with tf.Session(graph=graph) as sess:
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING])
builder.save()
print(f'saved model to {path}')
