#!/bin/bash
export PYTHONPATH=.:proto/generated:proto/generated/tensorflow_serving/apis
source activate object_detection_mini
$*
