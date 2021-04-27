#!/bin/sh
echo "Executing fine-tuning.";
python ../../Tensorflow/models/research/object_detection/model_main_tf2.py --pipeline_config_path=../input/pre-trained-models/pipeline.config --model_dir=../output/models/ --alsologtostderr
