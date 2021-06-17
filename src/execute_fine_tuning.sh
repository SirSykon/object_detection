#!/bin/sh
echo "Executing fine-tuning.";
python ../../Tensorflow/models/research/object_detection/model_main_tf2.py --pipeline_config_path=../input/pre-trained-models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/pipeline.config --model_dir=../output/models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/ --alsologtostderr
python ../../Tensorflow/models/research/object_detection/exporter_main_v2.py --pipeline_config_path ../input/pre-trained-models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/pipeline.config --trained_checkpoint_dir ../output/models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/ --output_directory ../output/models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/exported/
