#!/bin/sh
echo "Executing object detection experiment.";
#python ./execute_object_detection.py -m faster -b tf -mo ../output/food_44_classes/models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/exported/ -f "../input/food/44_classes/dataset/val/*.jpg" -t None -o ../output/food_44_classes/ann/faster/annotations.json -p ../output/food_44_classes/images/faster
python ./evaluate.py --ground_truth ../input/food/44_classes/val_ann.json --results ../output/food_44_classes/ann/faster/annotations.json
