#!/bin/sh
echo "Executing object detection experiment.";
python ./execute_object_detection.py --model ssd --video ../../datasets/jodoin/rouen_video.avi --transformations None --output ../output/rouen/annotations.json --print_output_folder ../output/images
python ./evaluate.py --ground_truth ../../datasets/jodoin/rouen/annotations/rouen.json --results ../output/rouen/annotations.json