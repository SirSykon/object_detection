#!/bin/sh
echo "Executing object detection experiment.";
python ./execute_object_detection.py --model faster --video ../../datasets/jodoin/rouen_video.avi --transformations None --print_output_folder ../output/images