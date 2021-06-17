#!/bin/sh
echo "Executing object detection experiment.";
#python ./execute_object_detection.py --model faster --video ../../datasets/jodoin/rouen_video.avi --transformations rot270 rot90 --output ../output/rouen/ann/faster/annotations.json --print_output_folder ../output/rouen/images/faster
#python ./evaluate.py --ground_truth ../../datasets/jodoin/rouen/annotations/rouen_annotation.json --results ../output/rouen/ann/faster/annotations.json

python ./execute_object_detection.py --model ssd --video ../../datasets/jodoin/rouen_video.avi --transformations rot270 rot90 --output ../output/rouen/ann/ssd/annotations.json --print_output_folder ../output/rouen/images/ssd
#python ./evaluate.py --ground_truth ../../datasets/jodoin/rouen/annotations/rouen_annotation.json --results ../output/rouen/ann/ssd/annotations.json

#python ./execute_object_detection.py --model yolo --video ../../datasets/jodoin/rouen_video.avi --transformations rot270 rot90 --output ../output/rouen/ann/yolo/annotations.json --print_output_folder ../output/rouen/images/yolo
#python ./evaluate.py --ground_truth ../../datasets/jodoin/rouen/annotations/rouen_annotation.json --results ../output/rouen/ann/yolo/annotations.json