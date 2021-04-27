#!/bin/sh
echo "This script will ask for some paths in order to prepare the data to be used to fine-tune a network."
echo "Please insert the folder with the images and their .xml annotations."
read images_folder
python fine_tuning_utils/partition_dataset.py -x -i $images_folder -r 0.1
python fine_tuning_utils/generate_tf_record.py 
