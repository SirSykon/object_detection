#!/bin/sh
python create_annotations.py -a ../../input/food/output.xls -i ../../input/food/images -o ../../input/food/20_classes/xml_images -l ../../input/food/20_classes/label_map.pbtxt -n 20 -r 4
python partition_dataset.py -i ../../input/food/20_classes/xml_images -o ../../input/food/20_classes/dataset -r 0.1 -x
python generate_tf_record.py -x ../../input/food/20_classes/dataset/train -l ../../input/food/20_classes/label_map.pbtxt -o ../../input/food/20_classes/train.record -i ../../input/food/20_classes/dataset/train -c ../../input/food/20_classes/train.csv
python generate_tf_record.py -x ../../input/food/20_classes/dataset/test -l ../../input/food/20_classes/label_map.pbtxt -o ../../input/food/20_classes/test.record -i ../../input/food/20_classes/dataset/test -c ../../input/food/20_classes/test.csv
