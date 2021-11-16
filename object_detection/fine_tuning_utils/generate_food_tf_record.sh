#!/bin/sh
python create_food_annotations.py -a ../../input/food/output.xls -i ../../input/food/images -o ../../input/food/44_classes/xml_images -l ../../input/food/44_classes/label_map.pbtxt -n 44 -r 4
python partition_dataset.py -i ../../input/food/44_classes/xml_images -o ../../input/food/44_classes/dataset -r 0.1 -x
python generate_tf_record.py -x ../../input/food/44_classes/dataset/train -l ../../input/food/44_classes/label_map.pbtxt -o ../../input/food/44_classes/train.record -i ../../input/food/44_classes/dataset/train -c ../../input/food/44_classes/train.csv
python generate_tf_record.py -x ../../input/food/44_classes/dataset/test -l ../../input/food/44_classes/label_map.pbtxt -o ../../input/food/44_classes/test.record -i ../../input/food/44_classes/dataset/test -c ../../input/food/44_classes/test.csv
python ../xml_to_coco_json.py -l ../../input/food/44_classes/label_map.pbtxt -xi ../../input/food/44_classes/dataset/val -ii "../../input/food/44_classes/dataset/val/*.jpg" -o ../../input/food/44_classes/val_ann.json
