import argparse
import os
import cv2
import time
import numpy as np
from object_detectors import faster_rcnn_torch_object_detector, ssd_object_detector, yolo_object_detector
import print_utils
import test_time_augmentation
import coco_format_utils

parser = argparse.ArgumentParser(description='Script to execute object detection to some video.')
parser.add_argument('-m', '--model', help="Object Detection model to use. Current options: 'faster'")
parser.add_argument('-v', '--video', help="Path to video.")
parser.add_argument('-t', '--transformations', nargs = '+', default=None, help= "Transformations set. Expected list as follows: 'flipH flipV rot90'")
parser.add_argument('-o', '--output', default='../output/annotation.json', help= 'Path to output file to write obtained annotations as coco')
parser.add_argument('-p', '--print_output_folder', default = None, help='We print output if not None. Example: ../output/images')

args = parser.parse_args()

# We will obtain the parameters.

if args.model == 'faster':
    print("Loading Faster RCNN torch object detector")
    object_detector = faster_rcnn_torch_object_detector.Faster_RCNN_Torch_Object_Detector()
if args.model == 'ssd':
    print("Loading SSD torch object detector")
    object_detector = ssd_object_detector.SSD_Object_Detector()
if args.model == 'yolo':
    print("Loading YOLO torch object detector")
    object_detector = yolo_object_detector.YOLO_Object_Detector()

if not os.path.isfile(args.video):
    print(f"{args.video} is not file.")

if args.transformations:
    transform_images_functions = []
    untransform_point_functions = []
    for trans_name in args.transformations:
        print(trans_name)
        if trans_name == "flipH":
            t,u = test_time_augmentation.create_flip_transformation("horizontal")
        if trans_name == "flipV":
            t,u = test_time_augmentation.create_flip_transformation("vertical")
        if trans_name == "None":
            t,u = None, None

        transform_images_functions.append(t)
        untransform_point_functions.append(u)

if not os.path.isdir(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))

if args.print_output_folder:                                   # De we print outputs?
    if not os.path.isdir(args.print_output_folder):            # We will ensure the existence of the output folder.
        os.makedirs(args.print_output_folder)

coco_annotations = coco_format_utils.Coco_Annotation_Set()
print(coco_annotations)
vidcap = cv2.VideoCapture(args.video)

# Main loop
image_index = 1
success, image = vidcap.read()           # We try to read the next image
obj_id = 1

while success:   # While there is a next image.

    frame_filename = "frame_{:0>6}.png".format(image_index)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images_batch = []                          # We create the images batch.

    initial_time = time.time()

    if args.transformations:                            # We apply transformations if there is any.
        for trans in transform_images_functions:
            if trans:
                images_batch.append(trans(rgb_image))
            else:
                images_batch.append(rgb_image)

    preprocessed_images = object_detector.preprocess(np.array(images_batch))        # We preprocess the batch.
    outputs = object_detector.process(preprocessed_images)                          # We apply the model.
    outputs = object_detector.filter_output_by_confidence_treshold(outputs)         # We filter output using confidence.
    #print(outputs)

    coco_object_list = []   # List to contain the set of coco object from all images.

    # Now we create the coco format annotation for this image.
    for img_output, img_rgb in zip(outputs, images_batch):                                  # For outputs from each image...
        for bbox, _class, confidence in zip(img_output[0], img_output[1], img_output[2]):   
            #print(bbox)
            #print(_class)
            #print(confidence)
            coco_object = coco_format_utils.Coco_Annotation_Object(bbox=bbox, category_id=_class, id=obj_id, image_id=image_index, score=confidence)
            coco_annotations.insert_coco_annotation_object(coco_object)
            obj_id+=1

        if args.print_output_folder:
            drawn_image = print_utils.print_detections_on_image(img_output, img_rgb[:,:,[2,1,0]])
            cv2.imwrite(os.path.join(args.print_output_folder, frame_filename), drawn_image)

    print(f"process time {time.time()-initial_time}")

    success, image = vidcap.read()           # We try to read the next image
    image_index += 1

coco_annotations.to_json(args.output)