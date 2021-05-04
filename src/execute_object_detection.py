import argparse
import os
import cv2
import numpy as np
from object_detectors import faster_rcnn_torch_object_detector
import utils
import test_time_augmentation

parser = argparse.ArgumentParser(description='Script to execute object detection to some video.')
parser.add_argument('-m', '--model', help="Object Detection model to use. Current options: 'faster'")
parser.add_argument('-v', '--video', help="Path to video.")
parser.add_argument('-t', '--transformations', nargs = '+', default=None, help= "transformations set. Expected list as follows: 'flipH flipV rot90'")

args = parser.parse_args()

# We will obtain the parameters.

if args.model == 'faster':
    print("Loading Faster RCNN torch object detector")
    object_detector = faster_rcnn_torch_object_detector.Faster_RCNN_Torch_Object_Detector()

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

        transform_images_functions.append(t)
        untransform_point_functions.append(u)

vidcap = cv2.VideoCapture(args.video)

# Main loop
image_index = 0
succes, image = vidcap.read()           # We try to read the next image
while succes:   # While there is a next image.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images_batch = [rgb_image]

    if args.transformations:
        for trans in transform_images_functions:
            images_batch.append(trans(rgb_image))

    preprocessed_images = object_detector.preprocess(np.array(images_batch))
    objects = object_detector.process(preprocessed_images)
    objects = object_detector.filter_output_by_confidence_treshold(objects)

    drawn_image = utils.print_detections_on_image(objects[0], image)
    cv2.imshow("drawn", drawn_image)
    cv2.waitKey(0)

    succes, image = vidcap.read()           # We try to read the next image
    image_index += 1

    quit()



