import test_time_augmentation
import cv2
import numpy as np
import utils
from object_detectors.faster_rcnn_torch_object_detector import Faster_RCNN_Torch_Object_Detector
import coco_format_utils

coco_format_utils.read_annotations_from_json_file("../../datasets/jodoin/rouen/annotations/rouen.json")

quit()
image = cv2.imread("../input/test/wp1919724.jpg")
image = cv2.resize(image, (1920,1080))

fod = Faster_RCNN_Torch_Object_Detector()

input_ = fod.preprocess(np.array([image]))

output = fod(input_)

transform, untransform_point = test_time_augmentation.create_rotation_transformation(180)
drawn_image = utils.print_detections_on_image(output[0], image)
cv2.imshow("drawn transformed", drawn_image)
cv2.waitKey(0)
quit()
untransformed_object_information = test_time_augmentation.untransform_coco_format_object_information(fake_object_information, untransform_point, drawn_image.shape)

drawn_image = utils.print_detections_on_image(untransformed_object_information, image)
cv2.imshow("drawn", drawn_image)
cv2.waitKey(0)
