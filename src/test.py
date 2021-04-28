import test_time_augmentation
import cv2
import utils

image = cv2.imread("../input/images/20151127_114556.jpg")
image = cv2.resize(image, (1920,1080))

fake_bboxes = [[500,350,350,400]]
fake_object_information = [fake_bboxes, [1,1], [0.1, 0.5]]

transform, untransform_point = test_time_augmentation.create_rotation_transformation(180)

drawn_image = utils.print_detections_on_image(fake_object_information, transform(image))
cv2.imshow("drawn transformed", drawn_image)
cv2.waitKey(0)

untransformed_object_information = test_time_augmentation.untransform_coco_format_object_information(fake_object_information, untransform_point, drawn_image.shape)

drawn_image = utils.print_detections_on_image(untransformed_object_information, image)
cv2.imshow("drawn", drawn_image)
cv2.waitKey(0)
