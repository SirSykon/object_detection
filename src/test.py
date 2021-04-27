import test_time_augmentation
import cv2

image = cv2.imread("../input/images/20151127_114556.jpg")

point = [35, 600]
image_shape = image.shape

transform, untransform_point = test_time_augmentation.create_rotation_transformation(90)
print(untransform_point(point, image_shape))

transform, untransform_point = test_time_augmentation.create_rotation_transformation(180)
print(untransform_point(point, image_shape))

transform, untransform_point = test_time_augmentation.create_rotation_transformation(270)
print(untransform_point(point, image_shape))