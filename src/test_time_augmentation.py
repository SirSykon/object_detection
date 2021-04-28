"""
Author: Jorge García <sirsykon@gmail.com>
"""
"""
File to create test time augmentation functions to transform images in order to be applied to be processed by an object detection method.

The idea is to return a function to transform the images and a function to transform each object detected to the original position so we can get the detection on the original image coordinates.

All functions will assume mathematical point standard (width, height).

"""

import numpy as np

def create_flip_transformation(axis):
    """Function to create an image flip transformation and the function to transpose information from one image to the original.

    Args:
        axis (str): "horizontal", "vertical" or "both". Defines the axis used to flip de image so 
                    "horizontal" implies the image would be flipped left to right (top-left corner would be top-right),
                    "vertical" implies the image would be flipped top to bottom (top-left corner would be bottom-left) and horizontal both so the top-left corner would be the bottom-right.

    Returns:
        Function: Function that takes an image and flips it.
        Function: Function that takes information from a flipped image and transforms it to the original image position.
    """

    assert axis in ["horizontal", "vertical", "both"]

    def flip(image):
        if axis == "vertical":
            return np.flip(image, 0)

        if axis == "horizontal":
            return np.flip(image, 1)

        if axis == "both":
            return np.flip(image, (0,1))

    def unflip_point(point, image_shape):
        x, y = point
        max_h, max_w, _ = image_shape
        max_h -= 1                      # First index is 0.
        max_w -= 1                      # First index is 0.
        if axis == "vertical":
            return [x, max_h - y]

        if axis == "horizontal":
            return [max_w - x, y]

        if axis == "both":
            return [max_w - x, max_h - y]

    return flip, unflip_point

def create_rotation_transformation(degrees):
    """Function to create an image rotation transformation and the function to transpose information from one image to the original.

    Args:
        degrees (int): number of degrees to rotate. 90, 180 or 270.

    Returns:
        Function: Function that takes an image and rotate it.
        Function: Function that takes information from a rotated image and transforms it to the original image position.
    """

    assert degrees in [90, 180, 270]

    def rotate(image):
        if degrees == 90:
            rot_times = 1

        if degrees == 180:
            rot_times = 2

        if degrees == 270:
            rot_times = 3

        return np.rot90(image, rot_times)

    def unrotate_point(point, image_shape):
        x, y = point
        max_h, max_w, _ = image_shape
        max_h -= 1                      # First index is 0.
        max_w -= 1                      # First index is 0.
        if degrees == 90:
            return [max_h - y, x]

        if degrees == 180:
            return [max_w - x, max_h - y]

        if degrees == 270:
            return [y, max_w - x]

    return rotate, unrotate_point


def untransform_coco_format_object_information(object_detection_information, untransform_point_function, image_shape):
    """Function to transform an object detection information to undo transformation over the position.

    Args:
        objec_detection_information : [bboxes, classes, confidences]
                bboxes : list(bbox) defines the positions of the objects. bbox structure is as folows:
                    [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                classes : list(int) 
                confidences: list(float)
    """

    bboxes, classes, confidences = object_detection_information
    new_bboxes = []
    for bbox in bboxes:
        [x,y,width,height] = bbox
        initial_top_left = [x,y]
        initial_top_right = [x+width,y]
        initial_bottom_left = [x,y+height]
        initial_bottom_right = [x+width,y+height]

        untransformed_initial_top_left = untransform_point_function(initial_top_left, image_shape)
        untransformed_initial_top_right = untransform_point_function(initial_top_right, image_shape)
        untransformed_initial_bottom_left = untransform_point_function(initial_bottom_left, image_shape)
        untransformed_initial_bottom_right = untransform_point_function(initial_bottom_right, image_shape)

        # We don't know wich transformation was applied so we don't know wich untramsformation has been applied.
        # so we need to obtain the top-left corner (the minimum height and width) and the maximum one.
        aux_array = np.array([untransformed_initial_top_left, untransformed_initial_top_right, untransformed_initial_bottom_left, untransformed_initial_bottom_right])
        print(aux_array)
        untransformed_top_left = aux_array.min(0)       # We get the minimum positions.
        untransformed_bottom_right = aux_array.max(0)   # We get the maximum positions.
        width_and_height = untransformed_bottom_right-untransformed_top_left

        new_bbox = [untransformed_top_left[0], untransformed_top_left[1], width_and_height[0], width_and_height[1]]

    new_bboxes.append(new_bbox)

    return [new_bboxes, classes, confidences]


