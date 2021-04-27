"""
Author: Jorge García <sirsykon@gmail.com>
"""
"""
File to create test time augmentation functions to transform images in order to be applied to be processed by an object detection method.

The idea is to return a function to transform the images and a function to transform each object detected to the original position so we can get the detection on the original image coordinates.

All functions will assume computational point standard (height, width).

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
        h, w = point
        max_h, max_w, _ = image_shape
        max_h -= 1                      # First index is 0.
        max_w -= 1                      # First index is 0.
        if axis == "vertical":
            return [max_h - h, w]

        if axis == "horizontal":
            return [h, max_w - w]

        if axis == "both":
            return [max_h - h, max_w - w]

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
        h, w = point
        max_h, max_w, _ = image_shape
        max_h -= 1                      # First index is 0.
        max_w -= 1                      # First index is 0.
        if degrees == 90:
            return [w, max_h - h]

        if degrees == 180:
            return [max_h - h, max_w - w]

        if degrees == 270:
            return [max_w - w, h]

    return rotate, unrotate_point