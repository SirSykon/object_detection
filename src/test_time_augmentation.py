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
        Function: Function that takes point from a flipped image and transforms it to the original image position.
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
        Function: Function that takes point from a rotated image and transforms it to the original image position.
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

        # We don't know which transformation was applied so we don't know wich untramsformation has been applied.
        # so we need to obtain the top-left corner (the minimum height and width) and the maximum one.
        aux_array = np.array([untransformed_initial_top_left, untransformed_initial_top_right, untransformed_initial_bottom_left, untransformed_initial_bottom_right])
        print(aux_array)
        untransformed_top_left = aux_array.min(0)       # We get the minimum positions.
        untransformed_bottom_right = aux_array.max(0)   # We get the maximum positions.
        width_and_height = untransformed_bottom_right-untransformed_top_left

        new_bbox = [untransformed_top_left[0], untransformed_top_left[1], width_and_height[0], width_and_height[1]]

    new_bboxes.append(new_bbox)

    return [new_bboxes, classes, confidences]

def create_composition_of_transformations(input_functions):
    """Function to create a composition of transformations.

    Args:
        input_functions (list): list with the following shape:
            [transformation_function_1, untransform_point_function_1, transformation_function_2, untransform_point_function_2, ..., transformation_function_n, untransform_point_function_n]

    Returns:
        Function: Function that takes an image and applies all transformation function to it.
        Function: Function that takes point from an image with all transformation functions applied to it and transforms it to the original image position
    """

    transformations = input_functions[::2]
    untransformations = input_functions[1::2]

    assert len(transformations) == len(untransformations)

    def transformation_composition(image):
        aux_image = image.copy()
        for trans in transformations:
            aux_image = trans(aux_image)

        return aux_image

    def untransformation_composition(point):
        aux_point = point.copy()
        for untrans in reversed(untransformations):
            aux_point = untrans(aux_point)

        return aux_point

    return transformation_composition, untransformation_composition

def unify_test_time_augmentation_outputs(detected_objects_coco_format_list):
    """

    """
    return detected_objects_coco_format_list
    

#  Felzenszwalb et al. from https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
def non_max_suppression_slow(bboxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = bboxes[:,0]
	y1 = bboxes[:,1]
	x2 = bboxes[:,0] + bboxes[:,2]
	y2 = bboxes[:,1] + bboxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return pick