import cv2
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
import random

colors_dict = {
    1 : (0,0,0),
    2 : (0,0,50),
    3 : (0,0,75),
    4 : (0,0,100),
    5 : (0,50,0),
    6 : (0,75,0),
    7 : (0,100,0),
    8 : (50,0,0),
}

def print_detections_on_image(object_detection_information, image, bboxes_format = "coco"):
    """[summary]

    Args:
        object_detection_information ([bboxes, classes, confidences]): Objects detected.
        image (np.ndarray): The image.
        bboxes_format (str) : "coco" is the only format initially avaliable.
    """

    assert bboxes_format == "coco"

    drawn_image = image.copy()

    [bboxes, classes, confidences] = object_detection_information

    for bbox, _class, confidence in zip(bboxes, classes, confidences):
        [x, y, width, height] = bbox
        color = colors_dict[_class] if _class in colors_dict.keys() else (255,0,0)
        drawn_image = cv2.rectangle(drawn_image, (x,y), (x+width, y+height), color, 5)
    return drawn_image

def print_points_on_image(points:List[List[int]], image:np.ndarray, colors:Dict[int,Tuple[int]]) -> np.ndarray:
    """
    points:List[List[int]]
    image:np.ndarray
    colors:Dict[int,List[int]]
    """

    drawn_image = image.copy()
    for point, color in zip(points, colors):
        print("-")
        print(point)
        print((point[0], point[1]))
        drawn_image = cv2.circle(drawn_image, (point[0], point[1]), 5, color, -1)
    return drawn_image

def print_info_on_image(infos:List[str], points:List[List[int]], image:np.ndarray, colors:Dict[int,Tuple[int]]) -> np.ndarray:
    """
    points:List[List[int]]
    image:np.ndarray
    colors:Dict[int,List[int]]
    """

    drawn_image = image.copy()
    for point, info, color in zip(points, infos, colors):

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (point[0], point[1])
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        cv2.putText(drawn_image,info, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        print("-")
        print(point)
        print((point[0], point[1]))

    return drawn_image


def get_random_color(colormap:str='cool', number_of_colors:int=100, normalized:bool=False)-> Dict[int,Tuple[int]]:
    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(np.array(range(number_of_colors)))
    rand = random.randint(0, number_of_colors-1)
    color = colors[rand]
    if not normalized:
        color = color*255.
    return (color[0], color[1], color[2])

