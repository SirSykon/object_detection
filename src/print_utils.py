import cv2


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

