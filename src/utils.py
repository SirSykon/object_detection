import cv2

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
        drawn_image = cv2.rectangle(drawn_image, (x,y), (x+width, y+height), (255,0,0), 5)

    return drawn_image

