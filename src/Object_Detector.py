"""
Author: Jorge García <sirsykon@ŋmail.com>
"""


class Object_Detector():
    """
    Class to serve as abstract class in order to create wrappers following the same structure.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
    """
    def __input__(self, bbox_format:str="coco") -> None:
        """
        Method to initialize the network.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
        """
        self.bbox_format

    def set_bbox_format(self, bbox_format:str) -> None:
        """
        Args:
            bbox_format (str): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
        """

    def process_output_bbox_format_coco(self, images:list(np.ndarray)) -> list([list(float), list(int), list(float)]):
        """[summary]

        Args:
            images : list(np.ndarray): List of images.

        Returns:
            The output is a list containing the object detection information for each image as follows:
                output : list(objec_detection_information)
                objec_detection_information : [bboxes, classes, confidences]
                bboxes defines the position of the object. its structure changes according to bbox_format as follows:
                    [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                classes : list(int) 
                confidences: list(float)
        """
        raise(NotImplementedError)


    def process(self, images:list(np.ndarray)) -> list([list(float), list(int), list(float)]):
        """[summary]

        Args:
            images : list(np.ndarray): List of images.

        Returns:
            The output is a list containing the object detection information for each image as follows:
                output : list(objec_detection_information)
                objec_detection_information : [bboxes, classes, confidences]
                bboxes : list(bbox) defines the position of the object. bbox structure changes according to bbox_format as follows:
                    "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                    "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                    "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. 
                classes : list(int) 
                confidences : list(float)
        """
        output = process_output_bbox_format_coco(images)
        if not self.bbox_format == "coco":
            new_output = []
            for [coco_format_bboxes, classes, confidences] in output:
                new_format_bboxes = []
                for coco_format_bbox in coco_format_bboxes
                    x1, y1, width, height = coco_format_bbox
                    if self.bbox_format == "relative":

                        assert self.shape
                        relative_format_bbox = [x1 / self.shape[1],y1 / self.shape[0], (x1 + width) / self.shape[1], (y1 - height) / self.shape[0]]
                        new_format_bboxes.append(relative_format_bbox)

                    if self.bbox_format == "absolute":

                        absolute_format_bbox = [x1, y1, x1+width, y1+height]
                        new_format_bboxes.append(relative_format_bbox)

                new_output.append([new_format_bboxes, classes, confidences])

        else:
            return output




    def __call__(self, images):
        return self.process(images)



    

