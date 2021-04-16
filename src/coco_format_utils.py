import numpy as np
import json

def Coco_Annotation_Set():
    """ 
    Class to contain a set of Coco annotations.
    """
    def __init__(self, json_file_path:str = None):
        self.annotations = []
        if json_file_path:
            with open(path) as json_file:
                data = json.load(json_file)
                for input_ann in data:
                    if input_ann["iscrowd"] == 1:
                        print("We work assuming iscrowd is 0.")
                        raise(NotImplementedError)

                    ann = Coco_Annotation_Object(
                        id = input_ann["id"],
                        segmentation = input_ann["segmentation"],
                        area = input_ann["area"],
                        iscrowd = input_ann["iscrowd"],
                        image_id = input_ann["image_id"],
                        bbox = input_ann["bbox"]
                        category_id = input_ann["category_id"]
                    )
                    self.annotations.append(ann)


def Coco_Annotation_Object():

    """
    Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    Since initially this repository is to work with object detection methods, some attributes could be ignored.

    Example of annotations list: 

    The following JSON shows 2 different annotations.

    The first annotation:

        Has a segmentation list of vertices (x, y pixel positions)

        Has an area of 702 pixels (pretty small) and a bounding box of [473.07,395.93,38.65,28.67]

        Is not a crowd (meaning it’s a single object)

        Is category id of 18 (which is a dog)

        Corresponds with an image with id 289343 (which is a person on a strange bicycle and a tiny dog)

    The second annotation:

        Has a Run-Length-Encoding style segmentation

        Has an area of 220834 pixels (much larger) and a bounding box of [0,34,639,388]

        Is a crowd (meaning it’s a group of objects)

        Is a category id of 1 (which is a person)

        Corresponds with an image with id 250282 (which is a vintage class photo of about 50 school children)

    "annotations": [
        {
            "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
            "area": 702.1057499999998,
            "iscrowd": 0,
            "image_id": 289343,
            "bbox": [473.07,395.93,38.65,28.67],
            "category_id": 18,
            "id": 1768
        },
        ...
        {
            "segmentation": {
                "counts": [179,27,392,41,…,55,20],
                "size": [426,640]
            },
            "area": 220834,
            "iscrowd": 1,
            "image_id": 250282,
            "bbox": [0,34,639,388],
            "category_id": 1,
            "id": 900100250282
        }
    ]

    Args:
        bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
            "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
            "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
            "relative" -> []. Defaults to None.
        category_id (int): Cateogry identification number. Defaults to None.
        id (int, optional): Object identification number. Defaults to None.
        segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
        area (float, optional): Area accupied by the object. Defaults to None.
        iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
        image_id (int, optional): Image identification number. Defaults to None.
        image_shape (list, optional): Image object shape. Defaults to None.
        bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".

    """

    def __init__(self, bbox:list, category_id:int, id:int = None, segmentation:list = None, area:float = None, iscrowd:int = 0, image_id:int = None, image_shape:list = None, bbox_format:str = "coco") -> Coco_Annotation_Object:
        """
        Args:
            bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
                "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. 
                Defaults to None.
            category_id (int): Cateogry identification number. Defaults to None.
            id (int, optional): Object identification number. Defaults to None.
            segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
            area (float, optional): Area accupied by the object. Defaults to None.
            iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
            image_id (int, optional): Image identification number. Defaults to None.
            image_shape (list, optional): Image object shape. Defaults to None.
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".

        Returns:
            Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
            Since initially this repository is to work with object detection methods, some attributes could be ignored.
        """
        self.segmentation = np.array(segmentation)
        self.area = area
        if iscrowd == 1:
            print("WARNING: Current Coco_Format_Annotation class asusumes iscrowd is 0 (False).")
            self.iscrowd = True
        else:
            self.iscrowd = False
        self.image_id = image_id
        self.bbox = np.array(bbox)
        self.category_id = category_id
        self.id = id

        self.bbox_format = bbox_format
        self.image_shape = image_shape

        if np.max(self.bbox) < 1:
            "If there is no"

    def __str__(self):
        return str(self.to_dict())

    def set_image_shape(self, input_shape:list[int]) -> None:
        """Method to set image shape where this object appears.

        Args:
            input_shape (list[int]): Shape.

        Returns:
            None
        """
        self.image_shape = input_shape

    def to_coco_dict(self) -> dict[str, object]:
        """Method to turn this annotation into a dictionary with only coco data.

        Returns:
            dict[str, object]: dictionary with only coco data.
        """
        d = {
            "segmentation" : self.segmentation.tolist(),
            "area" :  self.area,
            "iscrowd" : self.iscrowd,
            "image_id" : self.image_id,
            "bbox" : self.bbox.tolist(),
            "category_id" : self.category_id,
            "id" : self.id
            }

        return d

    def get_corners_relative_format(self) -> list:
        """
        Returns:
            list: [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. Defaults to None.
        """

        if self.bbox_format == "coco":
            assert self.shape

            x1, y1, width, height = self.bbox
            return [x1 / self.shape[1],y1 / self.shape[0], (x1 + width) / self.shape[1], (y1 - height) / self.shape[0]]
        
        if self.bbox_format == "relative":
            return self.bbox_format.tolist()

        if self.bbox_format == "absolute":
            assert self.shape

            x1, y1, x2, y2 = self.bbox
            bbox = [x1 / self.shape[1],y1 / self.shape[0], x2 / self.shape[1], y2 / self.shape[0]]

    def get_corners_absolute_format(self) -> list:
        """
        Returns:
            list: [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2].
        """
        
        if self.bbox_format == "coco":
            x1, y1, width, height = self.bbox
            return [x1, y1, x1+width, y1+height]
        
        if self.bbox_format == "relative":
            assert self.shape

            x1, y1, x2, y2 = self.bbox
            bbox = [x1 * self.shape[1],y1 * self.shape[0], x2 * self.shape[1], y2 * self.shape[0]]

        if self.bbox_format == "absolute":
            return self.bbox_format.tolist()

    def get_corners_coco_format(self) -> list:
        """
        Returns:
            list: [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
        """
        
        if self.bbox_format == "coco":
            return self.bbox_format.tolist()
        
        if self.bbox_format == "relative":

            x1, y1, x2, y2 = self.get_corners_absolute_format()
            bbox = [x1, y1, x2-x1, y2-y1]

        if self.bbox_format == "absolute":
            x1, y1, x2, y2 = self.bbox
            return [x1, y1, x2-x1, y2-y1]

    def get_bbox_corners(self, format:str = "coco") -> list:
        """Method to get corners.

        Args:
            format (str, optional): format defines the corners format. Defaults to "coco".

        Returns:
            list :
                It dependes on format: 
                "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. Defaults to None.
        """

        if format == "coco":
            return self.get_corners_coco_format()
        
        if format == "relative":
            return self.get_corners_relative_format()

        if format == "absolute":
            return self.get_corners_absolute_format()

def new_object(bbox:list[float], category_id:int, id:int = None, segmentation:list = None, area:float = None, iscrowd:int = 0, image_id:int = None, image_shape:list[int] = None, bbox_format:str = "coco") -> Coco_Annotation_Object:
    """
    Args:
        bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
            "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
            "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
            "relative" -> []. Defaults to None.
        category_id (int): Cateogry identification number. Defaults to None.
        id (int, optional): Object identification number. Defaults to None.
        segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
        area (float, optional): Area accupied by the object. Defaults to None.
        iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
        image_id (int, optional): Image identification number. Defaults to None.
        image_shape (list, optional): Image object shape. Defaults to None.
        bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".

    Returns:
        Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
        Since initially this repository is to work with object detection methods, some attributes could be ignored.
    """

    return Coco_Annotation_Object(bbox, category_id, id=id, segmentation=segmentation, area=area, iscrowd=iscrowd, image_id = image_id, image_shape = image_shape, bbox_format = bbox_format)

def object_from_dict(self, input_dict:dict[str, object]) -> Coco_Annotation_Object:
    """Method to generate a coco format annotation from a dictionary.

    Args:
        input_dict (dict[str, object]): dict with coco annotation format data.

    Returns:
        Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
        Since initially this repository is to work with object detection methods, some attributes could be ignored.
    """

    return new_object(
        np.array(input_dict["bbox"]),
        input_dict["category_id"],
        id = input_dict["id"],
        segmentation = np.array(input_dict["segmentation"]), 
        area = input_dict["area"], 
        iscrowd = input_dict["iscrowd"],
        image_id = input_dict["image_id"]
    )