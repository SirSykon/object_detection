import os
import cv2
import numpy as np
from typing import List

from ..test_time_augmentation import test_time_augmentation
from ..print_utils import print_utils


class Object_Detector():
    """
    Class to detect objects from images.
    """
    def __init__(self, backend:str, model:str, transformations:List[str]=["None"], model_origin:str="default"):
        """
        Method to initialize the model.

        Args:
            backend:str -> "tf" for Tensorflow or "torch" for PyTorch.
            model:str -> "faster", "ssd" or "yolo".
            transformations:List[str] -> transformations list. Options are "None", "flipH", "flipV", "rot90", "rot270", "rot180".
            model_origin:str -> if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model. Default "default".
        """

        if backend == "torch":
            from .torch.faster_rcnn_object_detector import Faster_RCNN_Object_Detector
            from .torch.yolo_object_detector import YOLO_Object_Detector
            from .torch.ssd_object_detector import SSD_Object_Detector
        if backend == "tf":
            from .tf.faster_rcnn_object_detector import Faster_RCNN_Object_Detector
            #from .tf.yolo_object_detector import YOLO_Object_Detector
            #from .tf.ssd_object_detector import SSD_Object_Detector
        # We will obtain the parameters.

        if model == 'faster':
            print("Loading Faster RCNN torch object detector")
            self.object_detector = Faster_RCNN_Object_Detector(model=model_origin)
        if model == 'ssd':
            print("Loading SSD torch object detector")
            self.object_detector = SSD_Object_Detector(model=model_origin)
        if model == 'yolo':
            print("Loading YOLO torch object detector")
            self.object_detector = YOLO_Object_Detector(model=model_origin)

        # We will generate a list of transformation and untransform point functions.
        if transformations:
            self.transform_images_functions = []
            self.untransform_point_functions = []
            self.transformations_names = []

            for trans_name in transformations:
                if trans_name == "flipH":
                    t,u = test_time_augmentation.create_flip_transformation("horizontal")
                if trans_name == "flipV":
                    t,u = test_time_augmentation.create_flip_transformation("vertical")
                if trans_name == "rot90":
                    t,u = test_time_augmentation.create_rotation_transformation(90)
                if trans_name == "rot270":
                    t,u = test_time_augmentation.create_rotation_transformation(270)
                if trans_name == "rot180":
                    t,u = test_time_augmentation.create_rotation_transformation(180)
                if trans_name == "None":
                    t,u = None, None

                self.transformations_names.append(trans_name)
                self.transform_images_functions.append(t)
                self.untransform_point_functions.append(u)

        else:
            self.transform_images_functions = None
            self.untransform_point_functions = None

    def process_images(self, rgb_images:List[np.ndarray], print_output_folders:List[str]=None, frame_filenames:List[str]=None):
        """
        Method to process various images in order to detect objects. If this object detector has transformations, they will be applied.

        Args:
            rgb_images:List[np.ndarray] -> List of matrix with the images to be processed in order to detect objects.
            print_output_folders:List[str] -> Output folders path to save images with partial tansformations. Default None.
            frame_filenames:List[str] -> Names of the files to save the partial transformations images. Default None.
        """
        outputs = []

        for index, rgb_image in enumerate(rgb_images):
            if not print_output_folders is None and not frame_filenames is None:
                print_output_folder = print_output_folders[index]
                frame_filename = frame_filenames[index]
            else:
                print_output_folder = None
                frame_filename = None
            outputs.append(self.process_single_image(rgb_image, print_output_folder=print_output_folder, frame_filename=frame_filename))

        return outputs

    def process_single_image(self, rgb_image:np.ndarray, print_output_folder:str=None, frame_filename:str=None) -> List:
        """
        Method to process a single image in order to detect objects. If this object detector has transformations, they will be applied.

        Args:
            rgb_image:np.ndarray -> Matrix with the image to be processed in order to detect objects.
            print_output_folder:str -> Output folder path to save images with partial tansformations. Default None.
            frame_filename:str -> Name of the filename to save the partial transformations images. Default None.
        """

        images_batch = []                                                       # We create the images batch.

        if not self.transform_images_functions is None:                         # We apply transformations if there is any.
            for trans in self.transform_images_functions:
                if trans:
                    images_batch.append(trans(rgb_image))
                else:
                    images_batch.append(rgb_image)

        preprocessed_images = self.object_detector.preprocess(images_batch)                             # We preprocess the batch.
        outputs = self.object_detector.process(preprocessed_images)                                     # We apply the model.
        outputs = self.object_detector.filter_output_by_confidence_treshold(outputs, treshold = 0.5)    # We filter output using confidence.
        
        if not print_output_folder is None:
            assert not frame_filename is None
            for img_output, img_rgb, trans_name in zip(outputs, images_batch, self.transformations_names):    # For outputs from each image...
                drawn_image = print_utils.print_detections_on_image(img_output, img_rgb[:,:,[2,1,0]])
                cv2.imwrite(os.path.join(print_output_folder, trans_name, frame_filename), drawn_image)

        untransformed_outputs = []
        for untransform_point_function, output in zip(self.untransform_point_functions,outputs):
            if untransform_point_function:
                _output = test_time_augmentation.untransform_coco_format_object_information(output, untransform_point_function, rgb_image.shape)
            else:
                _output = output
            untransformed_outputs.append(_output)

        clusters_objects = test_time_augmentation.create_clusters(untransformed_outputs,0.5)            # We nnify transformations.

        unified_output = test_time_augmentation.unify_clusters(clusters_objects)

        return unified_output

    def __call__(self, images):
        return self.process_images(images)
