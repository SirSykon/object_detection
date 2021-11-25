
from abstract_object_detector import Abstract_Object_Detector
from ..test_time_augmentation import test_time_augmentation

class Object_Detector(Abstract_Object_Detector):
    """
    Class to act as abstract class in order to create wrappers following the same structure.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
    """
    def __init__(self, backend, model, model_origin, transformations):
        """
        Method to initialize the model.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """

        if backend == "torch":
            from object_detectors.torch.faster_rcnn_object_detector import Faster_RCNN_Object_Detector
            from object_detectors.torch.yolo_object_detector import YOLO_Object_Detector
            from object_detectors.torch.ssd_object_detector import SSD_Object_Detector
        if backend == "tf":
            from object_detectors.tf.faster_rcnn_object_detector import Faster_RCNN_Object_Detector
            #from object_detectors.tf.yolo_object_detector import YOLO_Object_Detector
            #from object_detectors.tf.ssd_object_detector import SSD_Object_Detector
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

                self.transform_images_functions.append(t)
                self.untransform_point_functions.append(u)

        else:
            self.transform_images_functions = None
            self.untransform_point_functions = None

    def process_single_image(self, rgb_image):
        images_batch = []                          # We create the images batch.

        if not self.transform_images_functions is None:                         # We apply transformations if there is any.
            for trans in self.transform_images_functions:
                if trans:
                    images_batch.append(trans(rgb_image))
                else:
                    images_batch.append(rgb_image)

        preprocessed_images = self.object_detector.preprocess(images_batch)                             # We preprocess the batch.
        outputs = self.object_detector.process(preprocessed_images)                                     # We apply the model.
        outputs = self.object_detector.filter_output_by_confidence_treshold(outputs, treshold = 0.5)    # We filter output using confidence.
        
        if args.print_output_folder:
            for img_output, img_rgb, trans_name in zip(outputs, images_batch, args.transformations):                             # For outputs from each image...
                drawn_image = print_utils.print_detections_on_image(img_output, img_rgb[:,:,[2,1,0]])
                cv2.imwrite(os.path.join(args.print_output_folder, trans_name, frame_filename), drawn_image)

        untransformed_outputs = []
        for untransform_point_function, output in zip(self.untransform_point_functions,outputs):
            if untransform_point_function:
                _output = test_time_augmentation.untransform_coco_format_object_information(output, untransform_point_function, rgb_image.shape)
            else:
                _output = output
            untransformed_outputs.append(_output)

        clusters_objects = test_time_augmentation.create_clusters(untransformed_outputs,0.5)

        unified_output = test_time_augmentation.unify_clusters(clusters_objects)

        return unified_output
        
    def __call__(self, images):
        return self.process(images)