
import numpy as np
import os
from ..object_detector import Object_Detector
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

class Faster_RCNN_Object_Detector(Object_Detector):
    def __init__(self, model="default"):
        """Constructor.
        Args:
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        super().__init__(model=model)
        if model == "default":
            raise(NotImplementedError)
        else:
            configs = config_util.get_configs_from_pipeline_file(os.path.join(model, "pipeline.config"))
            model_config = configs['model']
            detection_model = model_builder.build(model_config=model_config, is_training=False)

            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(os.path.join(model,"ckpt-0")).expect_partial()

            self.model = detection_model

    def process_output_bbox_format_coco(self, images):
        """Function to apply object detection method and obtain detections with coco structure. 

        Args:
            images (torch.tensor): torch tensor batch of images as shown in Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

        Returns:
            Objects detections with coco structure.
        """
        # We ensure images is a torch tensor with channel-first.

        outputs = self.model(images)
        print(output)
        #outputs = self.turn_faster_rcnn_outputs_into_coco_format(outputs)   # We need to manipulate the output.
        return outputs

    
    def preprocess(self, images):
        """ Function to preprocess a batch of images.

        Args:
            images (list(np.ndarray)): batch of images between 0 and 255. Expected dimensions are (batch, height, width, channel) or (batch, channel, height, width).

        Returns:
            A torch tensor with the preprocessed images as self.model expects.
        """  
        assert len(images[0].shape) == 3

        # We ceate a tensor with dtype float32.
        tf_tensor = tf.convert_to_tensor(np.array(images), dtype=tf.float32)

        # We normalize.
        #imgs = imgs/255.

        return tf_tensor

    def turn_faster_rcnn_outputs_into_coco_format(self, all_images_output):
        """Funtion to turn torch faster rcnn output into coco format.

        Args:
            all_images_output (dictionary): A dictionary with the following structure:
                {"boxes" : list(box),
                labels : list(int),
                scores : list(float)}
            with  box = [x1,y1,x2,y2] for each object with (x1,y1) as topp-left corner and (x2,y2) with bottom-right corner-

        Returns:

        """
        new_all_images_output = []
        for img_output in all_images_output:    # For each image objects detections...
            # We get the information.
            bboxes = img_output["boxes"]
            labels = img_output["labels"]
            scores = img_output["scores"]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            if self.to == "cuda":
                bboxes = bboxes.to("cpu")
                labels = labels.to("cpu")
                scores = scores.to("cpu")
            
            bboxes = np.int32(bboxes.numpy())
            labels = np.int32(labels.numpy())
            scores = scores.numpy()
            new_all_images_output.append([bboxes, labels, scores])
        
        return new_all_images_output
