import torch
import torchvision
import numpy as np
from .object_detector import Object_Detector
import time

# Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

class Faster_RCNN_Torch_Object_Detector(Object_Detector):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def process_output_bbox_format_coco(self, images):
        """[summary]

        Args:
            images (torch.tensor): [description]
        """
        # We ensure images is a torch tensor with channel-first.
        assert type(images) == torch.Tensor
        assert list(images.size())[1] == 3

        with torch.no_grad():
            outputs = self.model(images)

        outputs = turn_faster_rcnn_outputs_into_coco_format(outputs, list(images.size())[1:])
        return outputs
    
    def preprocess(self, images):
        """ Function to preprocess a batch of images.

        Args:
            images (np.ndarray): batch of images between 0 and 255. Expected dimensions are (batch, height, width, channel) or (batch, channel, height, width).

        Returns:
            A torch tensor with the preprocessed images as self.model expects.
        """

        assert len(images.shape) == 4
        # We copy the images np array to ensure they are not destroyed.
        imgs = images.copy()
        if imgs.shape[-1] == 3: # We guess the last dimensions is channels.                
            # We need channel first so we transform the data.
            imgs = np.moveaxis(imgs,-1, 1)
        # We normalize.
        imgs = imgs/255.
        # We ceate a tensor with dtype float32.
        return torch.tensor(imgs, dtype=torch.float32)

def turn_faster_rcnn_outputs_into_coco_format(all_images_output, image_shape):
    """Funtion to turn torch faster rcnn output into coco format.

    Args:
        all_images_output (dictionary): A dictionary with the following structure:
            {"boxes" : list(box),
            labels : list(int),
            scores : list(float)}
        with  box = [x1,y1,x2,y2] for each object with (x1,y1) as topp-left corner and (x2,y2) with bottom-right corner-

    Returns:

    """
    initial_time = time.time()
    new_all_images_output = []
    _, height, width = image_shape
    for img_output in all_images_output:
        bboxes = img_output["boxes"].numpy()
        labels = img_output["labels"].numpy()
        scores = img_output["scores"].numpy()
        bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
        bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
        new_all_images_output.append([bboxes, labels, scores])
    
    print(time.time()-initial_time)
    return new_all_images_output