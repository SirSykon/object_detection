import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)
    
    
def get_gt_labels_one_hot(gt_boxes, class_label_ids, label_id_offset = 1, images_np = None):

    images = []
    gt_labels_one_hot = []
    for index, (gt_box_np, gt_label_id) in enumerate(zip(gt_boxes, class_label_ids)):
        if images_np: images.append(np.expand_dims(images_np[index], axis=0))
        zero_indexed_groundtruth_label_id = gt_label_id - label_id_offset
        targets = np.array(zero_indexed_groundtruth_label_id).reshape(-1)
        one_hot_targets = np.eye(num_classes)[targets]
        gt_labels_one_hot.append(one_hot_targets)
        
    return gt_labels_one_hot, images
