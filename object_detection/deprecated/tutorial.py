import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from glob import glob
import pathlib
import GPU_utils

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(0)
# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

# Root directory of object detection
TENSORFLOW_ROOT_DIR = os.path.abspath("../../Tensorflow/")
TENSORFLOW_RESEARCH_ROOT_DIR = os.path.join(TENSORFLOW_ROOT_DIR, "models/research/")
print(TENSORFLOW_RESEARCH_ROOT_DIR)
sys.path.append(TENSORFLOW_RESEARCH_ROOT_DIR)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODELS_PATH = os.path.join(TENSORFLOW_ROOT_DIR, "downloaded_models")

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    base_url = MODELS_PATH
    """
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    """

    model_dir = os.path.join(base_url, model_name, "saved_model")
    print(model_dir)
    model = tf.saved_model.load(str(model_dir))

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def run_inference(model, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #image_np = np.array(Image.open(image_path))
    image_np = cv2.imread(image_path)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    print(output_dict)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow("image", image_np)
    cv2.waitKey(0)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(TENSORFLOW_RESEARCH_ROOT_DIR, "object_detection/data/mscoco_label_map.pbtxt")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = os.path.join(TENSORFLOW_RESEARCH_ROOT_DIR, "object_detection/test_images")
TEST_IMAGE_PATHS = sorted(glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.jpg")))

print(TEST_IMAGE_PATHS)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model_name = 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8'
detection_model = load_model(model_name)

# Check the model's input signature, it expects a batch of 3-color images of type uint8:
"""
print(detection_model.signatures['serving_default'].inputs)

print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)
"""

show_inference(detection_model, TEST_IMAGE_PATHS[1])