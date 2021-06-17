import argparse
import os
import cv2
import time
import numpy as np
from glob import glob

import print_utils
import test_time_augmentation
import coco_format_utils

def generator_from_video(video_path):
    vidcap = cv2.VideoCapture(video_patho)
    success, image = vidcap.read()           # We try to read the next image
    while success:
        yield image
    
def generator_from_files_input_format(files_format):
    input_files = sorted(glob(files_format))
    for image_path in input_files:
        image = cv2.imread(image_path)
        yield image

def main(args):
    if args.backend == "torch":
        from object_detectors import torch.faster_rcnn_object_detector, torch.ssd_object_detector, torch.yolo_object_detector as faster_rcnn_object_detector, ssd_object_detector, yolo_object_detector
    if args.backend == "tf":
        from object_detectors import tf.faster_rcnn_object_detector, tf.ssd_object_detector, tf.yolo_object_detector as faster_rcnn_object_detector, ssd_object_detector, yolo_object_detector
    # We will obtain the parameters.

    if args.model == 'faster':
        print("Loading Faster RCNN torch object detector")
        object_detector = faster_rcnn_object_detector.Faster_RCNN_Object_Detector(model=args.model_origin)
    if args.model == 'ssd':
        print("Loading SSD torch object detector")
        object_detector = ssd_object_detector.SSD_Object_Detector(model=args.model_origin)
    if args.model == 'yolo':
        print("Loading YOLO torch object detector")
        object_detector = yolo_object_detector.YOLO_Object_Detector(model=args.model_origin)

    print(f"Input video: {args.video}")
    print(f"Input file format: {args.input_files_format}")
    print(f"Using transformations: {args.transformations}")
    print(f"Output results json will be written in {args.output}")
    if args.print_output_folder: print(f"Output images will be saved in {args.print_output_folder}")

    read_from_video = True                                                                                   # Do we read from video?
    if not os.path.isfile(args.video):
        print(f"{args.video} is not file.")
        read_from_video = False

    if not read_from_video and not args.input_files_format:
        print("ERROR: There is no input.")
        quit()

    if read_from_video and args.input_files_format:
        print("WARNING: There are video and files input. Video input will be used")
    
    # We will generate a list of transformation and untransform point functions.
    if args.transformations:
        transform_images_functions = []
        untransform_point_functions = []
        for trans_name in args.transformations:
            if trans_name == "flipH":
                t,u = test_time_augmentation.create_flip_transformation("horizontal")
            if trans_name == "flipV":
                t,u = test_time_augmentation.create_flip_transformation("vertical")
            if trans_name == "rot90":
                t,u = test_time_augmentation.create_rotation_transformation(90)
            if trans_name == "rot270":
                t,u = test_time_augmentation.create_rotation_transformation(270)
            if trans_name == "None":
                t,u = None, None

            transform_images_functions.append(t)
            untransform_point_functions.append(u)

    if not os.path.isdir(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    if args.print_output_folder:                                                                # De we print outputs?
        for trans_name in args.transformations:
            if not os.path.isdir(os.path.join(args.print_output_folder,trans_name)):            # We will ensure the existence of the output folder.
                os.makedirs(os.path.join(args.print_output_folder,trans_name))

    coco_annotations = coco_format_utils.Coco_Annotation_Set()

    # We need to initialize the image input generator.
    if read_from_video:
        image_generator = generator_from_video(args.video)

    else:
        image_generator = generator_from_files_input_format(args.input_files_format)

    # Main loop
    image_index = 1
    obj_id = 1
    process_times_list = []
    initial_process_time = time.time()
    print(f"Start video process.")

    for image in image_generator:                                   # While there is a next image.

        frame_filename = "frame_{:0>6}.png".format(image_index)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(rgb_image.shape)
        images_batch = []                          # We create the images batch.

        initial_frame_time = time.time()

        if args.transformations:                            # We apply transformations if there is any.
            for trans in transform_images_functions:
                if trans:
                    images_batch.append(trans(rgb_image))
                else:
                    images_batch.append(rgb_image)

        preprocessed_images = object_detector.preprocess(images_batch)                  # We preprocess the batch.
        outputs = object_detector.process(preprocessed_images)                          # We apply the model.
        outputs = object_detector.filter_output_by_confidence_treshold(outputs, treshold = 0.5)         # We filter output using confidence.
        #print(outputs)
        
        if args.print_output_folder:
            for img_output, img_rgb, trans_name in zip(outputs, images_batch, args.transformations):                             # For outputs from each image...
                drawn_image = print_utils.print_detections_on_image(img_output, img_rgb[:,:,[2,1,0]])
                cv2.imwrite(os.path.join(args.print_output_folder, trans_name, frame_filename), drawn_image)

        untransformed_outputs = []
        for untransform_point_function, output in zip(untransform_point_functions,outputs):
            if untransform_point_function:
                _output = test_time_augmentation.untransform_coco_format_object_information(output, untransform_point_function, rgb_image.shape)
            else:
                _output = output
            untransformed_outputs.append(_output)

        clusters_objects = test_time_augmentation.create_clusters(untransformed_outputs,0.5)

        unified_output = test_time_augmentation.unify_clusters(clusters_objects)

        # Now we create the coco format annotation for this image.
        for bbox, _class, confidence in zip(unified_output[0], unified_output[1], unified_output[2]):   
            coco_object = coco_format_utils.Coco_Annotation_Object(bbox=bbox, category_id=_class, id=obj_id, image_id=image_index, score=confidence)
            coco_annotations.insert_coco_annotation_object(coco_object)
            obj_id+=1

        if args.print_output_folder:
            drawn_image = print_utils.print_detections_on_image(unified_output, rgb_image[:,:,[2,1,0]])
            cv2.imwrite(os.path.join(args.print_output_folder, frame_filename), drawn_image)

        process_time = time.time()-initial_frame_time
        process_times_list.append(process_time)

        image_index += 1

    print(f"Total process time {time.time()-initial_process_time}, average time per frame {np.array(process_times_list).mean()}")

    coco_annotations.to_json(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to execute object detection to some video.')
    parser.add_argument('-m', '--model', help="Object Detection model to use. Current options: 'faster'")
    parser.add_argument('-v', '--video',default=None, help="Path to video.")
    parser.add_argument('-t', '--transformations', nargs = '+', default=None, help= "Transformations set. Expected list as follows: 'flipH flipV rot90'")
    parser.add_argument('-o', '--output', default='../output/annotation.json', help= 'Path to output file to write obtained annotations as coco')
    parser.add_argument('-p', '--print_output_folder', default = None, help='We print output if not None. Example: ../output/images')
    parser.add_argument('-f', '--input_files_format', default = None, help='Files format to read from compatibles with glob. If args.video is not None, video will be used instead of this.')
    parser.add_argument('-b', '--backend', default = "torch", help='Deep Learning backend to use. IT could be "torch" or "tf".')
    parser.add_argument('-mo', '--model_origin', default = "default", help='"default" to load the predefined model.')

    args = parser.parse_args()
    main(args)
