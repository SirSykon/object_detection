import argparse
import os
import cv2
import time
import numpy as np
from glob import glob

import print_utils
from coco_utils.coco_format_utils import Coco_Annotation_Set, Coco_Annotation_Object
from object_detectors.object_detector import Object_Detector
from utils.read_utils import *

def main(args):

    object_detector = Object_Detector(args.backend, args.model, args.transformations, args.model_origin)

    print(f"Input video: {args.video}")
    print(f"Input file format: {args.input_files_format}")
    print(f"Using transformations: {args.transformations}")
    print(f"Output results json will be written in {args.output}")
    if args.print_output_folder: print(f"Output images will be saved in {args.print_output_folder}")

    read_from_video = True                                      # Do we read from video?
    if args.video is None or not os.path.isfile(args.video):
        print(f"{args.video} is not file.")
        read_from_video = False

    if not read_from_video and not args.input_files_format:
        print("ERROR: There is no input.")
        quit()

    if read_from_video and args.input_files_format:
        print("WARNING: There are video and files input. Video input will be used")

    if not os.path.isdir(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    if args.print_output_folder:                                                                # De we print outputs?
        for trans_name in args.transformations:
            if not os.path.isdir(os.path.join(args.print_output_folder,trans_name)):            # We will ensure the existence of the output folder.
                os.makedirs(os.path.join(args.print_output_folder,trans_name))

    coco_annotations = Coco_Annotation_Set()

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
    if read_from_video:
        print("Start video process.")
    else:
        print("Starting image process.")

    for image in image_generator:                                   # While there is a next image.

        frame_filename = "frame_{:0>6}.png".format(image_index)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        initial_frame_time = time.time()

        output = object_detector.process_single_image(rgb_image)

        # Now we create the coco format annotation for this image.
        for bbox, _class, confidence in zip(output[0], output[1], output[2]):   
            coco_object = Coco_Annotation_Object(bbox=bbox, category_id=_class, id=obj_id, image_id=image_index, score=confidence)
            coco_annotations.insert_coco_annotation_object(coco_object)
            obj_id+=1

        if args.print_output_folder:
            drawn_image = print_utils.print_detections_on_image(output, rgb_image[:,:,[2,1,0]])
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
