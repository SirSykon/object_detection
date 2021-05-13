import argparse
import coco_evaluation_utils

parser = argparse.ArgumentParser(description='Script to execute the coco evaluation for object detection')
parser.add_argument('-gt', '--ground_truth', help="Path to groundtruth .json annotations file.")
parser.add_argument('-r', '--results', help="Path to results .json annotations file.")
args = parser.parse_args()

coco_evaluation_utils.evaluate(args.ground_truth, args.results)