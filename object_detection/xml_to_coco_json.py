import os
from glob import glob
import argparse
import cv2
import xml.etree.ElementTree as ET
from coco_format_utils import Coco_Annotation_Set, Coco_Annotation_Object

def read_label_map(label_map_path):

    item_id = None
    item_name = None
    categories = []
    category = {}
    category_name_to_category_id_relation_dict = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line=line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id:" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()
            if item_id is not None and item_name is not None:
                category_name_to_category_id_relation_dict[item_name] = item_id
                category['id'] = item_id
                category['name'] = item_name
                category['supercategory'] = ""
                categories.append(category)
                category = {}
                item_id = None
                item_name = None

    return categories, category_name_to_category_id_relation_dict

def insert_images_into_coco_ann_obj(images_format, coco_annotations):

    img_name_relation_to_img_id = {}
    for img_idx, image_path in enumerate(sorted(glob(images_format))):
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        coco_annotations.insert_image(img_idx+1, width, height, file_name = filename)
        img_name_relation_to_img_id[filename] = img_idx+1

    return coco_annotations, img_name_relation_to_img_id

def xml_to_coco_ann_obj(path, coco_annotations, img_name_relation_to_img_id, category_name_to_category_id_relation_dict):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines them in a single Pandas datagrame.
    Parameters:
    ----------
    path : {str}
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    obj_counter = 0
    for xml_file in sorted(glob(path + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            img_filename = root.find('filename').text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            class_name = member[0].text

            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)

            bbox_xmin = xmin
            bbox_ymin = ymin
            bbox_width = xmax-xmin
            bbox_height = ymax-ymin

            class_id = category_name_to_category_id_relation_dict[class_name]
            image_id = img_name_relation_to_img_id[img_filename]
            coco_object = Coco_Annotation_Object(bbox=[bbox_xmin,bbox_ymin,bbox_width,bbox_height], category_id=class_id, id=obj_counter, image_id=image_id, area=bbox_width*bbox_height, score=1)
            coco_annotations.insert_coco_annotation_object(coco_object)

            obj_counter += 1
    
    return coco_annotations


def main(args):
    if args.xml_input_folder is None or not os.path.isdir(args.xml_input_folder):
        print("ERROR: No input folder to get .xml from.")
        quit()
    if args.xml_input_folder is None:
        args.xml_input_folder = args.xml_input_folder
        print(f"WARNING: No output path provided. {args.output_path} will be used.")
    if args.output_path is None:
        args.output_path = os.path.join(args.xml_input_folder, "annotations.json")
        print(f"WARNING: No output path provided. {args.output_path} will be used.")

    categories, category_name_to_category_id_relation_dict = read_label_map(args.labelmap_path)
    coco_annotations = Coco_Annotation_Set(categories=categories)
    coco_annotations, img_name_relation_to_img_id = insert_images_into_coco_ann_obj(args.images_input_format, coco_annotations)
    coco_annotations = xml_to_coco_ann_obj(args.xml_input_folder, coco_annotations, img_name_relation_to_img_id, category_name_to_category_id_relation_dict)
    coco_annotations.to_json(args.output_path, only_annotations = False, only_essential_data = False)


if __name__ == '__main__':

    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="TensorFlow XML to coco json converter")
    parser.add_argument("-xi",
                        "--xml_input_folder",
                        help="Path to the folder where the input .xml files are stored",
                        default=None,
                        type=str)
    parser.add_argument("-ii",
                        "--images_input_format",
                        help="Path to the format to find the dataset images.",
                        default=None,
                        type=str)
    parser.add_argument("-o",
                        "--output_path",
                        help="Name of output .json file (including path)", default=None, type=str)
    parser.add_argument('-l', 
                        '--labelmap_path',
                        help='The path to save the label map. If not provided, will be saved in image_dir.',
                        default=None,
                        type=str)
    args = parser.parse_args()

    main(args)
