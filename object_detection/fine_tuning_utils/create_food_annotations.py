import pandas as pd
import os
import cv2
import argparse
import numpy as np
from shutil import copyfile
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, parse, ElementTree

def create_annotations(data_annotations_path, image_folder, output_annotations_folder, output_image_folder, labelmap_path, number_of_classes_to_get, reduce_factor):
    # We turn the .xls into a pandas Dataframe.
    df = pd.read_excel(io=data_annotations_path,
                    sheet_name='Hoja3',
                    names=['Image', 'Meal', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

    MEAL_LIST = list(df.Meal.unique())
    meal_count = df['Meal'].value_counts()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(meal_count[:number_of_classes_to_get])
        MEALS_TO_USE = list(meal_count[:number_of_classes_to_get].index)

    row = df.loc[df['Meal'].isin(MEALS_TO_USE)]

    for index, meal in enumerate(MEALS_TO_USE):
        item = {"id":index+1, "name":meal}
        print(item)
        with open(labelmap_path, 'a') as f:
            s = "\nitem {\n    id: "+str(index+1)+"\n    name: '"+meal+"'\n}"
            f.write(s)

    for index, (_, row) in enumerate(row.iterrows()):

        # We get the xml annotation path.
        xml_path = os.path.join(output_annotations_folder, row.Image + ".xml")

        # Does the .xm exists?
        if not os.path.exists(xml_path):
            print(f"{xml_path} not found.")
            # We get the image path.
            image_path = os.path.join(image_folder,row.Image + ".jpg")

            # We load the image and add it to the list.
            image = cv2.imread(image_path)
            if reduce_factor > 1:
                new_dim = (int(image.shape[1]/reduce_factor), int(image.shape[0]/reduce_factor))
                image = cv2.resize(image, new_dim)
            cv2.imwrite(os.path.join(output_image_folder,row.Image + ".jpg"), image)
            #copyfile(image_path, os.path.join(output_image_folder,row.Image + ".jpg"))
            # We get height and width to normalize.
            h, w, c = image.shape

            ann = Element('annotation')
            folder = SubElement(ann, "folder")
            folder.text = "images"
            filename = SubElement(ann, "filename")
            filename.text = os.path.basename(image_path)
            path = SubElement(ann, "path")
            path.text = os.path.abspath(image_path)

            source = SubElement(ann, "source")
            database = SubElement(source, "database")
            database.text = "Unknown"
            size = SubElement(ann, "size")
            width = SubElement(size, "width")
            width.text = str(w)
            height = SubElement(size, "height")
            height.text = str(h)
            depth = SubElement(size, "depth")
            depth.text = str(c)

            segmented = SubElement(ann, "segmented")
            segmented.text = str(0)

        else:
            print(f"{xml_path} found.")
            xml_tree = parse(xml_path)
            root = xml_tree.getroot()

            ann = root

        class_label_id = MEAL_LIST.index(row.Meal) + 1

        # We get bounding box corner coordinates.
        listX = [row.x1, row.x2, row.x3, row.x4]
        listY = [row.y1, row.y2, row.y3, row.y4]
        
        bbox_xmin = int(min(listX)/reduce_factor)
        bbox_ymin = int(min(listY)/reduce_factor)
        bbox_xmax = int(max(listX)/reduce_factor)
        bbox_ymax = int(max(listY)/reduce_factor)

        obj = SubElement(ann, "object")
        name = SubElement(obj, "name")
        name.text = str(row.Meal)
        pose = SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = SubElement(obj, "truncated")
        truncated.text = str(0)
        difficult = SubElement(obj, "difficult")
        difficult.text = str(0)
        bndbox = SubElement(obj, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(bbox_xmin)
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(bbox_ymin)
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(bbox_xmax)
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(bbox_ymax)

        tree = ElementTree(ann)
        tree.write(xml_path)

        """
        child = SubElement(top, 'child')
        child.text = 'This child contains text.'

        child_with_tail = SubElement(top, 'child_with_tail')
        child_with_tail.text = 'This child has regular text.'
        child_with_tail.tail = 'And "tail" text.'

        child_with_entity_ref = SubElement(top, 'child_with_entity_ref')
        child_with_entity_ref.text = 'This & that'
        """
    
def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Generate xml files.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--annotations_path',
        help='Path to .xls with annotation information.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-i', '--image_dir',
        help='Path to the folder where the image dataset is stored.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-o', '--output_dir',
        help='Path to the output folder where xml are saved.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-n', '--number_of_classes',
        help='The nummber of classes to take image of. If given number is lesser than 73 (default), the classes with most objects will be selected.',
        default=73,
        type=int)
    parser.add_argument(
        '-l', '--labelmap_path',
        help='The path to save the label map. If not provided, will be saved in image_dir.',
        default=None,
        type=str)
        
    parser.add_argument(
        '-r', '--reduce_factor',
        help='Factor to reduce image size.',
        default=1,
        type=int)


    args = parser.parse_args()
    
    if args.annotations_path is None:
        print("ERROR. No .xls with annotations provided.")
        quit()

    if args.image_dir is None:
        print("ERROR. No image folder provided.")
        quit()
        
    if args.output_dir is None:
        print("ERROR. No output folder provided.")
        quit()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.labelmap_path is None:
        args.labelmap_path = os.path.join(args.output_dir, "label_map.pbtxt")

    create_annotations(args.annotations_path, args.image_dir, args.output_dir, args.output_dir, args.labelmap_path, args.number_of_classes, args.reduce_factor)

if __name__ == '__main__':
    main()
