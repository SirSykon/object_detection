import pandas as pd
import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, parse, ElementTree

DATA_ANOTATIONS_PATH = "../old/output.xls"
IMAGE_FOLDER = "../images"
ANNOTATIONS_FOLDER = "../annotations"
LABELMAP_PATH = "../annotations/label_map.pbtxt"
# We turn the .xls into a pandas Dataframe.
df = pd.read_excel(io=DATA_ANOTATIONS_PATH,
                sheet_name='Hoja3',
                names=['Image', 'Meal', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

MEAL_LIST = list(df.Meal.unique())
row = df.loc[df['Meal'].isin(MEAL_LIST)]

for index, meal in enumerate(MEAL_LIST):
    item = {"id":index+1, "name":meal}
    print(item)
    with open(LABELMAP_PATH, 'a') as f:
        s = "\nitem {\n    id: "+str(index+1)+"\n    name: '"+meal+"'\n}"
        f.write(s)

for index, (_, row) in enumerate(row.iterrows()):

    # We get the xml annotation path.
    xml_path = os.path.join(ANNOTATIONS_FOLDER, row.Image + ".xml")

    # Does the .xm exists?
    if not os.path.exists(xml_path):
        print(f"{xml_path} not found.")
        # We get the image path.
        image_path = os.path.join(IMAGE_FOLDER,row.Image + ".jpg")

        # We load the image and add it to the list.
        image = cv2.imread(image_path)

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

    bbox = [min(listX), min(listY), max(listX), max(listY)]

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
    xmin.text = str(min(listX))
    ymin = SubElement(bndbox, "ymin")
    ymin.text = str(min(listY))
    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(max(listX))
    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(max(listY))

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