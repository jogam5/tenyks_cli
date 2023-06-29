import shutil
import os
import json
from PIL import Image

def copy_from_to(file_path_from, file_path_to):
    shutil.copy(file_path_from, file_path_to)

def copy_subset_of_kitti(input_folder_path, output_folder_path, idx_begin='000000', idx_end='001000'):
    # Improvements:
    # 1. Verify there is a pair (i.e. img: 0001.jpg, lab: 0001.txt) for each file
    for file_name in os.listdir(input_folder_path):
        if idx_begin <= file_name <= idx_end:
            copy_from_to(os.path.join(input_folder_path, file_name), output_folder_path)
    print('Subset of Kitti copied!')
    
def make_dir(new_dir_path):
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    print(f'{new_dir_path} created')
    
def image_size(img_path):
    with Image.open(img_path) as img:
        image_w, image_h = img.size
    #print(f"The image size is {image_w} x {image_h}")
    return image_w, image_h

def get_key(my_dict, value):
    '''Given a dict, get a key from a value'''
    for k, v in my_dict.items():
        if v == value:
            return k
    return None

def box_from_yolo_to_coco(coordinates_list, img_width, img_height):
    yolo_x, yolo_y, yolo_width, yolo_height =  [float(x) for x in coordinates_list] 
    x_min = round((yolo_x - yolo_width / 2) * img_width,2)
    y_min = round((yolo_y - yolo_height / 2) * img_height,2)
    x_max = round((yolo_x + yolo_width / 2) * img_width,2)
    y_max = round((yolo_y + yolo_height / 2) * img_height,2)
    coco_x, coco_y, coco_width, coco_height = x_min, y_min, x_max - x_min, y_max - y_min
    return coco_x, coco_y, coco_width, coco_height

import os
import yaml

def read_classes_from_yaml(dataset_path, destination_path):
    # Find the YAML file in the dataset folder
    yaml_file = None
    for file in os.listdir(dataset_path):
        if file.endswith('.yml') or file.endswith('.yaml'):
            yaml_file = file
            break

    if yaml_file is None:
        raise FileNotFoundError("No YAML file found in the dataset folder.")

    yaml_path = os.path.join(dataset_path, yaml_file)

    # Read the YAML file and extract the names
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        if 'names' in data:

            classes_dict = {}
            # Iterate over the list and assign indices as values
            for index, name in enumerate(data['names']):
                classes_dict[name] = index
                
            return classes_dict
        else:
            raise KeyError("No 'names' key found in the YAML file.")

def create_categories_file(names_dict, destination_folder):
    data = {
        "categories": list(names_dict.keys())
    }
    with open(f'{destination_folder}/class.json', 'w') as file:
        json.dump(data, file)

def get_annotations(images_path, anns_path, destination_folder, dataset_path):
    # Roboflow map of classes
    #labels_keys_ = {'DontCare': 2, 'Car': 0, 'Pedestrian': 4, 'Cyclist': 1, 'Truck': 7,
    #                'Misc': 3, 'Tram': 6, 'Person_sitting': 5, 'Van': 8}
    #labels_keys_ = {'Apple': 0}
    labels_keys_ = read_classes_from_yaml(dataset_path)
    create_categories_file(labels_keys_, destination_folder)

    json_dict = {}
    json_dict['images'] = []
    json_dict['annotations'] = []
    json_dict['categories'] = []
    image_id = 0
    annotation_id = 0

    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            image_dict = {}
            image_dict['id'] = image_id
            image_dict['file_name'] = filename
            image = Image.open(os.path.join(images_path, filename))
            image_dict['width'], image_dict['height'] = image.size
            json_dict['images'].append(image_dict)

            with open(os.path.join(anns_path, filename[:-4] + '.txt'), 'r') as f:
                for line in f:
                    line = line.split(' ')
                    category_id = int(line[0])
                    bbox = box_from_yolo_to_coco(line[1:5], image.size[0], image.size[1])
                    area = bbox[2] * bbox[3]
                    annotation_dict = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'segmentation': [],
                        'area': area,
                        'bbox': bbox,
                        'iscrowd': 0
                    }
                    json_dict['annotations'].append(annotation_dict)
                    annotation_id += 1

            image_id += 1

    for i in range(len(labels_keys_)):
        json_dict['categories'].append({'id': i, 'name': get_key(labels_keys_, i)})

    with open(f'{destination_folder}/_anns_tenyks.json', 'w') as f:
        json.dump(json_dict, f)
        
        
def get_predictions(images_path, preds_path, destination_folder, dataset_path):
    #images_path='/Users/gabriel/python/tenyks/converters/yolo_v7/images'
    #labels_path='/Users/gabriel/python/tenyks/converters/yolo_v7/predictions'
    
    #labels_keys_ = {'DontCare': 2, 'Car': 0, 'Pedestrian': 4, 'Cyclist': 1, 'Truck': 7,
    #                'Misc': 3, 'Tram': 6, 'Person_sitting': 5, 'Van': 8}
    
    #labels_keys_ = {'Apple': 0}
    labels_keys_ = read_classes_from_yaml(dataset_path)

    json_dict = {}
    json_dict['images'] = []
    json_dict['predictions'] = []
    json_dict['categories'] = []
    image_id = 0
    annotation_id = 0

    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            image_dict = {}
            image_dict['id'] = image_id
            image_dict['file_name'] = filename
            image = Image.open(os.path.join(images_path, filename))
            image_dict['width'], image_dict['height'] = image.size
            json_dict['images'].append(image_dict)

            with open(os.path.join(preds_path, filename[:-4] + '.txt'), 'r') as f:
                for line in f:
                    line = line.split(' ')
                    category_id = int(line[0])
                    bbox = box_from_yolo_to_coco(line[1:5], image.size[0], image.size[1])
                    area = bbox[2] * bbox[3]
                    annotation_dict = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'segmentation': [],
                        'area': area,
                        'bbox': bbox,
                        'iscrowd': 0,
                        'score': float(line[5])
                    }
                    json_dict['predictions'].append(annotation_dict)
                    annotation_id += 1

            image_id += 1

    for i in range(len(labels_keys_)):
        json_dict['categories'].append({'id': i, 'name': get_key(labels_keys_, i)})

    with open(f'{destination_folder}/_preds_tenyks.json', 'w') as f:
        json.dump(json_dict, f)
        
def count_files_in_dir(dir_path):
    ii = 0
    for file in os.listdir(dir_path):
        ii += 1
    print(ii)      