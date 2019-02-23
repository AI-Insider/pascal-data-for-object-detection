#This code can be used to extract and preprocess the data from the Pascal VOC dataset.
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pickle

def get_class_vectors(class_names):
    class_vectors = [np.zeros((len(class_names))) for i in range(len(class_names))]
    class_vectors = [np.insert(class_vectors[i],i,1) for i in range(len(class_names))]
    class_vectors = [np.delete(class_vectors[i],-1) for i in range(len(class_names))]

    return class_vectors

def get_data_from_files(folder,image_size):
    filenames = os.listdir(folder)
    object_lists = []
    object_names = []
    print("Getting data from XML-files...")
    for filename in tqdm(filenames):
        path = os.path.join(folder,filename)


        objects_in_image = []
        tree = ET.parse(path)
        root = tree.getroot()
        size_element = root.find("size")
        original_image_size = (int(size_element.find("width").text),int(size_element.find("height").text))
        objects_element = root.findall("object")
        ratio = (original_image_size[0]/image_size[0],original_image_size[1]/image_size[1])
        for object_element in objects_element:
            bndbox_element = object_element.find("bndbox")
            name_element = object_element.find("name")

            object= {"name":name_element.text,"box":[float(bndbox_element.find("xmin").text),float(bndbox_element.find("ymin").text),float(bndbox_element.find("xmax").text),float(bndbox_element.find("ymax").text)]}
            object["box"][0] =object["box"][0]/ ratio[0]
            object["box"][1] = object["box"][1]/ ratio[1]
            object["box"][2] = object["box"][2]/ ratio[0]
            object["box"][3] =  object["box"][3]/ ratio[1]

            objects_in_image.append(object)
            object_names.append(object["name"])
        object_lists.append(objects_in_image)
    return object_lists,object_names

def get_resized_images(folder,image_size):
    filenames = os.listdir(folder)
    images = []
    print("Loading and resizing images...")
    for filename in tqdm(filenames):
        path = os.path.join(folder,filename)
        image = Image.open(path)
        image = image.resize(image_size,Image.ANTIALIAS)
        images.append(np.asarray(image))

    return images

def get_targets(folder,image_size,cells):

    object_lists, object_names = get_data_from_files(folder,image_size)
    class_names = list(set(object_names))
    class_vectors = get_class_vectors(class_names)

    targets = []

    print("Creating target tensors.")

    for object_list in tqdm(object_lists):
        target = np.zeros((cells[0],cells[1],4+len(class_names)))

        for object in object_list:
            object_name = object["name"]
            bounding_box = object["box"]
            class_index= [i for i in range(len(class_names)) if class_names[i]==object_name][0]
            class_vector = class_vectors[class_index]

            object_center = ((bounding_box[0]+bounding_box[2])/2,(bounding_box[1]+bounding_box[3])/2)
            column = object_center[0]//(image_size[0]/cells[0])
            row = object_center[1]//(image_size[1]/cells[1])

            cell_center_x = ((column*(image_size[0]/cells[0]))+((column+1)*(image_size[0]/cells[0])))/2
            cell_center_y = ((row*(image_size[1]/cells[1]))+((row+1)*(image_size[1]/cells[1])))/2

            distance_x = object_center[0]-cell_center_x
            distance_y = object_center[1]-cell_center_y

            width = bounding_box[2]-bounding_box[0]
            height = bounding_box[3]-bounding_box[1]

            distance_x /= image_size[0]
            distance_y /= image_size[1]
            width /= image_size[0]
            height /= image_size[1]




            values = [distance_x,distance_y,width,height]
            cell_values = target[int(column),int(row)]


            for i in range(len(class_names)-1):
                cell_values[i] = class_vector[i]
            for i in range(len(class_names),24):
                cell_values[i] = values[i-len(class_names)]
            print(cell_values)
            target[int(column),int(row)] = cell_values



        targets.append(target)
    with open("classnames.pickle","wb") as file:
        pickle.dump(class_names,file)

    return targets

#Image size and cells are tuples. (width, height) and (columns,rows)
#This function uses above functions to extract data and save it in .pickle files. One file = one training batch.
def save_batch_files(image_size,cells,batch_size):
    images = get_resized_images("dataset/images",image_size)
    targets =get_targets("dataset/annotations",image_size,cells)
    image_index = 0
    print("Creating batch files")
    for batch_index in tqdm(range(len(images)//batch_size)):
        batch_images = images[image_index:image_index+batch_size]
        batch_targets =targets[image_index:image_index+batch_size]
        image_index+=batch_size

        with open("processed_data/batch_{0}.pickle".format(batch_index),"wb") as file:
            pickle.dump({"images":np.asarray(batch_images),"targets":np.asarray(batch_targets)},file)
