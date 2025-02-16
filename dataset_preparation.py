from glob import glob
from bs4 import BeautifulSoup
import os
from distutils.dir_util import copy_tree
from sklearn.model_selection import train_test_split
import yaml


labels_dir = list(sorted(glob("datasets/dog-and-cat-detection/annotations/*.xml")))

text_labels_dir = 'datasets/dog-and-cat-detection/labels'
os.makedirs(text_labels_dir, exist_ok = True)

for labels in labels_dir : 
    with open(labels, 'r') as f :
        data = f.read()
        soup = BeautifulSoup(data, 'xml')

        img_size = soup.find('size')
        img_width = int(img_size.find('width').text)
        img_height = int(img_size.find('height').text)

        objects = soup.find_all('object')
        obj_list = []
        class_lambda = lambda x : 0 if x == 'cat' else 1
        for obj in objects :
            label = class_lambda(obj.find('name').text)
            xmin = int(obj.find('xmin').text)
            ymin = int(obj.find('ymin').text)
            xmax = int(obj.find('xmax').text)
            ymax = int(obj.find('ymax').text)

            x = ((xmin + xmax) / 2) / img_width
            y = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            obj_list.append([label, x, y, width, height])
        
        txt_label_dir = text_labels_dir + '/' + labels[46:-4] + '.txt'
        with open(txt_label_dir, 'w') as f :
            for obj in obj_list :
                f.write(str(obj[0]) + ' ' +\
                        str(obj[1]) + ' ' +\
                        str(obj[2]) + ' ' +\
                        str(obj[3]) + ' ' +\
                        str(obj[4]))
                
imgs_dir = 'datasets/dog-and-cat-detection/images'

img_list = glob(imgs_dir + '/*.png')
train_img, valid_img = train_test_split(img_list, test_size = 0.1, random_state = 0)

# create images and labels directories in train and valid directories
os.makedirs('datasets/dog-and-cat-detection/train/images', exist_ok = True)
os.makedirs('datasets/dog-and-cat-detection/train/labels', exist_ok = True)
os.makedirs('datasets/dog-and-cat-detection/valid/images', exist_ok = True)
os.makedirs('datasets/dog-and-cat-detection/valid/labels', exist_ok = True)


for img in train_img :
    os.rename(img, 'datasets/dog-and-cat-detection/train/images/' + img[41:])
    os.rename(text_labels_dir + '/' + img[41:-4] + '.txt', 'datasets/dog-and-cat-detection/train/labels/' + img[41:-4] + '.txt')
for img in valid_img :
    os.rename(img, 'datasets/dog-and-cat-detection/valid/images/' + img[41:])
    os.rename(text_labels_dir + '/' + img[41:-4] + '.txt', 'datasets/dog-and-cat-detection/valid/labels/' + img[41:-4] + '.txt')

with open('data/data.yaml', 'w') as f:
    data = {
        'train' : 'datasets/dog-and-cat-detection/train/',
        'val' : 'datasets/dog-and-cat-detection/valid/',
        'nc' : 2,
        'names' : ['cat', 'dog']
    }
    yaml.dump(data, f)