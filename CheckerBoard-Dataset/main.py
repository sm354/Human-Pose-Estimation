import cocoapi.PythonAPI.pycocotools.mask as mask
from cocoapi.PythonAPI.pycocotools.coco import COCO
import argparse
import numpy as np
import PIL.Image as Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("--patterns_path", required = True, help = "Address of Checkerboard Pattern images")
parser.add_argument("--annotations_path", required = True, help = "Address of annotations.json")
parser.add_argument("--images_path", required = True, help = "Address of coco train images")
parser.add_argument("--dataset_path", required = True, help = "Address to save Checkerboard Dataset")
args = vars(parser.parse_args())

patterns_path = args['patterns_path']
patterns = os.listdir(patterns_path)
number = -1

if not os.path.exists(args['dataset_path']):
        os.mkdir(args['dataset_path'])
        os.mkdir(args['dataset_path']+'/Images')
        os.mkdir(args['dataset_path']+'/Labels')

annFile=args['annotations_path']
coco=COCO(annFile) # COCO is a class in COCO API

# get annotations for all images that belong to super category-person
# supNms = ['person','animal','vehicle','furniture','kitchen']
# ants = coco.loadAnns(coco.getAnnIds(catIds=coco.getCatIds(supNms=supNms),iscrowd=False,areaRng=[15000,50000]))
# print("number of annotations for supercategory person = {}".format(len(ants)))

ants = coco.loadAnns(coco.getAnnIds(iscrowd=False,areaRng=[15000,50000]))
print("number of annotations = {}; mask area range = [{}, {}]\n".format(len(ants),15000,50000))
error_ants = 0 # number of error annotations
exception_count = 0
image_count = 0

for i, ant in enumerate(ants): # ant : annotation
        image_id = ant['image_id']
        segmentation = ant['segmentation']

        if i%500 == 0: #each pattern will be pasted on 116 images : total imgs = 60*500 ~ 30000
                number += 1 

        image_name = coco.loadImgs([image_id])[0]['file_name']
        pattern_name = patterns[number]
        image = Image.open(os.path.join(args['images_path'], image_name))
        pattern = (Image.open(patterns_path+pattern_name)).resize(image.size)
        binary_mask = Image.fromarray(255*coco.annToMask(ant))

        if image.size != binary_mask.size:
                print("encountered unequal image and mask size : image id {}, annotation id:{}".format(image_id,ant['id']))
                error_ants += 1
                continue

        image.paste(pattern, mask = binary_mask)
        image.save(os.path.join(args['dataset_path'], 'Images', str(i)+'.jpg'))
        binary_mask.save(os.path.join(args['dataset_path'], 'Labels', str(i)+'.jpg'))
        image_count+=1

print("\ntotal annotations observed = {}".format(i))
print("total images(with labels) created = {}".format(image_count))
print("error anntations encountered = {}".format(error_ants))
print("exceptions encountered = {}".format(exception_count))
