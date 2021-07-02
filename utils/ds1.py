import mask
from coco import COCO
import numpy as np
import PIL.Image as Image
import os

patterns_path = 'Filtered/'
patterns = os.listdir(patterns_path)
number = -1

annFile='annotations.json'
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

        #try:
        image_name = coco.loadImgs([image_id])[0]['file_name']
        pattern_name = patterns[number]
        image = Image.open('./images/'+image_name)
        pattern = (Image.open(patterns_path+pattern_name)).resize(image.size)
        binary_mask = Image.fromarray(255*coco.annToMask(ant))

        if image.size != binary_mask.size:
                print("encountered unequal image and mask size : image id {}, annotation id:{}".format(image_id,ant['id']))
                error_ants += 1
                continue


        # binary_mask.save('label.jpg')
        image.paste(pattern, mask = binary_mask)
        # image.save('image.jpg')
        image.save('./Data_Filtered/Images/'+str(i)+'.jpg')
        binary_mask.save('./Data_Filtered/Labels/'+str(i)+'.jpg')
        image_count+=1
        print("success")

        '''except Exception as e:
                print("Exception '{}' occured for imageId, annotationId - {}, {}".format(e, image_id, ant['id']))
                exception_count += 1
                pass
'''
print("\ntotal annotations observed = {}".format(i))
print("total images(with labels) created = {}".format(image_count))
print("error anntations encountered = {}".format(error_ants))
print("exceptions encountered = {}".format(exception_count))
