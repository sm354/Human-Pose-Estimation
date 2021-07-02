import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
class GridDataset(Dataset):
# Downsampling 4 times => 16 times downsampled img in unet 


# 1. Data stored as numpy array
# 2. Data extracted as normalised tensors [ ToTensor() does the normalisation ] 
# Interchaning between these forms is done automatically:-
# - tensor dim - ch,h,w
# - PIL.Image dim - w,h,ch
# - np.array dim - h,w,ch

# Now images directly read when getItem called. : DROP THIS SINCE COCO HAS BOTH RGB & B/W IMGS
# image names are indices
# COCO has both b/w and rgb images
# In numpy, only rgb stored

    def __init__(self,type_, address):
        s=os.getcwd()
        Images_path=address + f"/{type_}_img.npy"
        Labels_path=address + f"/{type_}_lbl.npy"
        if os.path.isfile(Images_path) and os.path.isfile(Labels_path):
            self.imgs=np.load(Images_path)
            #self.imgs = np.divide(self.imgs, 255)
            self.imgs = self.imgs.astype(np.float16)
            self.lbls=np.load(Labels_path)
            self.lbls = self.lbls.astype(np.float16)
            #self.lbls[self.lbls < 128] = 0
            #self.lbls[self.lbls > 128] = 1
            #print(self.lbls)
            #print(self.lbls)
            print("Data Successfully Loaded into CPU")
        else:
            Images_path=address + "/Images/"
            Labels_path=address + "/Labels/"
            Imgs=os.listdir(Images_path)
            Lbls=os.listdir(Labels_path)
            #Imgs.remove(".DS_Store")
            self.imgs=[]
            self.lbls=[]

            if(type_ == "Train"):
                Imgs = Imgs[:int(0.7*len(Imgs))]
                Lbls = Lbls[:int(0.7*len(Lbls))]
            elif(type_ == "Val"):
                Imgs = Imgs[int(0.7*len(Imgs)):int(0.9*len(Imgs))]
                Lbls = Lbls[int(0.7*len(Lbls)):int(0.9*len(Lbls))]
            else:
                Imgs = Imgs[int(0.9*len(Imgs)):]
                Lbls = Lbls[int(0.9*len(Lbls)):]

            #print(Imgs)
            #i = 0
            #print(len(Imgs[0]))
            for img in Imgs:
                #if(type_ == "Train"):
                im=Image.open(Images_path+img)
                lb=Image.open(Labels_path+img) # label and image name same
                    #print(Labels_path+"mask"+img[5:])
                    #print(Labels_path + img[:-4] + "_predict" + img[-4:])
                #print(im.size)
                im=np.array(im.resize((128,128)))
                lb=np.array(lb.resize((128,128)))
                    #self.imgs.append(im)
	                #self.lbls.append(lb)
                
                #print(im.shape)
                if im.shape == (128,128,3) or im.shape == (128,128,1):
                    #print("here", im.shape)
                    self.imgs.append(im)
                    self.lbls.append(lb)
                #print(i)

            #print(len(self.imgs))
            self.imgs=np.array(self.imgs)
            self.lbls=np.array(self.lbls)
            #self.imgs = np.divide(self.imgs, 255)
            #self.lbls = np.divide(self.lbls, 140).astype("uint8")
            #print(self.imgs)
            #print(self.lbls)
            print("{}: size = {}, imgs[0] shape= {}".format(type_,self.imgs.shape[0], self.imgs[0].shape))
            np.save(address + f"/{type_}_img.npy", self.imgs)
            np.save(address + f"/{type_}_lbl.npy", self.lbls)
            
        '''
        self.imgs_path = s+'/Coco/DS1/'+type_+'/Images/'
        self.lbls_path = s+'/Coco/DS1/'+type_+'/Labels/'
        self.imgs = os.listdir(self.imgs_path)
        self.lbls = os.listdir(self.lbls_path)
        '''

    def __getitem__(self,index):
        '''
        there are imgs of different sizes in the coco
        if model not restricted to one img size, then not possible to load more than 1 img in a batch
        since tensor size must be restricted (n, ch, h, w)
        if img.size > (480,640):
            img=img.resize((480,640))
            lbl=lbl.resize((480,640))
        else:
            w,h=img.size
            w=16*(w//16)
            h=16*(h//16)
            img=img.resize((w,h))
            lbl=lbl.resize((w,h))
        Thus restricting to 480,640
        '''
        # img=Image.open(self.imgs_path+str(self.imgs[index]))
        # lbl=Image.open(self.lbls_path+str(self.lbls[index]))
        # img=img.resize((480,640))
        # lbl=lbl.resize((480,640))
        #print(self.imgs[index])
        img=Image.fromarray(self.imgs[index].astype(np.uint8))
        lbl=Image.fromarray(self.lbls[index].astype(np.uint8))

        tr=transforms.ToTensor() # implicit: pixel value in [0,1]
        img_t = tr(img) #float 32 tensor
        lbl_t = tr(lbl) #float 32 tensor
        return img_t, lbl_t

    def __len__(self):
        return len(self.imgs)
