import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import os

class CheckerBoard(Dataset):
"""
Given the path to the raw images or to numpy files, it creates Dataset 

About the checkerboard dataset made from COCO
    COCO has both b/w and rgb images
    Data stored as numpy array. In numpy, only rgb stored
    Image names are indices

Data extracted as normalised tensors 
    ToTensor() does the normalisation
    Interchaning between these forms is done automatically:-
        - tensor dim - ch,h,w
        - PIL.Image dim - w,h,ch
        - np.array dim - h,w,ch 

Since Coco has both RGB and B/W imgs, dropping this idea:
    Now images directly read when getItem called.

Parameters
----------
    type_ : string
        type of the Dataset i.e. 'Train', 'Val', or 'Test'
    address : string
        data directory
    size : int
        fixed size of the images used for DL models
        note: models like UNet downsamples the image by factor of 16, hence size to be chosen wisely

"""
    def __init__(self,type_, address, size=128):
        s=os.getcwd()
        Images_path=address + f"/{type_}_img.npy"
        Labels_path=address + f"/{type_}_lbl.npy"
        if os.path.isfile(Images_path) and os.path.isfile(Labels_path):
            self.imgs=np.load(Images_path)
            self.lbls=np.load(Labels_path)
            
            self.imgs = self.imgs.astype(np.float16)
            self.lbls = self.lbls.astype(np.float16)
            #self.lbls[self.lbls < 128] = 0
            #self.lbls[self.lbls > 128] = 1
            #print(self.lbls)
            #print(self.lbls)

        else:
            Images_path=address + "/Images/"
            Labels_path=address + "/Labels/"
            Imgs=os.listdir(Images_path)
            Lbls=os.listdir(Labels_path)
            #Imgs.remove(".DS_Store")
            self.imgs=[]
            self.lbls=[]

            temp = list(zip(Imgs, Lbls))
            random.shuffle(temp)
            Imgs, Lbls = zip(*temp)
            
            tri = Imgs[:int(0.7*len(Imgs))]
            trl = Lbls[:int(0.7*len(Lbls))]
        
            vai = Imgs[int(0.7*len(Imgs)):int(0.9*len(Imgs))]
            val = Lbls[int(0.7*len(Lbls)):int(0.9*len(Lbls))]
        
            tei = Imgs[int(0.9*len(Imgs)):]
            tel = Lbls[int(0.9*len(Lbls)):]

            val_i = []
            val_l = []
            test_i = []
            test_l = []

            for img in tri:

                im=Image.open(Images_path+img)
                lb=Image.open(Labels_path+img)

                im=np.array(im.resize((size,size)))
                lb=np.array(lb.resize((size,size)))
                
                if im.shape == (size,size,3) or im.shape == (size,size,1):
                    self.imgs.append(im)
                    self.lbls.append(lb)
            
            for img in vai:
                im=Image.open(Images_path+img)
                lb=Image.open(Labels_path+img) 

                im=np.array(im.resize((size,size)))
                lb=np.array(lb.resize((size,size)))
                    
                if im.shape == (size,size,3) or im.shape == (size,size,1):
                    val_i.append(im)
                    val_l.append(lb)
            
            for img in tei:
                im=Image.open(Images_path+img)
                lb=Image.open(Labels_path+img) 

                im=np.array(im.resize((size,size)))
                lb=np.array(lb.resize((size,size)))
                
                if im.shape == (size,size,3) or im.shape == (size,size,1):
                    test_i.append(im)
                    test_l.append(lb)

            self.imgs=np.array(self.imgs)
            self.lbls=np.array(self.lbls)
            val_i = np.array(val_i)
            val_l = np.array(val_l)
            test_i = np.array(test_i)
            test_l = np.array(test_l)

            print("{}: size = {}, imgs[0] shape= {}".format(type_,self.imgs.shape[0], self.imgs[0].shape))
            np.save(address + f"/Train_img.npy", self.imgs)
            np.save(address + f"/Train_lbl.npy", self.lbls)
            np.save(address + f"/Val_img.npy", val_i)
            np.save(address + f"/Val_lbl.npy", val_l)
            np.save(address + f"/Test_img.npy", test_i)
            np.save(address + f"/Test_lbl.npy", test_l)

    def __getitem__(self,index):
        img=Image.fromarray(self.imgs[index].astype(np.uint8))
        lbl=Image.fromarray(self.lbls[index].astype(np.uint8))

        tr=transforms.ToTensor() # implicit: pixel value in [0,1]
        img_t = tr(img) #float 32 tensor
        lbl_t = tr(lbl) #float 32 tensor
        return img_t, lbl_t

    def __len__(self):
        return len(self.imgs)
