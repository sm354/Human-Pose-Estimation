import numpy as np
import torch
import matplotlib.pyplot as plt
from Unet import UNet
import torchvision.transforms as transforms
from CheckerBoard import GridDataset
import random

img_train = np.load('./coco/Train/img.npy')
img_test = np.load('./coco/Test/img.npy')

lbl_train = np.load('./coco/Train/lbl.npy')
lbl_test = np.load('./coco/Test/lbl.npy')

model = UNet(in_channels = 3, n_classes = 1, padding = True)

model.load_state_dict(torch.load('./lr.01ep2000.pth', map_location = 'cpu'))
input_im = GridDataset('Train')
input_lb = GridDataset('Test')

input_im.imgs = img_train
input_im.lbls = lbl_train

train_loader=torch.utils.data.DataLoader(input_im,batch_size=1,shuffle=False)

count = 0
img_count = 25
'''
for batch_idx, (data, target) in enumerate(train_loader):
    if(img_count == 25):
        break
    count += 1
    if(count % 1000 != 0):
        continue
    img_count += 1
    img = img_train[count - 1]
    lbl = lbl_train[count - 1]
    #a = tr(k)
    #print(a)

    with torch.no_grad():
        output = model(data)
    
    output = output * 255
    output = np.array(output, dtype = np.uint8)
    print("plotting...")
    plt.figure(figsize = (15,15))
    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(img, cmap = 'gray')
    axarr[0,0].set_title('Image')
    axarr[0,1].imshow(lbl, cmap = 'gray')
    axarr[0,1].set_title('Expected')
    axarr[1,0].imshow(output[0][0], cmap = 'gray')
    axarr[1,0].set_title('Prediction')
    axarr[0,0].axis('off')
    axarr[0,1].axis('off')
    axarr[1,0].axis('off')
    plt.axis('off')
    print("Done...")
    plt.savefig(f'./imgs/img_{img_count}.png')
'''

input_im.imgs = img_test
input_im.lbls = lbl_test
train_loader=torch.utils.data.DataLoader(input_im,batch_size=1,shuffle=False)

count = 0

for batch_idx, (data, target) in enumerate(train_loader):
    if(img_count == 50):
        break
    count += 1
    if(count % 600 != 0):
        continue
    img_count += 1
    img = img_test[count - 1]
    lbl = lbl_test[count - 1]
    #a = tr(k)
    #print(a)

    with torch.no_grad():
        output = model(data)
    
    output = output * 255
    output = np.array(output, dtype = np.uint8)
    print("plotting...")
    plt.figure(figsize = (15,15))
    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(img, cmap = 'gray')
    axarr[0,0].set_title('Image')
    axarr[0,1].imshow(lbl, cmap = 'gray')
    axarr[0,1].set_title('Expected')
    axarr[1,0].imshow(output[0][0], cmap = 'gray')
    axarr[1,0].set_title('Prediction')
    axarr[0,0].axis('off')
    axarr[0,1].axis('off')
    axarr[1,0].axis('off')
    plt.axis('off')
    print("Done...")
    plt.savefig(f'./imgs/img_{img_count}.png')




'''
for i in range(25):
    a = random.randint(0, len(img_train)-1)

    img = img_train[a]
    lbl = lbl_train[a]
    k = np.array([img])
    tr = transforms.ToTensor()
    batch_idx, (data, target) = train_loader[a]
    #a = tr(k)
    #print(a)

    with torch.no_grad():
        output = model(data)
    
    output = output * 255
    output = np.array(output, dtype = uint8)
    print("plotting...")
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img, cmap = 'gray')
    axarr[0,1].imshow(lbl, cmap = 'gray')
    axarr[1,0].imshow(output, cmap = 'gray')
    print("Done...")
    plt.savefig(f'./imgs/img_{i}.png')
    

'''
